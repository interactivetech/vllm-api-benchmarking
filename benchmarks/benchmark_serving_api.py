# SPDX-License-Identifier: Apache-2.0
r"""Benchmark online serving throughput.
... (docstring remains the same) ...
"""
import argparse
import asyncio
import gc
import json
import logging # Added
import os
import random
import time
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
from backend_request_func_v2 import (ASYNC_REQUEST_FUNCS,
                                  OPENAI_COMPATIBLE_BACKENDS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    # logger.warning("Could not import get_tokenizer from vllm, using local version.") # Logged later
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    # logger.warning("Could not import FlexibleArgumentParser from vllm, using standard ArgumentParser.") # Logged later
    from argparse import ArgumentParser as FlexibleArgumentParser

from benchmark_dataset import (AIMODataset, BurstGPTDataset,
                               ConversationDataset, HuggingFaceDataset,
                               InstructCoderDataset, RandomDataset,
                               SampleRequest, ShareGPTDataset, SonnetDataset,
                               VisionArenaDataset)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json

# Added logger instance
logger = logging.getLogger(__name__)

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]



async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampleRequest, None]:
    # ... (function remains the same) ...
    input_requests_iter: Iterable[SampleRequest] = iter(input_requests) # Renamed for clarity
    logger.info("Request generator started. Rate: %.2f req/s, Burstiness: %.2f", request_rate, burstiness) # Added log

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}.")
    theta = 1.0 / (request_rate * burstiness) if request_rate != float("inf") else 0.0 # Handle inf rate

    request_idx = 0
    for request in input_requests_iter:
        # logger.debug("Yielding request %d", request_idx) # Can be verbose
        yield request
        request_idx += 1

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            # logger.debug("Infinite request rate, no sleep.") # Can be verbose
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        # logger.debug("Sleeping for %.4f seconds before next request.", interval) # Can be verbose
        await asyncio.sleep(interval)
    logger.info("Request generator finished after yielding %d requests.", request_idx) # Added log


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    logger.info("Calculating benchmark metrics...") # Added log
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    failed_requests = 0 # Added counter

    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if output_len is None:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                # logger.debug("Output length is None, tokenizing generated text for request %d", i) # Can be verbose
                tokenized_output = tokenizer(outputs[i].generated_text,
                                             add_special_tokens=False).input_ids
                output_len = len(tokenized_output)
                # logger.debug("Tokenized output length: %d", output_len) # Can be verbose

            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            # Ensure latency and ttft are valid numbers before calculation
            if output_len > 1 and isinstance(outputs[i].latency, (int, float)) and isinstance(outputs[i].ttft, (int, float)) and outputs[i].latency >= outputs[i].ttft:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
                # logger.debug("Request %d: TPOT calculated: %.4f ms", i, tpot * 1000)
            elif output_len <= 1:
                 logger.debug("Request %d: TPOT not calculated (output_len=%d <= 1)", i, output_len)
            else:
                 logger.warning("Request %d: Invalid latency (%.4f) or TTFT (%.4f) for TPOT calculation. Output len: %d",
                                i, outputs[i].latency, outputs[i].ttft, output_len)


            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            if isinstance(outputs[i].ttft, (int, float)): # Check type before appending
                 ttfts.append(outputs[i].ttft)
            else:
                 logger.warning("Request %d: Invalid TTFT value type: %s", i, type(outputs[i].ttft))
            if isinstance(outputs[i].latency, (int, float)): # Check type before appending
                e2els.append(outputs[i].latency)
            else:
                logger.warning("Request %d: Invalid latency value type: %s", i, type(outputs[i].latency))

            completed += 1
        else:
            failed_requests += 1
            actual_output_lens.append(0)
            all_tpots.append(0) # Append 0 tpot for failed requests for goodput check len consistency
            logger.warning("Request %d failed. Error: %s", i, outputs[i].error) # Added log for failed requests

    if goodput_config_dict:
        logger.info("Calculating goodput based on SLOs: %s", goodput_config_dict) # Added log
        valid_metrics_data = [] # Store the actual data lists
        slo_values = []
        metric_names_for_goodput = [] # Store names for logging

        # Prepare data and SLOs, ensuring data lists are not empty
        if "ttft" in goodput_config_dict and ttfts:
            valid_metrics_data.append(ttfts)
            slo_values.append(goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION)
            metric_names_for_goodput.append("ttft")
        if "tpot" in goodput_config_dict and all_tpots: # Use all_tpots which includes 0 for failed/short requests
            valid_metrics_data.append(all_tpots)
            slo_values.append(goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION)
            metric_names_for_goodput.append("tpot")
        if "e2el" in goodput_config_dict and e2els:
            valid_metrics_data.append(e2els)
            slo_values.append(goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION)
            metric_names_for_goodput.append("e2el")

        logger.info("Metrics used for goodput calculation: %s", metric_names_for_goodput)

        if valid_metrics_data and slo_values: # Only calculate if we have metrics and SLOs
            # We need to iterate through requests, not just metrics directly
            # This assumes ttfts, all_tpots, e2els correspond to the *successful* requests
            # Let's adjust: Check SLOs only for successful requests
            success_indices = [i for i, out in enumerate(outputs) if out.success]
            success_ttfts = [ttfts[idx] for idx, i in enumerate(success_indices) if i < len(ttfts)] if "ttft" in metric_names_for_goodput else []
            success_tpots = [all_tpots[i] for i in success_indices if i < len(all_tpots)] if "tpot" in metric_names_for_goodput else []
            success_e2els = [e2els[idx] for idx, i in enumerate(success_indices) if i < len(e2els)] if "e2el" in metric_names_for_goodput else []

            # Rebuild valid_metrics based on successful requests and configured SLOs
            successful_metrics_for_goodput = []
            if "ttft" in metric_names_for_goodput and success_ttfts: successful_metrics_for_goodput.append(success_ttfts)
            if "tpot" in metric_names_for_goodput and success_tpots: successful_metrics_for_goodput.append(success_tpots)
            if "e2el" in metric_names_for_goodput and success_e2els: successful_metrics_for_goodput.append(success_e2els)


            if successful_metrics_for_goodput:
                num_good_candidates = len(successful_metrics_for_goodput[0])
                logger.info("Checking SLOs for %d successful requests.", num_good_candidates)
                for i, req_metric_tuple in enumerate(zip(*successful_metrics_for_goodput)):
                    is_good_req = all([(r is not None and s >= r) for s, r in zip(slo_values, req_metric_tuple)])
                    if is_good_req:
                        good_completed += 1
                        # logger.debug("Request (original index %d) met goodput criteria.", success_indices[i])
                    # else:
                        # logger.debug("Request (original index %d) did NOT meet goodput criteria. Values: %s vs SLOs: %s",
                        #              success_indices[i], req_metric_tuple, slo_values)

            logger.info("Goodput calculation complete. Good requests: %d", good_completed)
        else:
            logger.warning("Could not calculate goodput. No valid metrics or SLOs provided/matched.")


    if completed == 0:
        logger.error("All requests failed. Cannot calculate meaningful metrics.") # Changed to error
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
        # Return default metrics object to avoid downstream errors
        return BenchmarkMetrics(completed=0, total_input=0, total_output=0, request_throughput=0, request_goodput=0,
                                output_throughput=0, total_token_throughput=0, mean_ttft_ms=0, median_ttft_ms=0,
                                std_ttft_ms=0, percentiles_ttft_ms=[], mean_tpot_ms=0, median_tpot_ms=0, std_tpot_ms=0,
                                percentiles_tpot_ms=[], mean_itl_ms=0, median_itl_ms=0, std_itl_ms=0, percentiles_itl_ms=[],
                                mean_e2el_ms=0, median_e2el_ms=0, std_e2el_ms=0, percentiles_e2el_ms=[]), actual_output_lens

    # Calculate stats only if lists are not empty
    mean_ttft_ms = np.mean(ttfts) * 1000 if ttfts else 0
    std_ttft_ms = np.std(ttfts) * 1000 if ttfts else 0
    median_ttft_ms = np.median(ttfts) * 1000 if ttfts else 0
    percentiles_ttft_ms = [(p, np.percentile(ttfts, p) * 1000) for p in selected_percentiles] if ttfts else []

    mean_tpot_ms = np.mean(tpots) * 1000 if tpots else 0
    std_tpot_ms = np.std(tpots) * 1000 if tpots else 0
    median_tpot_ms = np.median(tpots) * 1000 if tpots else 0
    percentiles_tpot_ms = [(p, np.percentile(tpots, p) * 1000) for p in selected_percentiles] if tpots else []

    mean_itl_ms = np.mean(itls) * 1000 if itls else 0
    std_itl_ms = np.std(itls) * 1000 if itls else 0
    median_itl_ms = np.median(itls) * 1000 if itls else 0
    percentiles_itl_ms = [(p, np.percentile(itls, p) * 1000) for p in selected_percentiles] if itls else []

    mean_e2el_ms = np.mean(e2els) * 1000 if e2els else 0
    std_e2el_ms = np.std(e2els) * 1000 if e2els else 0
    median_e2el_ms = np.median(e2els) * 1000 if e2els else 0
    percentiles_e2el_ms = [(p, np.percentile(e2els, p) * 1000) for p in selected_percentiles] if e2els else []

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s if goodput_config_dict else 0.0, # Handle case where goodput wasn't calculated
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=mean_ttft_ms,
        std_ttft_ms=std_ttft_ms,
        median_ttft_ms=median_ttft_ms,
        percentiles_ttft_ms=percentiles_ttft_ms,
        mean_tpot_ms=mean_tpot_ms,
        std_tpot_ms=std_tpot_ms,
        median_tpot_ms=median_tpot_ms,
        percentiles_tpot_ms=percentiles_tpot_ms,
        mean_itl_ms=mean_itl_ms,
        std_itl_ms=std_itl_ms,
        median_itl_ms=median_itl_ms,
        percentiles_itl_ms=percentiles_itl_ms,
        mean_e2el_ms=mean_e2el_ms,
        std_e2el_ms=std_e2el_ms,
        median_e2el_ms=median_e2el_ms,
        percentiles_e2el_ms=percentiles_e2el_ms,
    )
    logger.info("Metrics calculation finished. Completed requests: %d, Failed requests: %d", completed, failed_requests) # Added log
    logger.info("Throughput - Req: %.2f, GoodReq: %.2f, OutputTok: %.2f, TotalTok: %.2f",
                metrics.request_throughput, metrics.request_goodput, metrics.output_throughput, metrics.total_token_throughput) # Added log

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    logprobs: Optional[int],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: Optional[int],
    lora_modules: Optional[Iterable[str]],
    api_key: Optional[str], # Added api_key parameter
    extra_body: Optional[dict],
):
    logger.info("Starting benchmark function for backend: %s", backend) # Added log
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
        logger.info("Using request function: %s", request_func.__name__) # Added log
    else:
        logger.error("Unknown backend: %s", backend) # Added log
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    logger.info("Starting initial single prompt test run...") # Added log
    if not input_requests:
         logger.error("Input requests list is empty. Cannot run benchmark.")
         raise ValueError("Input requests list is empty.")

    test_prompt, test_prompt_len, test_output_len, test_mm_content = \
        input_requests[0].prompt, input_requests[0].prompt_len, \
        input_requests[0].expected_output_len, \
            input_requests[0].multi_modal_data

    if backend != "openai-chat" and test_mm_content is not None:
        err_msg = "Multi-modal content is only supported on 'openai-chat' backend."
        logger.error(err_msg + f" Current backend: {backend}") # Added log
        raise ValueError(err_msg)
    assert test_mm_content is None or isinstance(test_mm_content, dict)

    # Ensure extra_body is a dict if None
    extra_body = extra_body or {}

    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
        api_key=api_key, # Pass api_key
        extra_body=extra_body,
    )
    logger.info("Test request created for URL: %s", test_input.api_url) # Added log

    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        err_msg = ("Initial test run failed - Please make sure benchmark arguments "
                   f"are correctly specified. Error: {test_output.error}")
        logger.error(err_msg) # Added log
        raise ValueError(err_msg)
    else:
        logger.info("Initial test run completed successfully.") # Added log
        print("Initial test run completed. Starting main benchmark run...")

    if lora_modules:
        logger.info("Using LoRA modules: %s", lora_modules) # Added log
        # For each input request, choose a LoRA module at random.
        lora_modules_list = list(lora_modules) # Convert to list for random.choice
        chosen_loras = [random.choice(lora_modules_list) for _ in range(len(input_requests))]
        lora_iterator = iter(chosen_loras) # Create iterator

    if profile:
        print("Starting profiler...")
        logger.info("Starting profiler...") # Added log
        profile_start_url = base_url + "/start_profile"
        logger.info("Sending profile start request to: %s", profile_start_url) # Added log
        profile_input = RequestFuncInput(model=model_id,
                                         model_name=model_name,
                                         prompt=test_prompt,
                                         api_url=profile_start_url, # Correct URL
                                         prompt_len=test_prompt_len,
                                         output_len=test_output_len,
                                         logprobs=logprobs,
                                         multi_modal_content=test_mm_content,
                                         ignore_eos=ignore_eos,
                                         api_key=api_key, # Pass api_key
                                         extra_body=extra_body)
        # Use a non-OpenAI func if possible, or handle potential incompatibility
        # Assuming a generic func exists or the endpoint handles this GET/POST
        # Let's assume completions endpoint works for profile commands too for simplicity here
        # Ideally, we'd have a specific simple async request function
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            logger.info("Profiler started successfully.") # Added log
            print("Profiler started")
        else:
             logger.error("Failed to start profiler: %s", profile_output.error) # Added log
             print(f"Warning: Failed to start profiler: {profile_output.error}")


    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")
    logger.info("Traffic - Rate: %.2f, Burstiness: %.2f (%s), Max Concurrency: %s",
                request_rate, burstiness, distribution, max_concurrency) # Added log


    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = (asyncio.Semaphore(max_concurrency)
                 if max_concurrency is not None and max_concurrency > 0 else None) # Check > 0
    if semaphore:
        logger.info("Using semaphore with max concurrency: %d", max_concurrency) # Added log

    async def limited_request_func(request_func_input, pbar, req_idx): # Added req_idx for logging
        task_name = f"request-{req_idx}"
        # logger.debug("Task %s waiting for semaphore...", task_name) # Can be verbose
        if semaphore is None:
            # logger.debug("Task %s proceeding without semaphore.", task_name) # Can be verbose
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)
        async with semaphore:
            # logger.debug("Task %s acquired semaphore, executing request.", task_name) # Can be verbose
            result = await request_func(request_func_input=request_func_input,
                                      pbar=pbar)
            # logger.debug("Task %s finished request, releasing semaphore.", task_name) # Can be verbose
            return result

    logger.info("Starting main benchmark loop with %d requests.", len(input_requests)) # Added log
    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    req_counter = 0
    async for request in get_request(input_requests, request_rate, burstiness):
        prompt, prompt_len, output_len, mm_content = request.prompt, \
            request.prompt_len, request.expected_output_len, \
                request.multi_modal_data
        req_model_id, req_model_name = model_id, model_name
        if lora_modules:
            try:
                req_lora_module = next(lora_iterator)
                req_model_id, req_model_name = req_lora_module, req_lora_module
                logger.debug("Request %d using LoRA: %s", req_counter, req_lora_module) # Added log
            except StopIteration:
                 logger.error("Ran out of LoRA modules in iterator, reusing last one or default.")
                 # Handle potential error or reuse last one if needed

        request_func_input = RequestFuncInput(model=req_model_id,
                                              model_name=req_model_name,
                                              prompt=prompt,
                                              api_url=api_url,
                                              prompt_len=prompt_len,
                                              output_len=output_len,
                                              logprobs=logprobs,
                                              multi_modal_content=mm_content,
                                              ignore_eos=ignore_eos,
                                              api_key=api_key, # Pass api_key
                                              extra_body=extra_body)
        # logger.debug("Creating task for request %d", req_counter) # Can be verbose
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input,
                                     pbar=pbar,
                                     req_idx=req_counter), # Pass req_idx
                name=f"request-task-{req_counter}" # Name task
                ))
        req_counter += 1

    logger.info("All %d request tasks created. Waiting for completion...", len(tasks)) # Added log
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    logger.info("All request tasks finished.") # Added log

    if profile:
        print("Stopping profiler...")
        logger.info("Stopping profiler...") # Added log
        profile_stop_url = base_url + "/stop_profile"
        logger.info("Sending profile stop request to: %s", profile_stop_url) # Added log
        profile_input = RequestFuncInput(
            model=model_id, # model/prompt likely don't matter for stop
            model_name=model_name,
            prompt=test_prompt,
            api_url=profile_stop_url, # Correct URL
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            api_key=api_key, # Pass api_key
            extra_body=extra_body, # Pass extra_body
        )
        # Use same assumption as start profile about request func
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            logger.info("Profiler stopped successfully.") # Added log
            print("Profiler stopped")
        else:
            logger.error("Failed to stop profiler: %s", profile_output.error) # Added log
            print(f"Warning: Failed to stop profiler: {profile_output.error}")


    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time
    logger.info("Benchmark loop finished. Duration: %.4f seconds.", benchmark_duration) # Added log

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    # --- Result Reporting ---
    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    if goodput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):",
                                        metrics.request_goodput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "failed": len(outputs) - metrics.completed, # Add failed count
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput, # Keep even if None/0
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs if output.success], # Only successful ttfts
        "itls": [itl for output in outputs if output.success for itl in output.itl], # Flatten successful itls
        "latencies": [output.latency for output in outputs if output.success], # Add successful latencies
        "generated_texts": [output.generated_text for output in outputs], # Keep all for inspection
        "errors": [output.error for output in outputs if not output.success], # Only errors from failures
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified metric.
        if metric_attribute_name not in selected_percentile_metrics:
            # logger.debug("Skipping percentile metric: %s", metric_attribute_name)
            return
        mean_val = getattr(metrics, f"mean_{metric_attribute_name}_ms")
        median_val = getattr(metrics, f"median_{metric_attribute_name}_ms")
        std_val = getattr(metrics, f"std_{metric_attribute_name}_ms")
        percentiles_val = getattr(metrics, f"percentiles_{metric_attribute_name}_ms")

        # Only print if values are non-zero (or if list is not empty)
        # This avoids printing headers for metrics that weren't calculated (e.g., TTFT if all failed)
        if mean_val != 0 or median_val != 0 or percentiles_val:
            print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
            print("{:<40} {:<10.2f}".format(f"Mean {metric_name} (ms):", mean_val))
            print("{:<40} {:<10.2f}".format(f"Median {metric_name} (ms):", median_val))
            result[f"mean_{metric_attribute_name}_ms"] = mean_val
            result[f"median_{metric_attribute_name}_ms"] = median_val
            result[f"std_{metric_attribute_name}_ms"] = std_val
            for p, value in percentiles_val:
                p_word = str(int(p)) if int(p) == p else str(p)
                print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                                value))
                result[f"p{p_word}_{metric_attribute_name}_ms"] = value
        else:
             logger.info("Skipping printing stats for metric '%s' as values are zero/empty.", metric_attribute_name)


    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)
    logger.info("Benchmark function finished.") # Added log

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        valid_config = True
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                logger.error("Invalid metric name for goodput SLO: %s. Valid names are: %s", slo_name, VALID_NAMES) # Added log
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. ")
            if slo_val < 0:
                logger.error("Invalid value for goodput SLO %s: %s. Value must be non-negative.", slo_name, slo_val) # Added log
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative.")
        if valid_config:
            logger.info("Parsed goodput SLO configuration: %s", goodput_config_dict) # Added log
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name.strip()] = float(slo_val.strip()) # Added strip
    except ValueError as err:
        logger.error("Error parsing goodput SLO argument '%s': %s", slo_pair, err) # Added log
        raise argparse.ArgumentTypeError(
            f"Invalid format found for service level objective: '{slo_pair}'. "
            "Specify service level objectives for goodput as \"KEY:VALUE\" "
            "pairs (e.g., ttft:100 tpot:50), where the key is a metric name "
            "(ttft, tpot, e2el), and the value is a number in milliseconds.") from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any],
                                     file_name: str) -> None:
    logger.info("Converting results to PyTorch benchmark format...") # Added log
    metrics = [
        "median_ttft_ms", "mean_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
        "median_itl_ms", "mean_itl_ms", "std_itl_ms", "p99_itl_ms"
    ]
    # Filter metrics that actually exist in the results
    available_metrics = {k: [results[k]] for k in metrics if k in results}

    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors", "latencies", "input_lens", "output_lens"]
    extra_info={
            k: results[k]
            for k in results if k not in available_metrics and k not in ignored_metrics
    }
    logger.debug("Metrics for PT format: %s", list(available_metrics.keys())) # Added log
    logger.debug("Extra info for PT format: %s", list(extra_info.keys())) # Added log

    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics=available_metrics,
        extra_info=extra_info
        )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        logger.info("Saving PyTorch benchmark format to: %s", pt_file) # Added log
        write_to_json(pt_file, pt_records)
    else:
        logger.warning("No PyTorch benchmark records generated.") # Added log


def main(args: argparse.Namespace):
    # --- Setup Logging ---
    log_level = logging.DEBUG if args.debug_logging else logging.INFO
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger.info("Logging configured at level: %s", logging.getLevelName(log_level)) # Added log

    # --- Log vLLM Import Status ---
    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer as vllm_get_tokenizer
        logger.info("Successfully imported get_tokenizer from vllm.")
    except ImportError:
        logger.warning("Could not import get_tokenizer from vllm, using local version from backend_request_func.")
    try:
        from vllm.utils import FlexibleArgumentParser as VllmFlexParser
        logger.info("Successfully imported FlexibleArgumentParser from vllm.")
    except ImportError:
        logger.warning("Could not import FlexibleArgumentParser from vllm, using standard ArgumentParser.")

    # --- Start Main Logic ---
    logger.info("Starting benchmark main function.") # Added log
    logger.info("Benchmark arguments: %s", args) # Added log
    random.seed(args.seed)
    np.random.seed(args.seed)
    logger.info("Set random seed to: %d", args.seed) # Added log

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name if args.served_model_name else model_id # Use model_id if served_model_name not provided
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        # Ensure base_url doesn't have a trailing slash if endpoint starts with one
        base_url = args.base_url.rstrip('/')
        # Ensure endpoint starts with a slash
        endpoint = args.endpoint if args.endpoint.startswith('/') else '/' + args.endpoint
        api_url = f"{base_url}{endpoint}"
        logger.info("Using base_url: %s and endpoint: %s", base_url, endpoint) # Added log
    else:
        base_url = f"http://{args.host}:{args.port}"
        endpoint = args.endpoint if args.endpoint.startswith('/') else '/' + args.endpoint
        api_url = f"{base_url}{endpoint}"
        logger.info("Using host: %s, port: %d, endpoint: %s", args.host, args.port, endpoint) # Added log
    logger.info("Constructed API URL: %s", api_url) # Added log
    logger.info("Constructed Base URL: %s", base_url) # Added log

    try:
        tokenizer = get_tokenizer(tokenizer_id,
                                  tokenizer_mode=tokenizer_mode,
                                  trust_remote_code=args.trust_remote_code)
        logger.info("Tokenizer loaded successfully.") # Added log
    except Exception as e:
        logger.exception("Failed to load tokenizer: %s", tokenizer_id) # Use logger.exception
        raise e # Re-raise after logging

    if args.dataset_name is None:
        err_msg = "Please specify '--dataset-name' and the corresponding '--dataset-path' if required."
        logger.error(err_msg) # Added log
        raise ValueError(err_msg)

    logger.info("Loading dataset '%s'...", args.dataset_name) # Added log
    input_requests : list[SampleRequest] = [] # Initialize

    try: # Wrap dataset loading in try-except
        if args.dataset_name == "sonnet":
            logger.info("Using Sonnet dataset from path: %s", args.dataset_path) # Added log
            dataset = SonnetDataset(dataset_path=args.dataset_path)
            # For the "sonnet" dataset, formatting depends on the backend.
            if args.backend == "openai-chat":
                logger.info("Formatting Sonnet for OpenAI Chat backend.") # Added log
                input_requests = dataset.sample(num_requests=args.num_prompts,
                                                input_len=args.sonnet_input_len,
                                                output_len=args.sonnet_output_len,
                                                prefix_len=args.sonnet_prefix_len,
                                                tokenizer=tokenizer,
                                                return_prompt_formatted=False)
            else:
                logger.info("Formatting Sonnet using tokenizer chat template.") # Added log
                if not (hasattr(tokenizer, 'chat_template') and tokenizer.chat_template) and \
                   not (hasattr(tokenizer, 'default_chat_template') and tokenizer.default_chat_template):
                    logger.error("Tokenizer/model must have chat template for sonnet dataset with non-chat backend.")
                    raise ValueError("Tokenizer/model must have chat template for sonnet dataset with non-chat backend.")

                input_requests = dataset.sample(num_requests=args.num_prompts,
                                                input_len=args.sonnet_input_len,
                                                output_len=args.sonnet_output_len,
                                                prefix_len=args.sonnet_prefix_len,
                                                tokenizer=tokenizer,
                                                return_prompt_formatted=True)

        elif args.dataset_name == "hf":
            logger.info("Using HuggingFace dataset loader for path: %s", args.dataset_path) # Added log
            # all following datasets are implemented from the
            # HuggingFaceDataset base class
            dataset_class = None
            if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
                dataset_class = VisionArenaDataset
                args.hf_split = "train" # Override split/subset for specific datasets
                args.hf_subset = None
                logger.info("Identified HF dataset as VisionArena.")
            elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
                dataset_class = InstructCoderDataset
                args.hf_split = "train"
                logger.info("Identified HF dataset as InstructCoder.")
            elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
                dataset_class = ConversationDataset
                logger.info("Identified HF dataset as Conversation.")
            elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
                dataset_class = AIMODataset
                args.hf_split = "train"
                logger.info("Identified HF dataset as AIMO.")
            else:
                supported_datasets = set([
                    dataset_name for cls in HuggingFaceDataset.__subclasses__()
                    for dataset_name in getattr(cls, 'SUPPORTED_DATASET_PATHS', []) # Use getattr for safety
                ])
                err_msg = (f"Unsupported dataset path for HF loader: {args.dataset_path}. "
                           f"Supported paths: {supported_datasets}. "
                            "Please consider contributing if you would "
                            "like to add support for additional dataset formats.")
                logger.error(err_msg) # Added log
                raise ValueError(err_msg)

            logger.info("Loading HF dataset '%s', subset '%s', split '%s'",
                        args.dataset_path, args.hf_subset, args.hf_split) # Added log
            input_requests = dataset_class(
                dataset_path=args.dataset_path,
                dataset_subset=args.hf_subset,
                dataset_split=args.hf_split,
                random_seed=args.seed,
            ).sample(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_len=args.hf_output_len,
            )

        else:
            # For datasets that follow a similar structure, use a mapping.
            dataset_mapping = {
                "sharegpt":
                lambda: ShareGPTDataset(random_seed=args.seed,
                                        dataset_path=args.dataset_path).sample(
                                            tokenizer=tokenizer,
                                            num_requests=args.num_prompts,
                                            output_len=args.sharegpt_output_len,
                                        ),
                "burstgpt":
                lambda: BurstGPTDataset(random_seed=args.seed,
                                        dataset_path=args.dataset_path).
                sample(tokenizer=tokenizer, num_requests=args.num_prompts),
                "random":
                lambda: RandomDataset(dataset_path=args.dataset_path).sample(
                    tokenizer=tokenizer,
                    num_requests=args.num_prompts,
                    prefix_len=args.random_prefix_len,
                    input_len=args.random_input_len,
                    output_len=args.random_output_len,
                    range_ratio=args.random_range_ratio,
                )
            }

            if args.dataset_name in dataset_mapping:
                 logger.info("Loading dataset '%s' from path '%s' using mapped function.", args.dataset_name, args.dataset_path) # Added log
                 input_requests = dataset_mapping[args.dataset_name]()
            else:
                 err_msg = f"Unknown dataset name: {args.dataset_name}"
                 logger.error(err_msg) # Added log
                 raise ValueError(err_msg)

        logger.info("Dataset loaded successfully. Number of prompts: %d", len(input_requests)) # Added log
        if not input_requests:
             logger.error("Dataset loading resulted in an empty list of prompts.") # Added log
             raise ValueError("Dataset loaded successfully, but no prompts were generated. Check dataset path and parameters.")

    except Exception as e:
         logger.exception("Failed to load dataset '%s' from path '%s'", args.dataset_name, args.dataset_path) # Use logger.exception
         raise e # Re-raise after logging


    goodput_config_dict = check_goodput_args(args)

    # Collect the sampling parameters.
    sampling_params = {
        k: v
        for k, v in {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature
        }.items() if v is not None
    }
    logger.info("Sampling parameters (extra_body): %s", sampling_params) # Added log

    # Sampling parameters are only supported by openai-compatible backend.
    if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
        err_msg = ("Sampling parameters are only supported by openai-compatible "
                   f"backends (e.g., vllm, openai, openai-chat). Current backend: {args.backend}")
        logger.error(err_msg) # Added log
        raise ValueError(err_msg)

    # Ensure temperature is set if not provided, default to greedy for OpenAI compat
    if args.backend in OPENAI_COMPATIBLE_BACKENDS and "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0  # Default to greedy decoding.
        logger.info("Defaulting temperature to 0.0 for OpenAI-compatible backend.") # Added log

    # Avoid GC processing "static" data - reduce pause times.
    logger.info("Running garbage collection and freezing...") # Added log
    gc.collect()
    gc.freeze()
    logger.info("Garbage collection and freeze complete.") # Added log

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
            lora_modules=args.lora_modules,
            api_key=args.api_key, # Pass the api_key argument
            extra_body=sampling_params,
        ))
    logger.info("Benchmark run completed.") # Added log

    # Save config and results to json
    if args.save_result:
        logger.info("Saving benchmark results...") # Added log
        result_json: dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["arguments"] = vars(args) # Save command line arguments
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["served_model_name"] = model_name # Save effective model name
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts_requested"] = args.num_prompts # Requested vs actual
        result_json["num_prompts_processed"] = len(input_requests) # Actual processed


        # Metadata
        if args.metadata:
            logger.info("Adding metadata: %s", args.metadata) # Added log
            result_json["metadata"] = {}
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=", 1) # Split only once
                    result_json["metadata"][kvstring[0].strip()] = kvstring[1].strip()
                else:
                    err_msg = f"Invalid metadata format: '{item}'. Please use KEY=VALUE format."
                    logger.error(err_msg) # Added log
                    raise ValueError(err_msg)

        # Traffic
        result_json["request_rate"] = (args.request_rate if args.request_rate
                                       < float("inf") else "inf")
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Determine filename
        base_model_id = model_id.split("/")[-1]
        rate_str = "inf" if args.request_rate == float("inf") else str(args.request_rate)
        max_concurrency_str = (f"-concurrency{args.max_concurrency}"
                               if args.max_concurrency is not None else "")
        default_file_name = f"{backend}-{rate_str}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json" #noqa

        if args.result_filename:
            file_name = args.result_filename
            logger.info("Using provided result filename: %s", file_name) # Added log
        else:
            file_name = default_file_name
            logger.info("Using default result filename: %s", file_name) # Added log

        if args.result_dir:
            if not os.path.exists(args.result_dir):
                logger.info("Creating result directory: %s", args.result_dir) # Added log
                os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)
            logger.info("Prepending result directory to filename: %s", file_name) # Added log

        if not args.save_detailed:
            logger.info("Removing detailed per-request data before saving.") # Added log
            # Remove fields with too many data points
            detailed_fields = [
                    "input_lens", "output_lens", "ttfts", "itls",
                    "generated_texts", "errors", "latencies" # Added latencies
            ]
            for field in detailed_fields:
                if field in result_json:
                    del result_json[field]
                    # logger.debug("Removed detailed field: %s", field) # Can be verbose


        logger.info("Writing final results JSON to: %s", file_name) # Added log
        try:
            with open(file_name, "w", encoding='utf-8') as outfile:
                json.dump(result_json, outfile, indent=4) # Added indent for readability
            logger.info("Results JSON saved successfully.") # Added log
        except IOError as e:
            logger.exception("Error writing results JSON file: %s", file_name) # Use logger.exception


        # Save PyTorch benchmark format if requested/possible
        try:
            save_to_pytorch_benchmark_format(args, result_json, file_name)
        except Exception as e:
             logger.exception("Error saving results in PyTorch benchmark format") # Use logger.exception
    else:
        logger.info("Skipping saving results (--save-result not specified).") # Added log

    logger.info("Benchmark main function finished.") # Added log


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    # --- Existing arguments ---
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url (e.g., http://localhost:8000 or https://your.api.com). If set, overrides --host and --port.", # Modified help
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host if not using --base-url.") # Modified help
    parser.add_argument("--port", type=int, default=8000, help="Server port if not using --base-url.") # Modified help
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint path (e.g., /v1/completions, /generate_stream). Will be appended to --base-url or http://host:port.", # Modified help
    )
    parser.add_argument(
        "--dataset-name", type=str, default=None, # Changed default to None
        choices=["sharegpt", "burstgpt", "sonnet", "random", "hf"],
        help="Name of the dataset format to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset file (for sharegpt, sonnet, random) "
                        "or HuggingFace dataset ID/path (for hf). Required if --dataset-name is set.") # Modified help
    parser.add_argument(
        "--max-concurrency", type=int, default=None,
        help="Maximum number of concurrent requests allowed by the client semaphore.") # Modified help
    parser.add_argument("--model", type=str, required=True, help="Name of the model (e.g., NousResearch/Hermes-3-Llama-3.1-8B). Used for payload and potentially tokenizer loading.") # Modified help
    parser.add_argument("--tokenizer", type=str, help="Name or path of the tokenizer, if different from --model.") # Modified help
    # parser.add_argument("--use-beam-search", action="store_true") # Beam search not directly supported by OpenAI API format
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process from the dataset.") # Modified help
    parser.add_argument("--logprobs", type=int, default=None, help="Number of logprobs per token to request (OpenAI-compatible backends).") # Modified help
    parser.add_argument("--request-rate", type=float, default=float("inf"), help="Target requests per second (float, 'inf' for immediate dispatch).") # Modified help
    parser.add_argument("--burstiness", type=float, default=1.0, help="Burstiness factor for request arrival (1.0=Poisson, <1 more bursty, >1 more uniform).") # Modified help
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset sampling and request arrival.") # Modified help
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code when loading tokenizer from HuggingFace.") # Modified help
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable the tqdm progress bar.") # Modified help
    parser.add_argument("--profile", action="store_true", help="Enable vLLM profiler start/stop requests (requires server support at /start_profile, /stop_profile).") # Modified help
    parser.add_argument("--save-result", action="store_true", help="Save benchmark results to a JSON file.") # Modified help
    parser.add_argument("--save-detailed", action="store_true", help="Include per-request details (TTFTs, ITLs, texts, etc.) in the saved JSON.") # Modified help
    parser.add_argument("--metadata", metavar="KEY=VALUE", nargs="*", help="Add key-value pairs to the result JSON file (e.g., --metadata env=staging test_id=123).") # Modified help
    parser.add_argument("--result-dir", type=str, default=None, help="Directory to save the benchmark result JSON file.") # Modified help
    parser.add_argument("--result-filename", type=str, default=None, help="Custom filename for the result JSON file (overrides default).") # Modified help
    parser.add_argument("--ignore-eos", action="store_true", help="Send 'ignore_eos: true' in the request payload (if supported by backend).") # Modified help
    parser.add_argument("--percentile-metrics", type=str, default="ttft,tpot,itl,e2el", help="Comma-separated metrics for percentile calculation (ttft,tpot,itl,e2el).") # Added e2el default
    parser.add_argument("--metric-percentiles", type=str, default="99", help="Comma-separated percentiles to calculate (e.g., 50,90,99).") # Modified help
    parser.add_argument("--goodput", nargs="+", required=False, help="Calculate request goodput based on SLOs (e.g., --goodput ttft:100 tpot:50 e2el:2000). Values in ms.") # Modified help

    # --- New Arguments ---
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authenticating requests. If not provided, tries 'OPENAI_API_KEY' environment variable for relevant backends.", # Modified help
    )
    parser.add_argument(
        "--debug-logging",
        action="store_true",
        help="Enable DEBUG level logging for verbose output.", # Added help
    )

    # --- Dataset specific groups ---
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    # ... (sonnet args remain the same) ...
    sonnet_group.add_argument("--sonnet-input-len", type=int, default=550)
    sonnet_group.add_argument("--sonnet-output-len", type=int, default=150)
    sonnet_group.add_argument("--sonnet-prefix-len", type=int, default=200)

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    # ... (sharegpt args remain the same) ...
    sharegpt_group.add_argument("--sharegpt-output-len", type=int, default=None)

    random_group = parser.add_argument_group("random dataset options")
    # ... (random args remain the same) ...
    random_group.add_argument("--random-input-len", type=int, default=1024)
    random_group.add_argument("--random-output-len", type=int, default=128)
    random_group.add_argument("--random-range-ratio", type=float, default=1.0)
    random_group.add_argument("--random-prefix-len", type=int, default=0)

    hf_group = parser.add_argument_group("hf dataset options")
    # ... (hf args remain the same) ...
    hf_group.add_argument("--hf-subset", type=str, default=None)
    hf_group.add_argument("--hf-split", type=str, default=None)
    hf_group.add_argument("--hf-output-len", type=int, default=None)

    # --- Sampling group ---
    sampling_group = parser.add_argument_group("sampling parameters (OpenAI-compatible backends)") # Modified help
    # ... (sampling args remain the same) ...
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--temperature", type=float, default=None) # Default handling logic moved to main()

    # --- Other arguments ---
    parser.add_argument('--tokenizer-mode', type=str, default="auto", choices=['auto', 'slow', 'mistral', 'custom'], help='Tokenizer loading mode.') # Modified help
    parser.add_argument("--served-model-name", type=str, default=None, help="Model name expected by the API in the payload (if different from --model).") # Modified help
    parser.add_argument("--lora-modules", nargs='+', default=None, help="List of LoRA module names (passed to server on launch) to randomly assign to requests.") # Modified help

    args = parser.parse_args()

    # --- Argument Validation (Optional but Recommended) ---
    if args.dataset_name and not args.dataset_path:
        parser.error("--dataset-path is required when --dataset-name is specified.")
    if args.base_url and (args.host != "127.0.0.1" or args.port != 8000):
         print("Warning: --base-url is set, ignoring --host and --port.")
         # logger not configured yet, use print
    if args.api_key and "OPENAI_API_KEY" in os.environ:
         print("Warning: Both --api-key argument and OPENAI_API_KEY environment variable are set. The --api-key argument will be used.")
         # logger not configured yet, use print


    main(args)