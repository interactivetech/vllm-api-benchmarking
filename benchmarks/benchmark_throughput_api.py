# benchmark_api.py
# SPDX-License-Identifier: Apache-2.0
"""Benchmark an OpenAI-compatible API endpoint with enhanced logging."""
import argparse
import asyncio
import json
import logging # Added
import os
import random
import sys # Added for flushing stdout
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from benchmark_dataset import (AIMODataset, BurstGPTDataset,
                               ConversationDataset, InstructCoderDataset,
                               RandomDataset, SampleRequest, ShareGPTDataset,
                               SonnetDataset, VisionArenaDataset)
# Assume benchmark_utils contains these functions, otherwise define them inline or import
# from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.utils import FlexibleArgumentParser

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---


# --- Mock benchmark_utils if not available ---
# Define dummy functions if benchmark_utils is not in the same directory or path
try:
    from benchmark_utils import (convert_to_pytorch_benchmark_format,
                                 write_to_json)
except ImportError:
    logger.warning("benchmark_utils not found. Using dummy functions.")

    def write_to_json(filename: str, data: Any):
        """Writes data to a JSON file."""
        try:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            logger.error(f"Error writing to {filename}: {e}")

    def convert_to_pytorch_benchmark_format(*args, **kwargs):
        """Dummy function for PyTorch benchmark format."""
        logger.info("Skipping PyTorch benchmark format generation (benchmark_utils missing).")
        return None
# --- End Mock ---


async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    api_key: str,
    model_name: str,
    request_data: SampleRequest,
    n: int,
    request_timeout: float, # Added timeout parameter
    pbar: Optional[tqdm_asyncio] = None
) -> Dict[str, Any]:
    """Sends a single request to the OpenAI-compatible API with logging."""
    req_id = getattr(request_data, 'id', 'unknown_id') # Use assigned ID
    logger.debug(f"[{req_id}] Preparing request.")

    headers = {"Authorization": f"Bearer {api_key}"}
    # Add Content-Type header, which is standard
    headers["Content-Type"] = "application/json"

    payload = {
        "model": model_name,
        "temperature": 1.0,
        "top_p": 1.0,
        "n": n,
        "max_tokens": request_data.expected_output_len,
        "ignore_eos": True, # Matches benchmark_serving setting
        # Add other parameters like stop sequences if needed
    }

    # Determine endpoint and format payload accordingly
    try:
        if "chat/completions" in api_url:
            logger.debug(f"[{req_id}] Formatting for chat endpoint.")
            # Handle different prompt formats
            if isinstance(request_data.prompt, list): # Assume messages format
                payload["messages"] = request_data.prompt
            elif isinstance(request_data.prompt, dict) and "prompt_token_ids" in request_data.prompt:
                # This needs a tokenizer to decode, which we might not have easily here.
                # Raise error as before, or implement decoding if tokenizer is passed.
                logger.error(f"[{req_id}] Chat endpoint requires text prompts (messages format), but got token IDs.")
                raise ValueError("Chat endpoint requires text prompts (messages format), but got token IDs.")
            else: # Assume plain string prompt
                payload["messages"] = [{"role": "user", "content": str(request_data.prompt)}] # Ensure string

            # Handle multi-modal data
            if request_data.multi_modal_data:
                logger.debug(f"[{req_id}] Adding multi-modal data.")
                # Example structure (adjust based on actual API requirements):
                if payload["messages"] and isinstance(payload["messages"][0]["content"], str):
                    content_list = [{"type": "text", "text": payload["messages"][0]["content"]}]
                    if "image_urls" in request_data.multi_modal_data:
                        for url in request_data.multi_modal_data["image_urls"]:
                            content_list.append({"type": "image_url", "image_url": {"url": url}})
                    payload["messages"][0]["content"] = content_list
                else:
                     logger.warning(f"[{req_id}] Cannot automatically add multi-modal data - complex message structure.")

        elif "completions" in api_url:
            logger.debug(f"[{req_id}] Formatting for completions endpoint.")
            # Handle prompt formats for completion endpoint
            if isinstance(request_data.prompt, dict) and "prompt_token_ids" in request_data.prompt:
                logger.error(f"[{req_id}] Completion endpoint requires text prompt, but got token IDs.")
                raise ValueError("Completion endpoint requires text prompt, but got token IDs.")
            else:
                payload["prompt"] = str(request_data.prompt) # Ensure string

            if request_data.multi_modal_data:
                logger.warning(f"[{req_id}] Multi-modal data provided but /v1/completions endpoint typically doesn't support it.")
        else:
            logger.error(f"[{req_id}] Unsupported endpoint format in URL: {api_url}")
            raise ValueError(f"Unsupported endpoint format in URL: {api_url}")

    except Exception as e:
         logger.error(f"[{req_id}] Error formatting request payload: {e}")
         return {
             "request_id": req_id, "success": False, "latency": 0,
             "error": f"Payload formatting error: {e}", "prompt_tokens": request_data.prompt_len,
             "completion_tokens": 0, "total_tokens": request_data.prompt_len
         }

    # Log payload before sending (optional, can be verbose)
    # logger.debug(f"[{req_id}] Payload: {json.dumps(payload, indent=2)}")

    start_time = time.perf_counter()
    logger.info(f"[{req_id}] Sending request to {api_url}...")
    print(f"[{req_id}] Sending request to {api_url}...") # Added print statement
    sys.stdout.flush() # Ensure it gets printed immediately

    result = {}
    response_json = {}
    status_code = None

    try:
        # Set timeout for the request
        timeout = aiohttp.ClientTimeout(total=request_timeout)
        async with session.post(api_url, headers=headers, json=payload, timeout=timeout) as response:
            status_code = response.status
            logger.info(f"[{req_id}] Received response status: {status_code}")
            print(f"[{req_id}] Received response status: {status_code}") # Added print statement
            sys.stdout.flush()

            response_text = await response.text() # Read text first for better error reporting
            end_time = time.perf_counter()
            latency = end_time - start_time

            try:
                response_json = json.loads(response_text)
            except json.JSONDecodeError:
                logger.error(f"[{req_id}] Failed to decode JSON response. Status: {status_code}. Response text: {response_text[:500]}...") # Log beginning of text
                response_json = {"error": "Invalid JSON response from server", "details": response_text[:500]}


            if status_code == 200:
                logger.debug(f"[{req_id}] Request successful.")
                # Extract usage if available
                usage = response_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", request_data.prompt_len) # Fallback
                completion_tokens = usage.get("completion_tokens", request_data.expected_output_len * n) # Fallback
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens) # Fallback

                result = {
                    "request_id": req_id,
                    "success": True,
                    "latency": latency,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "response": response_json # Store full response if needed later
                }
                logger.debug(f"[{req_id}] Result: Success, Latency: {latency:.3f}s, Tokens: P{prompt_tokens}/C{completion_tokens}")
            else:
                 logger.warning(f"[{req_id}] Request failed: Status {status_code}, Error: {response_json}")
                 result = {
                    "request_id": req_id,
                    "success": False,
                    "latency": latency,
                    "status_code": status_code,
                    "error": response_json,
                    "prompt_tokens": request_data.prompt_len, # Input tokens used
                    "completion_tokens": 0,
                    "total_tokens": request_data.prompt_len,
                }

    except aiohttp.ClientConnectorError as e:
        end_time = time.perf_counter()
        logger.error(f"[{req_id}] Connection Error: Failed to connect to {api_url}. Error: {e}")
        print(f"[{req_id}] Connection Error: Failed to connect to {api_url}. Error: {e}") # Added print statement
        sys.stdout.flush()
        result = { "request_id": req_id, "success": False, "latency": end_time - start_time, "error": f"Connection Error: {e}", "prompt_tokens": request_data.prompt_len, "completion_tokens": 0, "total_tokens": request_data.prompt_len }
    except asyncio.TimeoutError:
        end_time = time.perf_counter()
        logger.error(f"[{req_id}] Request timed out after {request_timeout} seconds.")
        print(f"[{req_id}] Request timed out after {request_timeout} seconds.") # Added print statement
        sys.stdout.flush()
        result = { "request_id": req_id, "success": False, "latency": end_time - start_time, "error": "Request Timeout", "prompt_tokens": request_data.prompt_len, "completion_tokens": 0, "total_tokens": request_data.prompt_len }
    except aiohttp.ClientError as e: # Catch other client errors
        end_time = time.perf_counter()
        logger.error(f"[{req_id}] Client Error: {e}. Status Code: {status_code}")
        print(f"[{req_id}] Client Error: {e}. Status Code: {status_code}") # Added print statement
        sys.stdout.flush()
        result = { "request_id": req_id, "success": False, "latency": end_time - start_time, "error": f"ClientError: {e}", "status_code": status_code, "prompt_tokens": request_data.prompt_len, "completion_tokens": 0, "total_tokens": request_data.prompt_len }
    except Exception as e: # Catch broader unexpected errors
        end_time = time.perf_counter()
        logger.exception(f"[{req_id}] Unexpected error during request: {e}") # Use logger.exception to include traceback
        print(f"[{req_id}] Unexpected error during request: {e}") # Added print statement
        sys.stdout.flush()
        result = { "request_id": req_id, "success": False, "latency": end_time - start_time, "error": f"Unexpected error: {e}", "prompt_tokens": request_data.prompt_len, "completion_tokens": 0, "total_tokens": request_data.prompt_len }

    if pbar:
        pbar.update(1)
    return result


async def run_api_benchmark(
    requests: List[SampleRequest],
    api_url: str,
    api_key: str,
    model_name: str,
    n: int,
    concurrency: int,
    request_timeout: float, # Added
    disable_tqdm: bool = False,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Runs the benchmark by sending concurrent requests to the API."""
    logger.info(f"Starting benchmark with concurrency={concurrency}, timeout={request_timeout}s")
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    results = []

    # Create session once
    # Consider adding connection pool limits if needed: conn_limit=concurrency + buffer
    connector = aiohttp.TCPConnector(limit=None) # No limit on connections by default in aiohttp
    async with aiohttp.ClientSession(connector=connector) as session:
        start_time = time.perf_counter()
        pbar = tqdm_asyncio(total=len(requests), desc="Sending requests", disable=disable_tqdm)

        async def task_wrapper(request_data):
            async with semaphore:
                # Add a small delay before starting the actual request if needed for debugging rate limits
                # await asyncio.sleep(0.01)
                return await send_request(session, api_url, api_key, model_name, request_data, n, request_timeout, pbar) # Pass timeout

        logger.info(f"Creating {len(requests)} request tasks...")
        tasks = [asyncio.create_task(task_wrapper(req)) for req in requests]

        logger.info("Waiting for tasks to complete...")
        # Use return_exceptions=True to get tracebacks from failed tasks if needed
        results = await asyncio.gather(*tasks, return_exceptions=False)
        pbar.close()
        logger.info("All tasks finished.")
        end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    logger.info(f"Benchmark run finished in {elapsed_time:.2f} seconds.")
    return elapsed_time, results

# save_to_pytorch_benchmark_format remains the same

def get_requests(args, tokenizer):
    # Common parameters for all dataset types.
    common_kwargs = {
        "dataset_path": args.dataset_path,
        "random_seed": args.seed,
    }
    # For API benchmark, LoRA path isn't directly used for requests,
    # but might be relevant if dataset generation depends on it. Keep for now.
    sample_kwargs = {
        "tokenizer": tokenizer,
        "lora_path": args.lora_path,
        "max_loras": args.max_loras,
        "num_requests": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
    }

    # Dataset selection logic (copied from benchmark_serving.py)
    # Important: Need text prompts for standard OpenAI API.
    # Ensure the dataset yields text or handle token_ids appropriately.
    enable_multimodal = False
    logger.info(f"Loading dataset: {args.dataset_name}, Path: {args.dataset_path}")
    if args.dataset_path is None or args.dataset_name == "random":
        sample_kwargs["range_ratio"] = args.random_range_ratio
        sample_kwargs["prefix_len"] = args.prefix_len
        dataset_cls = RandomDataset
        logger.info("Using RandomDataset. Ensure generated prompts are text.")
    elif args.dataset_name == "sharegpt":
        dataset_cls = ShareGPTDataset
        sample_kwargs["enable_multimodal_chat"] = False # Assuming text-only for now
    elif args.dataset_name == "sonnet":
        if not (tokenizer.chat_template or tokenizer.default_chat_template):
             logger.warning("Tokenizer may lack chat template needed for sonnet dataset formatting.")
             # Fallback or error? For now, let it proceed but log warning.
        dataset_cls = SonnetDataset
        sample_kwargs["prefix_len"] = args.prefix_len
        sample_kwargs["return_prompt_formatted"] = True # Get chat-formatted prompts
    elif args.dataset_name == "burstgpt":
        dataset_cls = BurstGPTDataset # Check format
    elif args.dataset_name == "hf":
        # Use text-based or multimodal datasets that work with chat format
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = VisionArenaDataset
            common_kwargs['dataset_subset'] = None
            common_kwargs['dataset_split'] = "train"
            sample_kwargs["enable_multimodal_chat"] = True
            enable_multimodal = True
            logger.info(f"Using VisionArenaDataset ({args.dataset_path}). Ensure API endpoint supports multi-modal inputs.")
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = ConversationDataset
            common_kwargs['dataset_subset'] = args.hf_subset
            common_kwargs['dataset_split'] = args.hf_split
            sample_kwargs["enable_multimodal_chat"] = True # Will yield messages format
            enable_multimodal = True # Potentially
            logger.info(f"Using ConversationDataset ({args.dataset_path}).")
        # Add other HF datasets if needed, ensure they yield suitable prompt format
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
             dataset_cls = InstructCoderDataset
             common_kwargs['dataset_split'] = "train"
             logger.info(f"Using InstructCoderDataset ({args.dataset_path}). Requires text prompt.")
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
             dataset_cls = AIMODataset
             common_kwargs['dataset_subset'] = None
             common_kwargs['dataset_split'] = "train"
             enable_multimodal = True # Potentially
             logger.info(f"Using AIMODataset ({args.dataset_path}). Ensure API endpoint supports multi-modal inputs if needed.")
        else:
            logger.error(f"Unsupported HF dataset path for API benchmark: {args.dataset_path}")
            raise ValueError(f"Unsupported HF dataset path for API benchmark: {args.dataset_path}")
    else:
        logger.error(f"Unknown dataset name: {args.dataset_name}")
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    # Remove None values from sample_kwargs
    sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}
    dataset = dataset_cls(**common_kwargs)
    logger.info(f"Sampling {args.num_prompts} requests from {args.dataset_name} dataset...")
    requests = dataset.sample(**sample_kwargs)
    logger.info(f"Sampled {len(requests)} requests.")


    # Add unique IDs if dataset doesn't provide them
    for i, req in enumerate(requests):
        req.id = f"req_{i}" # Assign unique ID for tracking

    # Add flag for multi-modal
    args.is_multi_modal_dataset = enable_multimodal

    return requests


def main(args: argparse.Namespace):
    if args.seed is None:
        args.seed = 0
    logger.info(f"Starting main execution with args: {args}")
    random.seed(args.seed)

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code)
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{args.tokenizer}': {e}")
        return # Cannot proceed without tokenizer for some datasets

    # Sample the requests.
    logger.info("Generating requests from dataset...")
    try:
        requests = get_requests(args, tokenizer)
    except Exception as e:
        logger.error(f"Failed to generate requests: {e}")
        return

    if not requests:
        logger.warning("No requests generated from the dataset. Exiting.")
        return
    logger.info(f"Successfully generated {len(requests)} requests.")

    if not args.api_model_name:
         # Default to tokenizer name if api_model_name is not provided
         args.api_model_name = args.tokenizer
         logger.warning(f"--api-model-name not set, defaulting to tokenizer name: {args.api_model_name}")


    # --- Prepare API URL ---
    full_api_url = args.openai_api_url.rstrip('/') + '/' + args.endpoint.lstrip('/')
    logger.info(f"Targeting API endpoint: {full_api_url}")
    if "chat/completions" not in full_api_url and "completions" not in full_api_url:
         logger.warning(f"Endpoint '{args.endpoint}' doesn't look like a standard OpenAI endpoint. Ensure payload formatting is correct.")
    if args.is_multi_modal_dataset and "chat/completions" not in full_api_url:
         logger.warning(f"Using a potentially multi-modal dataset with non-chat endpoint '{full_api_url}'. Multi-modal data might be ignored.")


    # Run the benchmark
    logger.info("Starting API benchmark run...")
    print("\nStarting API benchmark run...") # Added print
    sys.stdout.flush()
    try:
        elapsed_time, api_results = asyncio.run(
            run_api_benchmark(
                requests=requests,
                api_url=full_api_url,
                api_key=args.openai_api_key,
                model_name=args.api_model_name,
                n=args.n,
                concurrency=args.concurrency,
                request_timeout=args.request_timeout, # Pass timeout
                disable_tqdm=False # Or make this an arg
            )
        )
        logger.info(f"Benchmark completed. Received {len(api_results)} results.")
    except Exception as e:
        logger.exception(f"An error occurred during the benchmark run: {e}") # Log full traceback
        print(f"An error occurred during the benchmark run: {e}")
        sys.stdout.flush()
        return

    # Process results
    logger.info("Processing benchmark results...")
    successful_requests = [r for r in api_results if r.get("success")]
    failed_requests = [r for r in api_results if not r.get("success")]
    num_requests = len(requests)
    num_successful = len(successful_requests)
    num_failed = len(failed_requests)

    # Use .get() with default 0 for safer summing
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in api_results)
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in successful_requests)
    total_tokens = total_prompt_tokens + total_completion_tokens

    print("\n========== API Benchmark Results ==========")
    print(f"Total time elapsed: {elapsed_time:.2f} s")
    print(f"Total requests submitted: {num_requests}")
    print(f"Successful requests: {num_successful}")
    print(f"Failed requests: {num_failed}")
    logger.info(f"Results Summary: Time={elapsed_time:.2f}s, Total={num_requests}, Success={num_successful}, Failed={num_failed}")

    requests_per_second = 0
    output_tokens_per_second = 0
    total_tokens_per_second = 0
    if num_successful > 0 and elapsed_time > 0.001: # Avoid division by zero or near-zero
        requests_per_second = num_successful / elapsed_time
        output_tokens_per_second = total_completion_tokens / elapsed_time
        total_tokens_per_second = total_tokens / elapsed_time # Based on successful completions

        print(f"Throughput (successful requests): {requests_per_second:.2f} req/s")
        print(f"Output token throughput: {output_tokens_per_second:.2f} tokens/s")
        print(f"Total token throughput (prompt+output): {total_tokens_per_second:.2f} tokens/s")
        logger.info(f"Throughput: RPS={requests_per_second:.2f}, Output TPS={output_tokens_per_second:.2f}, Total TPS={total_tokens_per_second:.2f}")
    else:
        print("No successful requests or near-zero elapsed time, cannot calculate throughput.")
        logger.warning("No successful requests or near-zero elapsed time, throughput calculation skipped.")


    print(f"Total prompt tokens processed (estimated/reported): {total_prompt_tokens}")
    print(f"Total completion tokens generated (estimated/reported): {total_completion_tokens}")
    print(f"Total tokens (prompt + completion): {total_tokens}")
    logger.info(f"Token Counts: Prompt={total_prompt_tokens}, Completion={total_completion_tokens}, Total={total_tokens}")


    if failed_requests:
        print("\n--- Failed Request Details (limit 5) ---")
        logger.warning(f"--- {len(failed_requests)} Failed Requests ---")
        for i, failed in enumerate(failed_requests[:5]):
            req_id = failed.get('request_id', 'N/A')
            status = failed.get('status_code', 'N/A')
            error = failed.get('error', 'Unknown')
            print(f"  - Request {req_id}: Status {status}, Error: {error}")
            logger.warning(f"  - Fail ID:{req_id}, Status:{status}, Error:{error}")
        if len(failed_requests) > 5:
            print(f"  ... and {len(failed_requests) - 5} more failures (see logs for details).")
            logger.warning(f"  ... (logged first 5 failures)")
        print("----------------------------------------")


    # Output JSON results if specified
    if args.output_json:
        logger.info(f"Saving results summary to {args.output_json}")
        results_summary = {
            "elapsed_time": elapsed_time,
            "num_requests": num_requests,
            "num_successful_requests": num_successful,
            "num_failed_requests": num_failed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "requests_per_second": requests_per_second,
            "output_tokens_per_second": output_tokens_per_second,
            "total_tokens_per_second": total_tokens_per_second,
            "config": {}, # Populated below
            # "individual_results": api_results # Optionally include detailed results (can be large)
        }

        # Ensure config args are serializable
        serializable_args = {}
        for k, v in vars(args).items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serializable_args[k] = v
            else:
                serializable_args[k] = str(v) # Convert non-serializable types to string
        results_summary["config"] = serializable_args

        try:
            with open(args.output_json, "w") as f:
                json.dump(results_summary, f, indent=4)
            print(f"\nResults saved to {args.output_json}")
            # Save in PyTorch benchmark format if utils are available
            # save_to_pytorch_benchmark_format(args, results_summary)
        except Exception as e:
            logger.error(f"Error saving results to JSON: {e}")
            print(f"Error saving results to JSON: {e}")

    logger.info("Script finished.")


# validate_args remains the same as before

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark an OpenAI-compatible API endpoint.")

    # --- API Specific Arguments ---
    parser.add_argument("--openai-api-url", type=str, required=True, help="Base URL of the OpenAI-compatible API endpoint (e.g., http://localhost:8000)")
    parser.add_argument("--openai-api-key", type=str, default="EMPTY", help="API key for the endpoint.")
    parser.add_argument("--endpoint", type=str, default="/v1/chat/completions", help="API endpoint path (e.g., /v1/chat/completions or /v1/completions).")
    parser.add_argument("--api-model-name", type=str, default=None, help="Model name to be sent in the API request payload.")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests to send.")
    parser.add_argument("--request-timeout", type=float, default=180.0, help="Timeout in seconds for each API request.") # Added timeout arg

    # --- Dataset Arguments (Copied/Adapted) ---
    parser.add_argument("--dataset-name", type=str, choices=["sharegpt", "random", "sonnet", "burstgpt", "hf"], help="Name of the dataset.", default="sharegpt")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Deprecated alias for --dataset-path.")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3-8B-Instruct", help="Tokenizer path/name.")
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for RandomDataset.")
    parser.add_argument("--output-len", type=int, default=1024, help="Max output length for API request.")
    parser.add_argument("--n", type=int, default=1, help="Number of sequences per prompt (passed to API).")
    parser.add_argument("--num-prompts", type=int, default=200, help="Number of prompts to process.")
    parser.add_argument('--output-json', type=str, default=None, help='Path to save the results in JSON format.')
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for tokenizer.")

    # Dataset specific args (copied)
    parser.add_argument("--lora-path", type=str, default=None, help="Path for LoRA (dataset generation only).")
    parser.add_argument("--max-loras", type=int, default=1, help="Max LoRAs (dataset generation only).")
    parser.add_argument("--enable-lora", action='store_true', help="Enable LoRA (dataset generation only).")
    parser.add_argument("--prefix-len", type=int, default=None, help="Prefix tokens for Random/Sonnet dataset.")
    parser.add_argument("--random-range-ratio", type=float, default=None, help="Range ratio for RandomDataSet.")
    parser.add_argument("--hf-subset", type=str, default=None, help="Subset of the HF dataset.")
    parser.add_argument("--hf-split", type=str, default=None, help="Split of the HF dataset.")

    args = parser.parse_args()
    # validate_args(args) # Assuming validation is done elsewhere or simple enough not to need a separate function for now
    main(args)