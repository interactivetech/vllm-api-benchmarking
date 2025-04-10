# SPDX-License-Identifier: Apache-2.0

import json
import logging # Added
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional, Union

import aiohttp
import huggingface_hub.constants
from tqdm.asyncio import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

# NOTE(simon): do not import vLLM here so the benchmark script
# can run without vLLM installed.

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

# Added basic logging configuration (can be overridden by the main script)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__) # Added logger


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    api_key: Optional[str] = None  # Added api_key field


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(
        default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""


async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")
    logger.info("Sending request to TGI endpoint: %s", api_url) # Added log

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        params = {
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
            "truncate": request_func_input.prompt_len,
            "ignore_eos_token": request_func_input.ignore_eos,
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        # logger.debug("TGI Payload: %s", payload) # Optional: log payload details
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        if request_func_input.ignore_eos:
            output.output_tokens = request_func_input.output_len
        else:
            output.output_tokens = None

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            logger.debug("Posting to TGI URL: %s", api_url) # Added log
            async with session.post(url=api_url, json=payload) as response:
                logger.info("Received TGI response status: %d for %s", response.status, api_url) # Added log
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")

                        # NOTE: Sometimes TGI returns a ping response without
                        # any data, we should skip it.
                        if chunk_bytes.startswith(":"):
                            continue
                        chunk = chunk_bytes.removeprefix("data:")

                        try: # Added specific try-except for JSON loading
                           data = json.loads(chunk)
                        except json.JSONDecodeError:
                            logger.error("Failed to decode TGI JSON chunk: %s", chunk)
                            continue # Skip malformed chunk

                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                            logger.debug("TTFT recorded: %.4f s", ttft)

                        # Decoding phase
                        else:
                            itl_val = timestamp - most_recent_timestamp
                            output.itl.append(itl_val)
                            logger.debug("ITL recorded: %.4f s", itl_val)


                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]
                    logger.info("TGI request successful. Latency: %.4f s", output.latency) # Added log
                else:
                    output.error = response.reason or f"Status code {response.status}"
                    output.success = False
                    logger.error("TGI request failed. Status: %d, Reason: %s", response.status, output.error) # Added log
        except Exception as e:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            # Use logger.exception to include traceback automatically
            logger.exception("Exception during TGI request to %s", api_url) # Modified log

        if pbar:
            pbar.update(1)
        return output


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")
    logger.info("Sending request to TRT-LLM endpoint: %s", api_url) # Added log

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        if request_func_input.ignore_eos:
            payload["min_length"] = request_func_input.output_len
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        # logger.debug("TRT-LLM Payload: %s", payload) # Optional

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            logger.debug("Posting to TRT-LLM URL: %s", api_url) # Added log
            async with session.post(url=api_url, json=payload) as response:
                logger.info("Received TRT-LLM response status: %d for %s", response.status, api_url) # Added log
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data:")

                        try: # Added specific try-except for JSON loading
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            logger.error("Failed to decode TRT-LLM JSON chunk: %s", chunk)
                            continue # Skip malformed chunk

                        output.generated_text += data.get("text_output", "") # Use .get for safety
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft
                            logger.debug("TTFT recorded: %.4f s", ttft)

                        # Decoding phase
                        else:
                            itl_val = timestamp - most_recent_timestamp
                            output.itl.append(itl_val)
                            logger.debug("ITL recorded: %.4f s", itl_val)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    logger.info("TRT-LLM request successful. Latency: %.4f s", output.latency) # Added log

                else:
                    output.error = response.reason or f"Status code {response.status}"
                    output.success = False
                    logger.error("TRT-LLM request failed. Status: %d, Reason: %s", response.status, output.error) # Added log
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            logger.exception("Exception during TRT-LLM request to %s", api_url) # Modified log

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    logger.info("Sending request to DeepSpeed-MII endpoint: %s", request_func_input.api_url) # Added log
    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:

        payload = {
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temp.
            "top_p": 1.0,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        # logger.debug("DeepSpeed-MII Payload: %s", payload) # Optional

        # NOTE: DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # See https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            logger.debug("Posting to DeepSpeed-MII URL: %s", request_func_input.api_url) # Added log
            async with session.post(url=request_func_input.api_url,
                                    json=payload) as response:
                logger.info("Received DeepSpeed-MII response status: %d for %s", response.status, request_func_input.api_url) # Added log
                if response.status == 200:
                    parsed_resp = await response.json()
                    output.latency = time.perf_counter() - st
                    if "choices" in parsed_resp:
                        output.generated_text = parsed_resp["choices"][0][
                            "text"]
                    elif "text" in parsed_resp:
                        output.generated_text = parsed_resp["text"][0]
                    else:
                        output.error = ("Unexpected response format: "
                                        "neither 'choices' nor 'text' found")
                        output.success = False
                        logger.error(output.error + " in response: %s", parsed_resp) # Added log
                    if output.success is None: # Check if not already set to False
                       output.success = True
                    if output.success:
                       logger.info("DeepSpeed-MII request successful. Latency: %.4f s", output.latency) # Added log
                else:
                    output.error = response.reason or f"Status code {response.status}"
                    output.success = False
                    logger.error("DeepSpeed-MII request failed. Status: %d, Reason: %s", response.status, output.error) # Added log
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            logger.exception("Exception during DeepSpeed-MII request to %s", request_func_input.api_url) # Modified log

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), f"OpenAI Completions API URL must end with 'completions' or 'profile', got: {api_url}" # Added URL to assert msg
    logger.info("Sending request to OpenAI Completions endpoint: %s", api_url) # Added log

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model_name \
                if request_func_input.model_name else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)

        # --- API Key Handling ---
        api_key_to_use = request_func_input.api_key or os.environ.get('OPENAI_API_KEY')
        headers = {}
        if api_key_to_use:
            logger.info("Using API key for request to %s", api_url) # Added log
            headers["Authorization"] = f"Bearer {api_key_to_use}"
        else:
            logger.warning("No API key found (checked argument and OPENAI_API_KEY env var) for %s", api_url) # Added warning
        # --- End API Key Handling ---

        # logger.debug("OpenAI Completions Payload: %s", payload) # Optional
        # logger.debug("OpenAI Completions Headers: %s", {k: ('***' if k == 'Authorization' else v) for k, v in headers.items()}) # Log headers safely

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            logger.debug("Posting to OpenAI Completions URL: %s", api_url) # Added log
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                logger.info("Received OpenAI Completions response status: %d for %s", response.status, api_url) # Added log
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk_str = chunk_bytes.decode("utf-8") # Decode once
                        if chunk_str.startswith(":"): # Handle potential ping messages
                            logger.debug("Skipping ping message: %s", chunk_str)
                            continue

                        chunk = chunk_str.removeprefix("data: ")
                        if chunk != "[DONE]":
                            try: # Added specific try-except for JSON loading
                                data = json.loads(chunk)
                            except json.JSONDecodeError:
                                logger.error("Failed to decode OpenAI Completions JSON chunk: %s", chunk)
                                continue # Skip malformed chunk

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                    logger.debug("TTFT recorded: %.4f s", ttft)

                                # Decoding phase
                                else:
                                    itl_val = timestamp - most_recent_timestamp
                                    output.itl.append(itl_val)
                                    logger.debug("ITL recorded: %.4f s", itl_val)


                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")
                                logger.debug("Received usage info: %s", usage) # Added log
                        else:
                            logger.debug("Received [DONE] message.") # Added log

                    if first_chunk_received:
                        output.success = True
                        logger.info("OpenAI Completions request successful. Latency: %.4f s", most_recent_timestamp - st) # Added log
                    else:
                        # Read the potential error message from the body if stream didn't start
                        try:
                            error_body = await response.text()
                            logger.error("Stream never started. Response body: %s", error_body)
                            output.error = f"Stream never started. Body: {error_body[:500]}..." # Truncate long errors
                        except Exception as read_err:
                            logger.error("Stream never started and failed to read error body: %s", read_err)
                            output.error = "Stream never started. Failed to read error body."

                        output.success = False
                        # logger.error previously added below

                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    error_body = await response.text()
                    output.error = response.reason or f"Status code {response.status}. Body: {error_body[:500]}..." # Include body snippet
                    output.success = False
                    logger.error("OpenAI Completions request failed. Status: %d, Reason: %s", response.status, output.error) # Added log
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            logger.exception("Exception during OpenAI Completions request to %s", api_url) # Modified log

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("chat/completions", "profile")
    ), f"OpenAI Chat Completions API URL must end with 'chat/completions' or 'profile', got: {api_url}" # Added URL to assert msg
    logger.info("Sending request to OpenAI Chat Completions endpoint: %s", api_url) # Added log

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            content.append(request_func_input.multi_modal_content)
            logger.info("Including multi-modal content in request.") # Added log
        payload = {
            "model": request_func_input.model_name \
                if request_func_input.model_name else request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                },
            ],
            "temperature": 0.0,
            # Note: vLLM uses max_tokens, standard OpenAI uses max_completion_tokens
            # We might need to adjust this based on the specific server implementation
            # For now, assume 'max_tokens' works for vLLM's OpenAI compatible endpoint
            "max_tokens": request_func_input.output_len, # Changed from max_completion_tokens for vLLM compatibility
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)

        # --- API Key Handling ---
        api_key_to_use = request_func_input.api_key or os.environ.get('OPENAI_API_KEY')
        headers = {
             # This is generally required for chat completions
            "Content-Type": "application/json",
        }
        if api_key_to_use:
            logger.info("Using API key for request to %s", api_url) # Added log
            headers["Authorization"] = f"Bearer {api_key_to_use}"
        else:
            logger.warning("No API key found (checked argument and OPENAI_API_KEY env var) for %s", api_url) # Added warning
        # --- End API Key Handling ---

        # logger.debug("OpenAI Chat Completions Payload: %s", payload) # Optional
        # logger.debug("OpenAI Chat Completions Headers: %s", {k: ('***' if k == 'Authorization' else v) for k, v in headers.items()}) # Log headers safely


        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        first_chunk_received = False # Added flag for chat
        try:
            logger.debug("Posting to OpenAI Chat Completions URL: %s", api_url) # Added log
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                logger.info("Received OpenAI Chat Completions response status: %d for %s", response.status, api_url) # Added log
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk_str = chunk_bytes.decode("utf-8") # Decode once
                        if chunk_str.startswith(":"): # Handle potential ping messages
                            logger.debug("Skipping ping message: %s", chunk_str)
                            continue

                        chunk = chunk_str.removeprefix("data: ")
                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            try: # Added specific try-except for JSON loading
                                data = json.loads(chunk)
                            except json.JSONDecodeError:
                                logger.error("Failed to decode OpenAI Chat Completions JSON chunk: %s", chunk)
                                continue # Skip malformed chunk

                            if choices := data.get("choices"):
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content is not None: # Check if content exists (could be role chunk)
                                    # First token with actual content
                                    if not first_chunk_received:
                                        first_chunk_received = True
                                        ttft = timestamp - st
                                        output.ttft = ttft
                                        logger.debug("TTFT recorded: %.4f s", ttft)

                                    # Decoding phase
                                    else:
                                        itl_val = timestamp - most_recent_timestamp
                                        output.itl.append(itl_val)
                                        logger.debug("ITL recorded: %.4f s", itl_val)

                                    generated_text += content or ""
                                else:
                                    logger.debug("Received chunk without content: %s", delta) # Log role/empty chunks


                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")
                                logger.debug("Received usage info: %s", usage) # Added log

                            most_recent_timestamp = timestamp
                        else:
                             logger.debug("Received [DONE] message.") # Added log

                    if first_chunk_received: # Check if we got any content
                        output.success = True
                        logger.info("OpenAI Chat Completions request successful. Latency: %.4f s", most_recent_timestamp - st) # Added log
                    else:
                         # Read the potential error message from the body if stream didn't start
                        try:
                            error_body = await response.text()
                            logger.error("Stream never started or no content received. Response body: %s", error_body)
                            output.error = f"No content received. Body: {error_body[:500]}..." # Truncate long errors
                        except Exception as read_err:
                            logger.error("Stream never started/no content and failed to read error body: %s", read_err)
                            output.error = "No content received. Failed to read error body."
                        output.success = False
                        # logger.error previously added below

                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    error_body = await response.text()
                    output.error = response.reason or f"Status code {response.status}. Body: {error_body[:500]}..." # Include body snippet
                    output.success = False
                    logger.error("OpenAI Chat Completions request failed. Status: %d, Reason: %s", response.status, output.error) # Added log
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            logger.exception("Exception during OpenAI Chat Completions request to %s", api_url) # Modified log

    if pbar:
        pbar.update(1)
    return output


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv('VLLM_USE_MODELSCOPE', 'False').lower() == 'true':
        # Added logging for ModelScope download attempt
        logger.info("Attempting to download model '%s' from ModelScope...", pretrained_model_name_or_path)
        from modelscope import snapshot_download

        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(pretrained_model_name_or_path):
            try: # Added try-except for download
                model_path = snapshot_download(
                    model_id=pretrained_model_name_or_path,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])
                logger.info("ModelScope download successful. Path: %s", model_path) # Added log
            except Exception as e:
                 logger.exception("ModelScope download failed for %s", pretrained_model_name_or_path) # Added log
                 raise e # Re-raise the exception

            return model_path
    # Added log for non-ModelScope path
    logger.info("Using provided model path/name: %s", pretrained_model_name_or_path)
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    logger.info("Loading tokenizer '%s' with mode '%s'", pretrained_model_name_or_path, tokenizer_mode) # Added log
    if pretrained_model_name_or_path is not None and not os.path.exists(
            pretrained_model_name_or_path):
        # Model download attempt happens within get_model
        pretrained_model_name_or_path = get_model(
            pretrained_model_name_or_path)
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
        logger.info("Using slow tokenizer.") # Added log
    if tokenizer_mode == "mistral":
        logger.info("Attempting to load MistralTokenizer.") # Added log
        try:
            from vllm.transformers_utils.tokenizer import MistralTokenizer
        except ImportError as e:
            logger.error("Failed to import MistralTokenizer. Ensure vLLM is installed.") # Added log
            raise ImportError("MistralTokenizer requires vllm package.\n"
                              "Please install it with `pip install vllm` "
                              "to use mistral tokenizer mode.") from e
        return MistralTokenizer.from_pretrained(
            str(pretrained_model_name_or_path))
    else:
        logger.info("Using AutoTokenizer.") # Added log
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "sglang": async_request_openai_completions,
}

OPENAI_COMPATIBLE_BACKENDS = [
    k for k, v in ASYNC_REQUEST_FUNCS.items()
    if v in (async_request_openai_completions,
             async_request_openai_chat_completions)
]