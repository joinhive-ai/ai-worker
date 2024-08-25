import asyncio
import base64
import io
import logging
import os
import psutil
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from queue import Queue
from threading import Thread

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from peft import PeftModel, PeftConfig
from safetensors.torch import load_file as safe_load_file
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from huggingface_hub import file_download, snapshot_download

logger = logging.getLogger(__name__)

def get_max_memory():
    num_gpus = torch.cuda.device_count()
    gpu_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory // 1024**3}GiB" for i in range(num_gpus)}
    cpu_memory = f"{psutil.virtual_memory().available // 1024**3}GiB"
    max_memory = {**gpu_memory, "cpu": cpu_memory}
    
    logger.info(f"Max memory configuration: {max_memory}")
    return max_memory

def load_model_8bit(model_id: str, **kwargs):
    max_memory = get_max_memory()

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory,
        offload_folder="offload",
        low_cpu_mem_usage=True,
        **kwargs
    )

    return tokenizer, model

def load_model_fp16(model_id: str, **kwargs):
    device = get_torch_device()
    max_memory = get_max_memory()
    
    local_model_path = os.path.join(get_model_dir(), file_download.repo_folder_name(repo_id=model_id, repo_type="model"))
    has_fp16_variant = any(".fp16.safetensors" in fname for _, _, files in os.walk(local_model_path) for fname in files)
    
    if device != "cpu" and has_fp16_variant:
        logger.info("Loading fp16 variant for %s", model_id)
        kwargs["torch_dtype"] = torch.float16
        kwargs["variant"] = "fp16"
    elif device != "cpu":
        kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    
    config = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).config
    
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    checkpoint_dir = snapshot_download(model_id, cache_dir=get_model_dir(), local_files_only=True)
    
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint_dir,
        device_map="auto",
        max_memory=max_memory,
        no_split_module_classes=["LlamaDecoderLayer"],  # Adjust based on your model architecture
        dtype=kwargs.get("torch_dtype", torch.float32),
        offload_folder="offload",
        offload_state_dict=True,
    )

    return tokenizer, model

class LLMGeneratePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {
            "cache_dir": get_model_dir(),
            "local_files_only": True,
        }
        self.device = get_torch_device()

        folder_path = file_download.repo_folder_name(repo_id=model_id, repo_type="model")
        self.local_model_path = os.path.join(get_model_dir(), folder_path)
        self.checkpoint_dir = snapshot_download(model_id, cache_dir=get_model_dir(), local_files_only=True)

        logger.info(f"Local model path: {self.local_model_path}")
        logger.info(f"Directory contents: {os.listdir(self.local_model_path)}")

        use_8bit = os.getenv("USE_8BIT", "").strip().lower() == "true"
        
        if use_8bit:
            logger.info("Using 8-bit quantization")
            self.tokenizer, self.model = load_model_8bit(model_id, **kwargs)
        else:
            logger.info("Using fp16/bf16 precision")
            self.tokenizer, self.model = load_model_fp16(model_id, **kwargs)

        logger.info(f"Model loaded and distributed. Device map: {self.model.hf_device_map}")

        self.generation_config = self.model.generation_config

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        sfast_enabled = os.getenv("SFAST", "").strip().lower() == "true"
        if sfast_enabled:
            logger.info(
                "LLMGeneratePipeline will be dynamically compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.optim.sfast import compile_model
            self.model = compile_model(self.model)

        self.request_queue = Queue()
        self.worker_thread = Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def _process_queue(self):
        while True:
            task = self.request_queue.get()
            if task is None:
                break
            self._process_task(*task)

    def _process_task(self, prompt, history, system_msg, lora_weights, future, **kwargs):
        try:
            if lora_weights:
                model = self.apply_lora(self.model, lora_weights)
            else:
                model = self.model

            result = self._generate(model, prompt, history, system_msg, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            if lora_weights:
                self.model.delete_adapters()
            torch.cuda.empty_cache()

    def apply_lora(self, model, lora_weights: str):
        try:
            lora_bytes = base64.b64decode(lora_weights)
            with io.BytesIO(lora_bytes) as f:
                lora_state_dict = safe_load_file(f)
            model = PeftModel.from_pretrained(model, lora_state_dict, is_trainable=False)
            logger.info("Applied LoRA weights")
            return model
        except Exception as e:
            logger.error(f"Error applying LoRA weights: {str(e)}")
            raise

    def _generate(self, model, prompt, history, system_msg, **kwargs):
        conversation = []
        if system_msg:
            conversation.append({"role": "system", "content": system_msg})
        if history:
            conversation.extend(history)
        conversation.append({"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)

        max_new_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)

        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = self.generation_config.to_dict()
        generate_kwargs.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        })

        thread = Thread(target=self.model_generate_wrapper, args=(model,), kwargs=generate_kwargs)
        thread.start()

        total_tokens = 0
        try:
            for text in streamer:
                total_tokens += 1
                yield text
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            raise
        finally:
            thread.join()

        input_length = input_ids.size(1)
        yield {"tokens_used": input_length + total_tokens}

    def model_generate_wrapper(self, model, **kwargs):
        try:
            logger.debug("Entering model.generate")
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                model.generate(**kwargs)
            logger.debug("Exiting model.generate")
        except Exception as e:
            logger.error(f"Error in model.generate: {str(e)}", exc_info=True)
            raise

    async def __call__(self, prompt: str, history: Optional[List[tuple]] = None, system_msg: Optional[str] = None, lora_weights: Optional[str] = None, **kwargs) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.request_queue.put((prompt, history, system_msg, lora_weights, future, kwargs))
        result = await future

        async for item in result:
            yield item

    def cleanup(self):
        self.request_queue.put(None)
        self.worker_thread.join()

    def __str__(self):
        return f"LLMGeneratePipeline(model_id={self.model_id})"