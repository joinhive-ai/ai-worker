import asyncio
import logging
import os
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir
from huggingface_hub import file_download
from threading import Thread
from typing import AsyncGenerator, Union

logger = logging.getLogger(__name__)


class LLMGeneratePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {"cache_dir": get_model_dir()}

        self.devices = [torch.device(f"cuda:{i}")
                        for i in range(torch.cuda.device_count())]
        if not self.devices:
            raise ValueError(
                "No CUDA devices available. This pipeline requires at least one GPU.")

        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model")
        folder_path = os.path.join(get_model_dir(), folder_name)

        # Check for fp16 variant
        has_fp16_variant = any(".fp16.safetensors" in fname for _,
                               _, files in os.walk(folder_path) for fname in files)
        if has_fp16_variant:
            logger.info("LLMGeneratePipeline loading fp16 variant for %s", model_id)
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        # Set up device map for model parallelism
        kwargs["device_map"] = self.create_device_map()

        logger.info(f"Loading model {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

        # Set up generation config
        self.generation_config = self.model.generation_config

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Optional: Add optimizations
        sfast_enabled = os.getenv("SFAST", "").strip().lower() == "true"
        if sfast_enabled:
            logger.info(
                "LLMGeneratePipeline will be dynamically compiled with stable-fast for %s", model_id)
            from app.pipelines.optim.sfast import compile_model
            self.model = compile_model(self.model)

    def create_device_map(self):
        num_gpus = len(self.devices)
        model_size = 70  # Size in billions of parameters for Llama 3.1 70B

        # Estimate memory required per GPU (in GB)
        mem_required = model_size * 2  # Rough estimate: 2GB per billion parameters

        # Get available memory for each GPU
        gpu_mem = [torch.cuda.get_device_properties(
            d).total_memory / 1e9 for d in self.devices]  # Convert to GB

        # Calculate how many GPUs we need
        gpus_needed = min(num_gpus, -(-mem_required // min(gpu_mem))
                          )  # Ceiling division

        if gpus_needed > num_gpus:
            logger.warning(f"Model might not fit in available GPU memory. Attempting to use all {num_gpus} GPUs.")
            gpus_needed = num_gpus

        # Create a balanced device map
        layers_per_gpu = -(-self.model.config.num_hidden_layers //
                           gpus_needed)  # Ceiling division
        device_map = {}

        for i in range(self.model.config.num_hidden_layers):
            gpu_index = min(i // layers_per_gpu, gpus_needed - 1)
            device_map[f'model.layers.{i}'] = self.devices[gpu_index]

        # Assign other model components
        device_map['model.embed_tokens'] = self.devices[0]
        device_map['model.norm'] = self.devices[-1]
        device_map['lm_head'] = self.devices[-1]

        logger.info(f"Created device map using {gpus_needed} GPUs: {device_map}")
        return device_map

    async def __call__(self, prompt: str, history: Optional[List[tuple]] = None, system_msg: Optional[str] = None, **kwargs) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        conversation = []
        if system_msg:
            conversation.append({"role": "system", "content": system_msg})
        if history:
            for user, assistant in history:
                conversation.extend([{"role": "user", "content": user}, {
                                    "role": "assistant", "content": assistant}])
        conversation.append({"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(
            conversation, return_tensors="pt").to(self.devices[0])
        attention_mask = torch.ones_like(input_ids)

        max_new_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)

        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

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

        thread = Thread(target=self.model_generate_wrapper, kwargs=generate_kwargs)
        thread.start()

        total_tokens = 0
        try:
            for text in streamer:
                total_tokens += 1
                yield text
                await asyncio.sleep(0)  # Allow other tasks to run
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            raise

        input_length = input_ids.size(1)
        yield {"tokens_used": input_length + total_tokens}

    def __str__(self):
        return f"LLMGeneratePipeline(model_id={self.model_id})"

    def model_generate_wrapper(self, **kwargs):
        try:
            logger.debug("Entering model.generate")
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                self.model.generate(**kwargs)
            logger.debug("Exiting model.generate")
        except Exception as e:
            logger.error(f"Error in model.generate: {str(e)}", exc_info=True)
            raise
