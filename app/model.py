from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import MODEL_ID, MODEL_DTYPE


class RWKVModel:
    """CPU loader and text-generation wrapper for the official RWKV-7 HF checkpoint."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        print(f"Loading tokenizer: {MODEL_ID}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        dtype = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(MODEL_DTYPE)

        if dtype is None:
            raise ValueError(
                f"Unsupported BUZYA_DTYPE={MODEL_DTYPE!r}. "
                "Use 'bfloat16' or 'float32'."
            )

        print("Loading RWKV-7 weights on CPU. First launch downloads the model.", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Portable reference path; avoids requiring the optional optimized backend.
        if hasattr(self.model, "config"):
            self.model.config.wkv_implementation = "eager"

        print("RWKV-7 is loaded on CPU.", flush=True)

    @torch.inference_mode()
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 128) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            thinking=False,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {key: value.to("cpu") for key, value in inputs.items()}

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.7,
            eos_token_id=0,
            pad_token_id=0,
        )

        prompt_length = inputs["input_ids"].shape[1]
        answer = self.tokenizer.decode(
            output[0, prompt_length:],
            skip_special_tokens=True,
        )
        answer = answer.split("\n\nUser:", 1)[0].strip()
        return answer
