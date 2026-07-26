"""
Model loading and QLoRA adapter configuration utilities utilizing Unsloth / HuggingFace PEFT.
"""

import torch
import gc
from typing import Tuple, Dict, Any, Optional

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig


def get_memory_stats() -> Dict[str, float]:
    """
    Returns current GPU VRAM allocation and peak usage in Gigabytes.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(0) / (1024**3)
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "max_allocated_gb": round(max_allocated, 2)
        }
    return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}


def load_base_model(
    model_name: str = "unsloth/Qwen2.5-3B",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True
) -> Tuple[Any, Any]:
    """
    Loads base language model and tokenizer using Unsloth (or fallback HuggingFace).
    """
    print(f"Loading base model '{model_name}' (max_seq_length={max_seq_length}, load_in_4bit={load_in_4bit})...")
    
    if HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )
    else:
        print("Unsloth not detected. Falling back to standard HuggingFace + BitsAndBytes...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ) if load_in_4bit else None
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

    mem = get_memory_stats()
    print(f"Model loaded successfully! Allocated VRAM: {mem['allocated_gb']} GB")
    return model, tokenizer


def apply_qlora(
    model: Any,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None
) -> Any:
    """
    Applies QLoRA PEFT adapters to target attention and MLP projection layers.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    print(f"Applying QLoRA adapters (r={r}, lora_alpha={lora_alpha}, target_modules={target_modules})...")
    
    if HAS_UNSLOTH:
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percent = 100 * trainable_params / all_params

    print(f"Trainable params: {trainable_params:,} / {all_params:,} ({percent:.4f}%)")
    return model


def clear_gpu_memory() -> None:
    """
    Frees up cached PyTorch GPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
