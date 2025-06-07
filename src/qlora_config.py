"""QLoRA configuration for Qwen2 models with platform awareness.

This module provides platform-aware QLoRA configuration for fine-tuning
Qwen2 models with 4-bit quantization support.
"""

import warnings
from typing import Any, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from config.platform_config import (
    get_device_map,
    get_gradient_accumulation_steps,
    get_optimal_qwen_model,
    get_quantization_config,
    get_torch_dtype,
    get_training_batch_size,
)


def setup_qlora_config() -> LoraConfig:
    """Set up QLoRA configuration optimized for Qwen2 architecture."""
    return LoraConfig(
        r=16,  # Low rank for parameter efficiency
        lora_alpha=32,  # Scaling factor
        # Target modules specific to Qwen2 architecture
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )


def setup_training_arguments(
    output_dir: str = "./coffee-qwen2-qlora",
) -> TrainingArguments:
    """Set up training arguments with platform awareness."""
    batch_size = get_training_batch_size()
    gradient_accumulation = get_gradient_accumulation_steps()

    # Adjust learning rate based on effective batch size
    effective_batch_size = batch_size * gradient_accumulation
    base_lr = 2e-4
    learning_rate = base_lr * (effective_batch_size / 16)  # Scale from base of 16

    return TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=3,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),  # Use fp16 only on CUDA
        bf16=False,  # Disable bf16 for compatibility
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True,
        warmup_ratio=0.05,
        weight_decay=0.01,
        report_to="tensorboard",
        dataloader_pin_memory=False,  # Disable for compatibility
        remove_unused_columns=False,
    )


def load_model_and_tokenizer() -> Tuple[Any, Any]:
    """Load Qwen2 model and tokenizer with platform-aware configuration."""
    model_name = get_optimal_qwen_model()
    device_map = get_device_map()
    torch_dtype = get_torch_dtype()
    quant_config = get_quantization_config(use_4bit=True)

    print(f"Loading model: {model_name}")
    print(f"Device mapping: {device_map}")
    print(f"Torch dtype: {torch_dtype}")
    print(f"Quantization: {'Enabled' if quant_config else 'Disabled'}")

    # Base model kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
    }

    # Add quantization config if available
    if quant_config:
        model_kwargs.update(quant_config)
    else:
        warnings.warn("Quantization disabled. Model will use more memory.", UserWarning)

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        # Fallback to smaller model
        if "1.5B" in model_name:
            print("Falling back to Qwen2-0.5B...")
            model_name = "Qwen/Qwen2-0.5B"
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            raise

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare model for QLoRA training if quantization is enabled
    if quant_config:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_qlora_model(model: Any) -> Any:
    """Apply LoRA adapters to the model."""
    lora_config = setup_qlora_config()
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def setup_training_environment() -> Tuple[Any, Any, TrainingArguments]:
    """Set up complete training environment with model, tokenizer, and training args."""
    print("Setting up QLoRA training environment...")

    # Check GPU availability and print info
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    elif torch.backends.mps.is_available():
        print("Apple Silicon GPU (MPS) available")
    else:
        print("No GPU available, training will be slow!")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Apply LoRA adapters
    model = setup_qlora_model(model)

    # Setup training arguments
    training_args = setup_training_arguments()

    print("QLoRA training environment setup complete!")

    return model, tokenizer, training_args


def format_training_data(examples: dict, tokenizer: Any, max_length: int = 512) -> dict:
    """Format training data for Qwen2 fine-tuning."""
    # Combine input and output into a single text
    texts = []
    for i in range(len(examples["input"])):
        text = f"### Input:\n{examples['input'][i]}\n\n### Output:\n{examples['output'][i]}"
        texts.append(text)

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )

    # Set labels (for causal LM, labels are the same as input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized
