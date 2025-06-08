"""Reinforcement Learning Environment Setup for CoffeeRL.

This module provides the core RL environment setup using TRL's PPO implementation
with configurations optimized for coffee brewing recommendation tasks.
"""

import warnings
from typing import Any, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import PPOConfig, PPOTrainer

from config.platform_config import (
    get_device_map,
    get_optimal_qwen_model,
    get_quantization_config,
    get_torch_dtype,
)


def setup_ppo_config() -> PPOConfig:
    """Set up PPO configuration optimized for coffee brewing RL tasks.

    Configuration follows task 1 specifications:
    - Batch size: 4
    - Gradient accumulation: 4 steps
    - Learning rate: 1e-5
    - Gradient checkpointing enabled
    """
    return PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=1,  # Process one sample at a time within batch
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        num_ppo_epochs=4,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        kl_coef=0.2,  # KL divergence coefficient
        exp_name="coffeerl-ppo",
        seed=42,
        report_to="tensorboard",
        logging_steps=10,
        save_steps=100,
        output_dir="./logs/ppo_training",
    )


def setup_rl_lora_config() -> LoraConfig:
    """Set up LoRA configuration specifically for RL training.

    Uses slightly different parameters than supervised fine-tuning
    to account for the different training dynamics in RL.
    """
    return LoraConfig(
        r=16,  # Low rank for parameter efficiency
        lora_alpha=32,  # Scaling factor
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,  # Lower dropout for RL stability
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_rl_model_and_tokenizer() -> Tuple[Any, Any]:
    """Load model and tokenizer optimized for RL training.

    Returns:
        Tuple of (model, tokenizer) ready for PPO training
    """
    model_name = get_optimal_qwen_model()
    device_map = get_device_map()
    torch_dtype = get_torch_dtype()
    quant_config = get_quantization_config(use_4bit=True)

    print(f"Loading RL model: {model_name}")
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
        warnings.warn(
            "Quantization disabled. RL training will use more memory.", UserWarning
        )

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        # Fallback to smaller model for RL
        if "1.5B" in model_name:
            print("Falling back to Qwen2-0.5B for RL training...")
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

    # Prepare model for RL training if quantization is enabled
    if quant_config:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_rl_model_with_lora(model: Any) -> Any:
    """Apply LoRA adapters to the model for RL training.

    Args:
        model: Base model to apply LoRA to

    Returns:
        Model with LoRA adapters applied
    """
    lora_config = setup_rl_lora_config()
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    print("RL Model LoRA Configuration:")
    model.print_trainable_parameters()

    return model


def create_reference_model(base_model: Any) -> Any:
    """Create a reference model for PPO training.

    The reference model is used to compute KL divergence penalties
    to prevent the policy from deviating too much from the original model.
    This should be called BEFORE applying LoRA to the main model.

    Args:
        base_model: The base model (before LoRA) to create a reference from

    Returns:
        Reference model (frozen copy of the base model)
    """
    # Create a copy of the base model for reference
    ref_model = type(base_model)(base_model.config)
    ref_model.load_state_dict(base_model.state_dict())

    # Freeze the reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    ref_model.eval()

    print("Reference model created and frozen for PPO training")
    return ref_model


def setup_rl_environment() -> Tuple[Any, Any, Any, PPOConfig]:
    """Set up complete RL environment for coffee brewing optimization.

    Returns:
        Tuple of (model, ref_model, tokenizer, ppo_config)
    """
    print("🚀 Setting up Reinforcement Learning environment for CoffeeRL...")

    # Check GPU availability and print info
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    elif torch.backends.mps.is_available():
        print("Apple Silicon GPU (MPS) available")
    else:
        print("No GPU available, RL training will be very slow!")

    # Load base model and tokenizer
    base_model, tokenizer = load_rl_model_and_tokenizer()

    # Create reference model BEFORE applying LoRA
    ref_model = create_reference_model(base_model)

    # Apply LoRA adapters to the main model for RL training
    model = setup_rl_model_with_lora(base_model)

    # Setup PPO configuration
    ppo_config = setup_ppo_config()

    print("✅ RL environment setup complete!")
    print(f"Model: {get_optimal_qwen_model()}")
    print(f"Batch size: {ppo_config.batch_size}")
    print(f"Learning rate: {ppo_config.learning_rate}")
    print(f"Gradient accumulation steps: {ppo_config.gradient_accumulation_steps}")

    return model, ref_model, tokenizer, ppo_config


def create_ppo_trainer(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    ppo_config: PPOConfig,
    train_dataset: Optional[Any] = None,
    reward_model: Optional[Any] = None,
    value_model: Optional[Any] = None,
) -> PPOTrainer:
    """Create PPO trainer instance.

    Args:
        model: Main model for training
        ref_model: Reference model for KL penalty
        tokenizer: Tokenizer for text processing
        ppo_config: PPO configuration
        train_dataset: Optional training dataset
        reward_model: Optional reward model (uses main model if None)
        value_model: Optional value model (uses main model if None)

    Returns:
        Configured PPOTrainer instance
    """
    print("Creating PPO trainer...")

    # Use main model as reward and value model if not provided
    if reward_model is None:
        reward_model = model
    if value_model is None:
        value_model = model

    # Create a dummy dataset if none provided (for testing)
    if train_dataset is None:
        from datasets import Dataset

        dummy_data = {
            "input_ids": [[1, 2, 3, 4, 5]],  # Dummy token IDs
            "attention_mask": [[1, 1, 1, 1, 1]],
        }
        train_dataset = Dataset.from_dict(dummy_data)

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        value_model=value_model,
    )

    print("✅ PPO trainer created successfully")
    return trainer


def validate_rl_environment() -> bool:
    """Test the RL environment setup with dummy data.

    This function validates that:
    1. Models load correctly
    2. PPO configuration is valid
    3. Trainer can be created
    4. Basic forward pass works

    Returns:
        True if all tests pass, False otherwise
    """
    print("🧪 Testing RL environment setup...")

    try:
        # Setup environment
        model, ref_model, tokenizer, ppo_config = setup_rl_environment()

        # Test tokenizer
        test_text = "Coffee grind size: medium, water temperature: 93°C"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenizer test passed - tokens shape: {tokens['input_ids'].shape}")

        # Test model forward pass
        with torch.no_grad():
            outputs = model(**tokens)
            print(
                f"✅ Model forward pass test passed - output shape: {outputs.logits.shape}"
            )

        # Test reference model
        with torch.no_grad():
            ref_outputs = ref_model(**tokens)
            print(
                f"✅ Reference model test passed - output shape: {ref_outputs.logits.shape}"
            )

        # Test PPO trainer creation (without dataset for now)
        create_ppo_trainer(model, ref_model, tokenizer, ppo_config)
        print("✅ PPO trainer creation test passed")

        print("🎉 All RL environment tests passed!")
        return True

    except Exception as e:
        print(f"❌ RL environment test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run environment validation when script is executed directly
    success = validate_rl_environment()
    exit(0 if success else 1)
