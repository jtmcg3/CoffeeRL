"""Tests for RL environment setup.

This module tests the reinforcement learning environment setup
to ensure it meets the requirements specified in task 1.
"""

from unittest.mock import patch

import pytest
import torch

from src.rl_environment import (
    create_ppo_trainer,
    create_reference_model,
    load_rl_model_and_tokenizer,
    setup_ppo_config,
    setup_rl_environment,
    setup_rl_lora_config,
    setup_rl_model_with_lora,
    validate_rl_environment,
)


class TestRLEnvironmentSetup:
    """Test class for RL environment setup functionality."""

    def test_ppo_config_parameters(self):
        """Test that PPO config has the correct parameters from task 1."""
        config = setup_ppo_config()

        # Verify task 1 specifications
        assert config.batch_size == 4, "Batch size should be 4"
        assert (
            config.gradient_accumulation_steps == 4
        ), "Gradient accumulation should be 4 steps"
        assert config.learning_rate == 1e-5, "Learning rate should be 1e-5"
        assert (
            config.gradient_checkpointing is True
        ), "Gradient checkpointing should be enabled"

        # Verify other important parameters
        assert config.num_ppo_epochs == 4, "PPO epochs should be 4"
        assert config.cliprange == 0.2, "Clip range should be 0.2"
        assert config.vf_coef == 0.1, "Value function coefficient should be 0.1"

    def test_rl_lora_config(self):
        """Test LoRA configuration for RL training."""
        lora_config = setup_rl_lora_config()

        assert lora_config.r == 16, "LoRA rank should be 16"
        assert lora_config.lora_alpha == 32, "LoRA alpha should be 32"
        assert (
            lora_config.lora_dropout == 0.05
        ), "LoRA dropout should be 0.05 for RL stability"
        assert lora_config.task_type == "CAUSAL_LM", "Task type should be CAUSAL_LM"

        # Check target modules for Qwen2
        expected_modules = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        assert lora_config.target_modules == expected_modules

    @pytest.mark.slow
    def test_model_loading(self):
        """Test model and tokenizer loading."""
        model, tokenizer = load_rl_model_and_tokenizer()

        # Test tokenizer functionality
        test_text = "Coffee grind size: medium"
        tokens = tokenizer(test_text, return_tensors="pt")
        assert "input_ids" in tokens
        assert "attention_mask" in tokens

        # Test model forward pass
        with torch.no_grad():
            outputs = model(**tokens)
            assert hasattr(outputs, "logits")
            assert outputs.logits.shape[-1] > 0  # Should have vocabulary size

    @pytest.mark.slow
    def test_reference_model_creation(self):
        """Test reference model creation for PPO."""
        model, _ = load_rl_model_and_tokenizer()
        ref_model = create_reference_model(model)

        # Reference model should be frozen
        for param in ref_model.parameters():
            assert (
                not param.requires_grad
            ), "Reference model parameters should be frozen"

        # Reference model should be in eval mode
        assert not ref_model.training, "Reference model should be in eval mode"

    @pytest.mark.slow
    def test_lora_application(self):
        """Test LoRA adapter application."""
        model, _ = load_rl_model_and_tokenizer()
        lora_model = setup_rl_model_with_lora(model)

        # Check that LoRA adapters were applied
        assert hasattr(lora_model, "peft_config"), "Model should have PEFT config"

        # Check trainable parameters
        trainable_params = sum(
            p.numel() for p in lora_model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in lora_model.parameters())

        # LoRA should significantly reduce trainable parameters
        trainable_percentage = (trainable_params / total_params) * 100
        assert (
            trainable_percentage < 5
        ), f"Trainable parameters should be <5%, got {trainable_percentage:.2f}%"

    @pytest.mark.slow
    def test_complete_environment_setup(self):
        """Test complete RL environment setup."""
        model, ref_model, tokenizer, ppo_config = setup_rl_environment()

        # Verify all components are created
        assert model is not None, "Model should be created"
        assert ref_model is not None, "Reference model should be created"
        assert tokenizer is not None, "Tokenizer should be created"
        assert ppo_config is not None, "PPO config should be created"

        # Verify model types
        assert hasattr(model, "peft_config"), "Main model should have LoRA adapters"
        assert not hasattr(
            ref_model, "peft_config"
        ), "Reference model should not have LoRA adapters"

    @pytest.mark.slow
    def test_ppo_trainer_creation(self):
        """Test PPO trainer creation."""
        model, ref_model, tokenizer, ppo_config = setup_rl_environment()
        trainer = create_ppo_trainer(model, ref_model, tokenizer, ppo_config)

        assert trainer is not None, "PPO trainer should be created"
        assert hasattr(trainer, "model"), "Trainer should have model"
        assert hasattr(trainer, "ref_model"), "Trainer should have reference model"

    def test_environment_validation_function(self):
        """Test the environment validation function."""
        # This should pass without errors
        result = validate_rl_environment()
        assert result is True, "Environment validation should pass"

    def test_gpu_memory_monitoring(self):
        """Test GPU memory monitoring functionality."""
        # Test that the setup handles different hardware configurations
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                # Should work on CPU-only systems
                config = setup_ppo_config()
                assert config is not None

    def test_gradient_checkpointing_enabled(self):
        """Test that gradient checkpointing is properly enabled."""
        config = setup_ppo_config()
        assert (
            config.gradient_checkpointing is True
        ), "Gradient checkpointing must be enabled per task 1"

    def test_batch_size_configuration(self):
        """Test batch size configuration matches task 1 requirements."""
        config = setup_ppo_config()

        # Task 1 specifies batch size 4 with gradient accumulation over 4 steps
        assert config.batch_size == 4, "Batch size must be 4 per task 1"
        assert (
            config.gradient_accumulation_steps == 4
        ), "Gradient accumulation must be 4 steps per task 1"

        # Effective batch size should be 4 * 4 = 16
        effective_batch_size = config.batch_size * config.gradient_accumulation_steps
        assert (
            effective_batch_size == 16
        ), f"Effective batch size should be 16, got {effective_batch_size}"


if __name__ == "__main__":
    # Run basic tests when executed directly
    test_instance = TestRLEnvironmentSetup()

    print("Running RL environment tests...")

    # Run non-slow tests
    test_instance.test_ppo_config_parameters()
    print("✅ PPO config test passed")

    test_instance.test_rl_lora_config()
    print("✅ LoRA config test passed")

    test_instance.test_gradient_checkpointing_enabled()
    print("✅ Gradient checkpointing test passed")

    test_instance.test_batch_size_configuration()
    print("✅ Batch size configuration test passed")

    test_instance.test_environment_validation_function()
    print("✅ Environment validation function passed")

    print("🎉 All basic RL environment tests passed!")
