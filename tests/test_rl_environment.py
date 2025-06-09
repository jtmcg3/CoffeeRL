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


class TestEarlyStopping:
    """Test class for early stopping functionality."""

    def test_early_stopping_configuration(self):
        """Test early stopping configuration parameters."""
        from unittest.mock import MagicMock

        from src.reward_calculator import RewardCalculator
        from src.train_rl import RLTrainingLoop

        # Create mock objects
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_ppo_trainer = MagicMock()
        reward_calculator = RewardCalculator()

        # Test default configuration
        rl_loop = RLTrainingLoop(
            model=mock_model,
            ref_model=mock_model,
            tokenizer=mock_tokenizer,
            ppo_trainer=mock_ppo_trainer,
            reward_calculator=reward_calculator,
        )

        assert rl_loop.config["early_stopping_patience"] == 5
        assert rl_loop.config["early_stopping_threshold"] == 0.01
        assert rl_loop.config["min_training_episodes"] == 100
        assert rl_loop.best_eval_reward == float("-inf")
        assert rl_loop.patience_counter == 0
        assert rl_loop.early_stopped is False

    def test_early_stopping_custom_configuration(self):
        """Test early stopping with custom configuration."""
        from unittest.mock import MagicMock

        from src.reward_calculator import RewardCalculator
        from src.train_rl import RLTrainingLoop

        # Create mock objects
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_ppo_trainer = MagicMock()
        reward_calculator = RewardCalculator()

        # Custom configuration
        custom_config = {
            "early_stopping_patience": 3,
            "early_stopping_threshold": 0.05,
            "min_training_episodes": 50,
        }

        rl_loop = RLTrainingLoop(
            model=mock_model,
            ref_model=mock_model,
            tokenizer=mock_tokenizer,
            ppo_trainer=mock_ppo_trainer,
            reward_calculator=reward_calculator,
            config=custom_config,
        )

        assert rl_loop.config["early_stopping_patience"] == 3
        assert rl_loop.config["early_stopping_threshold"] == 0.05
        assert rl_loop.config["min_training_episodes"] == 50

    def test_plateau_detection_no_improvement(self):
        """Test plateau detection when there's no improvement."""
        from unittest.mock import MagicMock

        from src.reward_calculator import RewardCalculator
        from src.train_rl import RLTrainingLoop

        # Create mock objects
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_ppo_trainer = MagicMock()
        reward_calculator = RewardCalculator()

        # Configuration with low patience for testing
        config = {
            "early_stopping_patience": 2,
            "early_stopping_threshold": 0.01,
            "min_training_episodes": 10,
        }

        rl_loop = RLTrainingLoop(
            model=mock_model,
            ref_model=mock_model,
            tokenizer=mock_tokenizer,
            ppo_trainer=mock_ppo_trainer,
            reward_calculator=reward_calculator,
            config=config,
        )

        # Simulate training progress
        rl_loop.episode_count = 20  # Above minimum

        # Add evaluation history with no improvement
        rl_loop.eval_history = [
            {"avg_reward": 0.5, "episode": 10},
            {"avg_reward": 0.49, "episode": 15},  # Slight decrease
            {"avg_reward": 0.48, "episode": 20},  # Another decrease
        ]
        rl_loop.best_eval_reward = 0.5
        rl_loop.patience_counter = 2  # At patience limit

        # Should detect plateau
        assert rl_loop.detect_performance_plateau() is True

    def test_plateau_detection_with_improvement(self):
        """Test plateau detection when there's significant improvement."""
        from unittest.mock import MagicMock

        from src.reward_calculator import RewardCalculator
        from src.train_rl import RLTrainingLoop

        # Create mock objects
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_ppo_trainer = MagicMock()
        reward_calculator = RewardCalculator()

        config = {
            "early_stopping_patience": 2,
            "early_stopping_threshold": 0.01,
            "min_training_episodes": 10,
        }

        rl_loop = RLTrainingLoop(
            model=mock_model,
            ref_model=mock_model,
            tokenizer=mock_tokenizer,
            ppo_trainer=mock_ppo_trainer,
            reward_calculator=reward_calculator,
            config=config,
        )

        # Simulate training progress
        rl_loop.episode_count = 20  # Above minimum

        # Add evaluation history with improvement
        rl_loop.eval_history = [
            {"avg_reward": 0.5, "episode": 10},
            {"avg_reward": 0.52, "episode": 15},  # Significant improvement
        ]
        rl_loop.best_eval_reward = 0.5
        rl_loop.patience_counter = 1

        # Should not detect plateau due to improvement
        assert rl_loop.detect_performance_plateau() is False
        assert rl_loop.patience_counter == 0  # Reset due to improvement
        assert rl_loop.best_eval_reward == 0.52  # Updated best reward

    def test_plateau_detection_before_minimum_episodes(self):
        """Test that plateau detection doesn't trigger before minimum episodes."""
        from unittest.mock import MagicMock

        from src.reward_calculator import RewardCalculator
        from src.train_rl import RLTrainingLoop

        # Create mock objects
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_ppo_trainer = MagicMock()
        reward_calculator = RewardCalculator()

        config = {
            "early_stopping_patience": 1,
            "early_stopping_threshold": 0.01,
            "min_training_episodes": 100,
        }

        rl_loop = RLTrainingLoop(
            model=mock_model,
            ref_model=mock_model,
            tokenizer=mock_tokenizer,
            ppo_trainer=mock_ppo_trainer,
            reward_calculator=reward_calculator,
            config=config,
        )

        # Simulate training progress below minimum
        rl_loop.episode_count = 50  # Below minimum
        rl_loop.eval_history = [{"avg_reward": 0.5, "episode": 50}]
        rl_loop.patience_counter = 2  # Above patience limit

        # Should not detect plateau due to minimum episodes not reached
        assert rl_loop.detect_performance_plateau() is False

    @patch("src.train_rl.Path")
    def test_best_model_saving(self, mock_path):
        """Test best model saving functionality."""
        from unittest.mock import MagicMock, mock_open

        from src.reward_calculator import RewardCalculator
        from src.train_rl import RLTrainingLoop

        # Create mock objects
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_ppo_trainer = MagicMock()
        reward_calculator = RewardCalculator()

        # Mock file operations
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.mkdir = MagicMock()

        rl_loop = RLTrainingLoop(
            model=mock_model,
            ref_model=mock_model,
            tokenizer=mock_tokenizer,
            ppo_trainer=mock_ppo_trainer,
            reward_calculator=reward_calculator,
        )

        # Test saving best model
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("json.dump") as mock_json_dump:
                rl_loop.save_best_model(episode=100, eval_reward=0.75)

        # Verify model and tokenizer save methods were called
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

        # Verify JSON file was written
        mock_file.assert_called()
        mock_json_dump.assert_called_once()

    def test_disabled_early_stopping(self):
        """Test that early stopping can be disabled."""
        from unittest.mock import MagicMock

        from src.reward_calculator import RewardCalculator
        from src.train_rl import RLTrainingLoop

        # Create mock objects
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_ppo_trainer = MagicMock()
        reward_calculator = RewardCalculator()

        # Configuration with disabled early stopping
        config = {
            "early_stopping_patience": float("inf"),
            "early_stopping_threshold": 0.0,
            "min_training_episodes": float("inf"),
        }

        rl_loop = RLTrainingLoop(
            model=mock_model,
            ref_model=mock_model,
            tokenizer=mock_tokenizer,
            ppo_trainer=mock_ppo_trainer,
            reward_calculator=reward_calculator,
            config=config,
        )

        # Simulate conditions that would normally trigger early stopping
        rl_loop.episode_count = 1000
        rl_loop.eval_history = [{"avg_reward": 0.5, "episode": 1000}]
        rl_loop.patience_counter = 100

        # Should not detect plateau when disabled
        assert rl_loop.detect_performance_plateau() is False

    @patch("src.train_rl.RLTrainingLoop.evaluate")
    @patch("src.train_rl.RLTrainingLoop.train_episode")
    def test_early_stopping_integration(self, mock_train_episode, mock_evaluate):
        """Integration test for early stopping during actual training."""
        from unittest.mock import MagicMock

        from src.reward_calculator import RewardCalculator
        from src.train_rl import RLTrainingLoop, create_dummy_rl_dataset

        # Create mock objects
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_ppo_trainer = MagicMock()
        reward_calculator = RewardCalculator()

        # Configuration for quick early stopping
        config = {
            "max_episodes": 50,
            "eval_freq": 5,
            "early_stopping_patience": 2,
            "early_stopping_threshold": 0.01,
            "min_training_episodes": 10,
            "log_freq": 5,
            "save_freq": 20,
        }

        rl_loop = RLTrainingLoop(
            model=mock_model,
            ref_model=mock_model,
            tokenizer=mock_tokenizer,
            ppo_trainer=mock_ppo_trainer,
            reward_calculator=reward_calculator,
            config=config,
        )

        # Mock train_episode to return consistent rewards and update stats
        def mock_train_episode_side_effect(episode_data):
            episode_stats = {
                "reward": 0.5,
                "kl_divergence": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
            }
            # Manually update training stats since we're mocking the method
            rl_loop.training_stats["rewards"].append(episode_stats["reward"])
            rl_loop.training_stats["kl_divergences"].append(
                episode_stats["kl_divergence"]
            )
            rl_loop.training_stats["policy_losses"].append(episode_stats["policy_loss"])
            rl_loop.training_stats["value_losses"].append(episode_stats["value_loss"])
            return episode_stats

        mock_train_episode.side_effect = mock_train_episode_side_effect

        # Mock evaluate to return plateauing rewards and update eval history
        def mock_evaluate_side_effect(dataset, num_eval_episodes=10):
            eval_count = len(rl_loop.eval_history)
            rewards = [0.6, 0.59, 0.58]  # Plateauing rewards

            if eval_count < len(rewards):
                eval_result = {
                    "episode": rl_loop.episode_count,
                    "avg_reward": rewards[eval_count],
                    "num_episodes": num_eval_episodes,
                    "timestamp": float(eval_count + 1),
                }

                # Manually update eval history and best model tracking
                rl_loop.eval_history.append(eval_result)
                if eval_result["avg_reward"] > rl_loop.best_eval_reward:
                    rl_loop.best_eval_reward = eval_result["avg_reward"]
                    rl_loop.best_model_episode = rl_loop.episode_count

                return eval_result
            else:
                # Fallback for unexpected calls
                return {
                    "episode": rl_loop.episode_count,
                    "avg_reward": 0.5,
                    "num_episodes": num_eval_episodes,
                    "timestamp": 999.0,
                }

        mock_evaluate.side_effect = mock_evaluate_side_effect

        # Create dummy dataset
        dataset = create_dummy_rl_dataset()

        # Run training
        results = rl_loop.train(dataset, num_episodes=50)

        # Verify early stopping occurred
        assert results["early_stopped"] is True
        assert (
            results["total_episodes"] == 16
        )  # Should stop at episode 15 (0-indexed), so 16 total
        assert results["best_eval_reward"] == 0.6  # Best reward from first evaluation
        assert len(rl_loop.eval_history) == 3  # Three evaluations before stopping

        # Verify evaluate was called the expected number of times
        assert mock_evaluate.call_count == 3


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
