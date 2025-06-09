#!/usr/bin/env python3
"""Reinforcement Learning Training Script for CoffeeRL.

This script implements the core RL training loop using PPO and the TRL library,
integrating with the existing reward calculator and RL environment setup.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from trl import PPOTrainer

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from reward_calculator import RewardCalculator  # noqa: E402
from rl_environment import (  # noqa: E402
    create_ppo_trainer,
    setup_rl_environment,
    validate_rl_environment,
)


class RLTrainingLoop:
    """Main RL training loop implementation using PPO and reward calculator."""

    def __init__(
        self,
        model: Any,
        ref_model: Any,
        tokenizer: AutoTokenizer,
        ppo_trainer: PPOTrainer,
        reward_calculator: RewardCalculator,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the RL training loop.

        Args:
            model: Main model for training
            ref_model: Reference model for KL penalty
            tokenizer: Tokenizer for text processing
            ppo_trainer: PPO trainer instance
            reward_calculator: Reward calculator instance
            config: Training configuration
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.ppo_trainer = ppo_trainer
        self.reward_calculator = reward_calculator

        # Default training configuration
        default_config = {
            "max_episodes": 1000,
            "max_sequence_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "save_freq": 100,
            "eval_freq": 50,
            "log_freq": 10,
        }

        self.config = {**default_config, **(config or {})}
        self.episode_count = 0
        self.training_stats = {
            "rewards": [],
            "kl_divergences": [],
            "policy_losses": [],
            "value_losses": [],
        }

    def format_coffee_prompt(self, coffee_data: Dict[str, Any]) -> str:
        """Format coffee brewing data into a prompt for the model.

        Args:
            coffee_data: Dictionary containing coffee brewing parameters

        Returns:
            Formatted prompt string
        """
        prompt = f"""Coffee Brewing Recommendation:

Bean Type: {coffee_data.get('bean_type', 'Unknown')}
Grind Size: {coffee_data.get('grind_size', 'Medium')}
Water Temperature: {coffee_data.get('water_temp', 93)}°C
Brewing Method: {coffee_data.get('brew_method', 'Pour Over')}

Please provide brewing recommendations including:
1. Extraction yield prediction (%)
2. Brew time estimate (seconds)
3. Flavor notes and brewing tips

Recommendation:"""

        return prompt

    def parse_model_response(self, response: str) -> Dict[str, Any]:
        """Parse model response to extract brewing predictions.

        Args:
            response: Model's text response

        Returns:
            Dictionary with extracted predictions
        """
        predictions = {
            "yield_percentage": None,
            "predicted_time": None,
            "flavor_notes": response,
        }

        # Simple parsing logic - can be enhanced with more sophisticated NLP
        lines = response.lower().split("\n")

        for line in lines:
            # Look for yield percentage
            if "yield" in line or "extraction" in line:
                import re

                yield_match = re.search(r"(\d+\.?\d*)%", line)
                if yield_match:
                    try:
                        predictions["yield_percentage"] = float(yield_match.group(1))
                    except ValueError:
                        pass

            # Look for time estimate
            if "time" in line or "seconds" in line or "minutes" in line:
                import re

                # Look for time in seconds
                time_match = re.search(r"(\d+\.?\d*)\s*(?:seconds?|secs?)", line)
                if time_match:
                    try:
                        predictions["predicted_time"] = float(time_match.group(1))
                    except ValueError:
                        pass
                else:
                    # Look for time in minutes and convert
                    time_match = re.search(r"(\d+\.?\d*)\s*(?:minutes?|mins?)", line)
                    if time_match:
                        try:
                            predictions["predicted_time"] = (
                                float(time_match.group(1)) * 60
                            )
                        except ValueError:
                            pass

        return predictions

    def calculate_reward_for_episode(
        self, predictions: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> float:
        """Calculate reward for a single episode using the reward calculator.

        Args:
            predictions: Model predictions
            ground_truth: Actual brewing results

        Returns:
            Calculated reward value
        """
        return self.reward_calculator.calculate_total_reward(
            yield_percentage=predictions.get("yield_percentage"),
            predicted_time=predictions.get("predicted_time"),
            actual_time=ground_truth.get("actual_time"),
            user_rating=ground_truth.get("user_rating"),
            data_reliability=ground_truth.get("data_reliability"),
        )

    def generate_response(self, prompt: str) -> Tuple[str, torch.Tensor]:
        """Generate response from the model for a given prompt.

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (generated_text, input_ids)
        """
        # Get model device
        model_device = next(self.model.parameters()).device

        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["max_sequence_length"],
        )

        # Move inputs to model device
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                do_sample=self.config["do_sample"],
                pad_token_id=self.config["pad_token_id"],
                eos_token_id=self.config["eos_token_id"],
            )

        # Decode the generated text
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        # Return input_ids on CPU for PPO trainer compatibility
        return generated_text, input_ids[0].cpu()

    def train_episode(self, episode_data: Dict[str, Any]) -> Dict[str, float]:
        """Train a single episode using PPO.

        Args:
            episode_data: Dictionary containing episode information

        Returns:
            Dictionary with episode statistics
        """
        # Format the prompt
        prompt = self.format_coffee_prompt(episode_data["input"])

        # Generate response
        response, input_ids = self.generate_response(prompt)

        # Parse model predictions
        predictions = self.parse_model_response(response)

        # Calculate reward
        reward = self.calculate_reward_for_episode(
            predictions, episode_data["ground_truth"]
        )

        # For now, we'll use a simplified approach without PPO updates
        # This will be enhanced in the next iteration
        episode_stats = {
            "reward": reward,
            "kl_divergence": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
        }

        # Store in training stats
        self.training_stats["rewards"].append(reward)
        self.training_stats["kl_divergences"].append(episode_stats["kl_divergence"])
        self.training_stats["policy_losses"].append(episode_stats["policy_loss"])
        self.training_stats["value_losses"].append(episode_stats["value_loss"])

        return episode_stats

    def train(
        self, dataset: Dataset, num_episodes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Main training loop.

        Args:
            dataset: Training dataset
            num_episodes: Number of episodes to train (uses config if None)

        Returns:
            Training statistics
        """
        if num_episodes is None:
            num_episodes = self.config["max_episodes"]

        print(f"🚀 Starting RL training for {num_episodes} episodes...")

        start_time = time.time()

        for episode in range(num_episodes):
            self.episode_count += 1

            # Sample episode data from dataset
            episode_idx = episode % len(dataset)
            episode_data = dataset[episode_idx]

            # Train episode
            episode_stats = self.train_episode(episode_data)

            # Logging
            if episode % self.config["log_freq"] == 0:
                avg_reward = sum(self.training_stats["rewards"][-10:]) / min(
                    10, len(self.training_stats["rewards"])
                )
                print(
                    f"Episode {episode}: Reward={episode_stats['reward']:.4f}, "
                    f"Avg Reward (last 10)={avg_reward:.4f}, "
                    f"KL Div={episode_stats['kl_divergence']:.6f}"
                )

            # Save checkpoint
            if episode % self.config["save_freq"] == 0 and episode > 0:
                self.save_checkpoint(episode)

            # Evaluation
            if episode % self.config["eval_freq"] == 0 and episode > 0:
                self.evaluate(dataset)

        training_time = time.time() - start_time

        print(f"✅ Training completed in {training_time:.2f} seconds")

        return {
            "total_episodes": num_episodes,
            "training_time": training_time,
            "final_avg_reward": sum(self.training_stats["rewards"][-10:])
            / min(10, len(self.training_stats["rewards"])),
            "stats": self.training_stats,
        }

    def evaluate(
        self, dataset: Dataset, num_eval_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate the model on a subset of the dataset.

        Args:
            dataset: Evaluation dataset
            num_eval_episodes: Number of episodes for evaluation

        Returns:
            Evaluation metrics
        """
        print("🧪 Running evaluation...")

        eval_rewards = []

        for i in range(min(num_eval_episodes, len(dataset))):
            episode_data = dataset[i]

            # Generate response without training
            prompt = self.format_coffee_prompt(episode_data["input"])
            response, _ = self.generate_response(prompt)
            predictions = self.parse_model_response(response)

            # Calculate reward
            reward = self.calculate_reward_for_episode(
                predictions, episode_data["ground_truth"]
            )
            eval_rewards.append(reward)

        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
        print(f"📊 Evaluation - Average Reward: {avg_eval_reward:.4f}")

        return {
            "avg_reward": avg_eval_reward,
            "num_episodes": len(eval_rewards),
        }

    def save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint.

        Args:
            episode: Current episode number
        """
        checkpoint_dir = Path("./checkpoints/rl_training")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"episode_{episode}"

        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training stats
        stats_path = checkpoint_path / "training_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.training_stats, f, indent=2)

        print(f"💾 Checkpoint saved: {checkpoint_path}")


def create_dummy_rl_dataset() -> Dataset:
    """Create a dummy dataset for RL training testing.

    Returns:
        Dataset with dummy coffee brewing data
    """
    dummy_data = []

    for i in range(100):
        episode = {
            "input": {
                "bean_type": f"Bean Type {i % 5}",
                "grind_size": ["Fine", "Medium", "Coarse"][i % 3],
                "water_temp": 90 + (i % 10),
                "brew_method": ["Pour Over", "French Press", "Espresso"][i % 3],
            },
            "ground_truth": {
                "actual_time": 180 + (i % 60),  # 180-240 seconds
                "user_rating": (i % 5) + 1,  # 1-5 rating
                "yield_percentage": 18 + (i % 5),  # 18-22% yield
                "data_reliability": {
                    "yield": 0.9,
                    "time": 0.8,
                    "satisfaction": 0.7,
                },
            },
        }
        dummy_data.append(episode)

    return Dataset.from_list(dummy_data)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for RL training."""
    parser = argparse.ArgumentParser(
        description="CoffeeRL Reinforcement Learning Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training arguments
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50,
        help="Run evaluation every N episodes",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=10,
        help="Log training metrics every N episodes",
    )

    # Data arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to RL training dataset (uses dummy data if not provided)",
    )

    # Development mode
    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Enable development mode with reduced episodes",
    )

    # Testing
    parser.add_argument(
        "--test-env",
        action="store_true",
        help="Test RL environment setup and exit",
    )

    return parser.parse_args()


def main() -> int:
    """Main function for RL training."""
    args = parse_arguments()

    # Test environment if requested
    if args.test_env:
        print("🧪 Testing RL environment...")
        success = validate_rl_environment()
        return 0 if success else 1

    try:
        # Setup RL environment
        print("🚀 Setting up RL environment...")
        model, ref_model, tokenizer, ppo_config = setup_rl_environment()

        # Load or create dataset
        if args.dataset_path:
            print(f"Loading dataset from: {args.dataset_path}")
            dataset = load_from_disk(args.dataset_path)
        else:
            print("Using dummy dataset for testing...")
            dataset = create_dummy_rl_dataset()

        print(f"Dataset size: {len(dataset)}")

        # Create PPO trainer
        ppo_trainer = create_ppo_trainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            ppo_config=ppo_config,
        )

        # Create reward calculator
        reward_calculator = RewardCalculator()

        # Setup training configuration
        training_config = {
            "max_episodes": args.episodes,
            "save_freq": args.save_freq,
            "eval_freq": args.eval_freq,
            "log_freq": args.log_freq,
        }

        # Create training loop
        rl_trainer = RLTrainingLoop(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            ppo_trainer=ppo_trainer,
            reward_calculator=reward_calculator,
            config=training_config,
        )

        # Start training
        training_results = rl_trainer.train(dataset)

        # Print final results
        print("\n🎉 Training completed!")
        print(f"Total episodes: {training_results['total_episodes']}")
        print(f"Training time: {training_results['training_time']:.2f} seconds")
        print(f"Final average reward: {training_results['final_avg_reward']:.4f}")

        # Save final results
        results_path = Path("./results/rl_training_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(training_results, f, indent=2)

        print(f"Results saved to: {results_path}")

        return 0

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
