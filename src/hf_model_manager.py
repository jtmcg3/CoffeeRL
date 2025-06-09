#!/usr/bin/env python3
"""Hugging Face Hub Model Manager for CoffeeRL.

This module handles model versioning, checkpointing, and management using
Hugging Face Hub instead of Git LFS for better ML model handling.
"""

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import (
    HfApi,
    create_repo,
    snapshot_download,
    upload_folder,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFModelManager:
    """Manages CoffeeRL model versioning and checkpointing on Hugging Face Hub."""

    def __init__(
        self,
        repo_id: str = "JTMCG3/coffeerl-qwen2-0.5b",
        private: bool = False,
        local_cache_dir: str = "./models/hf_cache",
    ):
        """Initialize the HF Model Manager.

        Args:
            repo_id: Hugging Face repository ID (username/repo-name)
            private: Whether the repository should be private
            local_cache_dir: Local directory for caching models
        """
        self.repo_id = repo_id
        self.private = private
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize HF API
        self.api = HfApi()

        # Repository metadata
        self.repo_exists = self._check_repo_exists()

        print(f"🤗 HF Model Manager initialized for {repo_id}")
        if self.repo_exists:
            print("✅ Repository exists and accessible")
        else:
            print("📝 Repository will be created on first upload")

    def _check_repo_exists(self) -> bool:
        """Check if the repository exists and is accessible.

        Returns:
            True if repository exists and is accessible
        """
        try:
            self.api.repo_info(repo_id=self.repo_id)
            return True
        except Exception:
            return False

    def create_repository(self, description: str = None) -> bool:
        """Create the repository on Hugging Face Hub.

        Args:
            description: Repository description

        Returns:
            True if repository was created successfully
        """
        if self.repo_exists:
            print(f"✅ Repository {self.repo_id} already exists")
            return True

        try:
            default_description = (
                "CoffeeRL: Reinforcement Learning for Coffee Brewing Optimization. "
                "Fine-tuned Qwen2-0.5B model for predicting optimal coffee brewing parameters."
            )

            create_repo(
                repo_id=self.repo_id,
                private=self.private,
                repo_type="model",
                exist_ok=True,
            )

            # Create initial README
            readme_content = self._generate_model_card(
                description or default_description
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(readme_content)
                readme_path = f.name

            try:
                self.api.upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=self.repo_id,
                    commit_message="Initial model card for CoffeeRL",
                )
            finally:
                os.unlink(readme_path)

            self.repo_exists = True
            print(f"✅ Created repository: https://huggingface.co/{self.repo_id}")
            return True

        except Exception as e:
            print(f"❌ Failed to create repository: {e}")
            return False

    def _generate_model_card(self, description: str) -> str:
        """Generate a model card for the repository.

        Args:
            description: Model description

        Returns:
            Model card content in markdown format
        """
        return f"""---
license: apache-2.0
base_model: Qwen/Qwen2-0.5B
tags:
- reinforcement-learning
- coffee
- brewing
- optimization
- qwen2
- lora
- ppo
library_name: transformers
pipeline_tag: text-generation
---

# CoffeeRL: Coffee Brewing Optimization with Reinforcement Learning

{description}

## Model Details

- **Base Model**: Qwen/Qwen2-0.5B
- **Training Method**: PPO (Proximal Policy Optimization) with LoRA
- **Task**: Coffee brewing parameter optimization
- **Framework**: TRL (Transformer Reinforcement Learning)

## Training Data

The model is trained on coffee brewing experiments including:
- Grind size, water temperature, brew time parameters
- Extraction yield measurements
- User satisfaction ratings
- Brewing method variations

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

# Format your coffee brewing prompt
prompt = "Coffee: Ethiopian Yirgacheffe, Grind: Medium-fine, Water: 200°F, Method: V60"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Performance Metrics

Performance metrics are updated with each training batch. See the version history for detailed performance data.

## Training Process

This model is continuously improved through:
1. Community brewing experiments
2. Batch reinforcement learning training
3. Performance evaluation against reference models
4. User feedback integration

## Version History

Each model version corresponds to a training batch with performance improvements tracked over time.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{coffeerl2025,
  title={{CoffeeRL: Reinforcement Learning for Coffee Brewing Optimization}},
  author={{JTMCG3}},
  year={{2025}},
  url={{https://huggingface.co/{self.repo_id}}}
}}
```

## License

This model is released under the Apache 2.0 License.
"""

    def upload_model(
        self,
        model_path: str,
        version_tag: str,
        training_results: Dict[str, Any],
        commit_message: str = None,
    ) -> bool:
        """Upload a model checkpoint to Hugging Face Hub.

        Args:
            model_path: Local path to the model directory
            version_tag: Version tag for this model (e.g., "batch-1")
            training_results: Training results dictionary
            commit_message: Custom commit message

        Returns:
            True if upload was successful
        """
        if not self.repo_exists:
            if not self.create_repository():
                return False

        model_path = Path(model_path)
        if not model_path.exists():
            print(f"❌ Model path does not exist: {model_path}")
            return False

        try:
            # Generate commit message
            if commit_message is None:
                final_reward = training_results.get("final_avg_reward", 0.0)
                dataset_size = training_results.get("dataset_size", 0)
                commit_message = (
                    f"Upload {version_tag}: "
                    f"Reward={final_reward:.4f}, "
                    f"Data={dataset_size} samples"
                )

            print(f"📤 Uploading model {version_tag} to {self.repo_id}...")

            # Create training metadata file
            metadata = {
                "version_tag": version_tag,
                "upload_timestamp": datetime.now().isoformat(),
                "training_results": training_results,
                "model_info": {
                    "base_model": "Qwen/Qwen2-0.5B",
                    "training_method": "PPO with LoRA",
                    "framework": "TRL",
                },
            }

            # Save metadata to model directory temporarily
            metadata_path = model_path / "training_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            try:
                # Upload the entire model directory
                upload_folder(
                    folder_path=str(model_path),
                    repo_id=self.repo_id,
                    commit_message=commit_message,
                    revision=version_tag,  # Create/update branch with version tag
                )

                print(f"✅ Successfully uploaded {version_tag}")
                print(
                    f"🔗 Model URL: https://huggingface.co/{self.repo_id}/tree/{version_tag}"
                )

                # Update main branch with latest if this is the best model
                self._update_main_branch_if_best(version_tag, training_results)

                return True

            finally:
                # Clean up temporary metadata file
                if metadata_path.exists():
                    metadata_path.unlink()

        except Exception as e:
            print(f"❌ Failed to upload model: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _update_main_branch_if_best(
        self, version_tag: str, training_results: Dict[str, Any]
    ) -> None:
        """Update main branch if this is the best performing model.

        Args:
            version_tag: Version tag of the uploaded model
            training_results: Training results for comparison
        """
        try:
            current_reward = training_results.get("final_avg_reward", float("-inf"))

            # Get current best reward from main branch
            try:
                main_metadata = self.download_model_metadata("main")
                best_reward = main_metadata.get("training_results", {}).get(
                    "final_avg_reward", float("-inf")
                )
            except Exception:
                best_reward = float("-inf")  # No main branch yet

            if current_reward > best_reward:
                print(
                    f"🏆 New best model! Updating main branch (reward: {current_reward:.4f} > {best_reward:.4f})"
                )

                # Create a commit on main that points to this version
                commit_message = (
                    f"Update main to {version_tag} (best reward: {current_reward:.4f})"
                )

                # This is a simplified approach - in practice, you might want to merge or copy files
                self.api.create_commit(
                    repo_id=self.repo_id,
                    operations=[],  # Empty operations, just create a commit reference
                    commit_message=commit_message,
                    revision="main",
                    parent_commit=version_tag,  # Point to the version tag
                )

        except Exception as e:
            print(f"⚠️ Warning: Could not update main branch: {e}")

    def download_model(
        self, version_tag: str = "main", local_dir: str = None
    ) -> Optional[str]:
        """Download a model from Hugging Face Hub.

        Args:
            version_tag: Version tag or branch to download
            local_dir: Local directory to download to (auto-generated if None)

        Returns:
            Path to downloaded model directory or None if failed
        """
        if not self.repo_exists:
            print(f"❌ Repository {self.repo_id} does not exist")
            return None

        try:
            if local_dir is None:
                local_dir = self.local_cache_dir / f"{version_tag}_{int(time.time())}"

            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)

            print(f"📥 Downloading model {version_tag} from {self.repo_id}...")

            downloaded_path = snapshot_download(
                repo_id=self.repo_id,
                revision=version_tag,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )

            print(f"✅ Downloaded model to: {downloaded_path}")
            return downloaded_path

        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return None

    def load_model_and_tokenizer(
        self, version_tag: str = "main"
    ) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """Load model and tokenizer from Hugging Face Hub.

        Args:
            version_tag: Version tag or branch to load

        Returns:
            Tuple of (model, tokenizer) or (None, None) if failed
        """
        try:
            print(f"🔄 Loading model {version_tag} from {self.repo_id}...")

            model = AutoModelForCausalLM.from_pretrained(
                self.repo_id,
                revision=version_tag,
                trust_remote_code=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.repo_id,
                revision=version_tag,
                trust_remote_code=True,
            )

            print("✅ Successfully loaded model and tokenizer")
            return model, tokenizer

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None, None

    def list_model_versions(self) -> List[Dict[str, Any]]:
        """List all available model versions.

        Returns:
            List of version information dictionaries
        """
        if not self.repo_exists:
            return []

        try:
            # Get all branches (which represent our versions)
            repo_info = self.api.repo_info(repo_id=self.repo_id)
            branches = [
                ref.name
                for ref in repo_info.siblings
                if ref.rfilename == ".gitattributes"
            ]

            versions = []
            for branch in branches:
                try:
                    metadata = self.download_model_metadata(branch)
                    if metadata:
                        versions.append(
                            {
                                "version_tag": branch,
                                "upload_timestamp": metadata.get("upload_timestamp"),
                                "final_reward": metadata.get(
                                    "training_results", {}
                                ).get("final_avg_reward"),
                                "dataset_size": metadata.get(
                                    "training_results", {}
                                ).get("dataset_size"),
                            }
                        )
                except Exception:
                    # Skip versions without metadata
                    continue

            # Sort by upload timestamp
            versions.sort(key=lambda x: x.get("upload_timestamp", ""), reverse=True)
            return versions

        except Exception as e:
            print(f"❌ Failed to list versions: {e}")
            return []

    def download_model_metadata(
        self, version_tag: str = "main"
    ) -> Optional[Dict[str, Any]]:
        """Download model metadata for a specific version.

        Args:
            version_tag: Version tag to get metadata for

        Returns:
            Metadata dictionary or None if not found
        """
        try:
            # Download just the metadata file
            with tempfile.TemporaryDirectory() as temp_dir:
                self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename="training_metadata.json",
                    revision=version_tag,
                    local_dir=temp_dir,
                )

                metadata_path = Path(temp_dir) / "training_metadata.json"
                with open(metadata_path, "r") as f:
                    return json.load(f)

        except Exception:
            return None

    def delete_model_version(self, version_tag: str) -> bool:
        """Delete a specific model version.

        Args:
            version_tag: Version tag to delete

        Returns:
            True if deletion was successful
        """
        if version_tag == "main":
            print("❌ Cannot delete main branch")
            return False

        try:
            self.api.delete_branch(repo_id=self.repo_id, branch=version_tag)
            print(f"✅ Deleted version {version_tag}")
            return True

        except Exception as e:
            print(f"❌ Failed to delete version: {e}")
            return False

    def get_model_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history across all model versions.

        Returns:
            List of performance data sorted by upload time
        """
        versions = self.list_model_versions()

        performance_history = []
        for version in versions:
            if version.get("final_reward") is not None:
                performance_history.append(
                    {
                        "version": version["version_tag"],
                        "timestamp": version["upload_timestamp"],
                        "reward": version["final_reward"],
                        "dataset_size": version.get("dataset_size", 0),
                    }
                )

        return performance_history

    def print_model_status(self) -> None:
        """Print current model repository status."""
        print("\n🤗 Hugging Face Model Status")
        print("=" * 60)
        print(f"Repository: {self.repo_id}")
        print(f"Exists: {'✅ Yes' if self.repo_exists else '❌ No'}")
        print(f"Private: {'🔒 Yes' if self.private else '🌍 Public'}")

        if self.repo_exists:
            print(f"URL: https://huggingface.co/{self.repo_id}")

            versions = self.list_model_versions()
            if versions:
                print(f"\nAvailable Versions ({len(versions)}):")
                for version in versions[:5]:  # Show last 5 versions
                    reward = version.get("final_reward", 0.0)
                    size = version.get("dataset_size", 0)
                    timestamp = version.get("upload_timestamp", "")[:19]
                    print(
                        f"  {version['version_tag']:12} | Reward: {reward:7.4f} | Data: {size:3d} | {timestamp}"
                    )

                if len(versions) > 5:
                    print(f"  ... and {len(versions) - 5} more versions")
            else:
                print("\nNo model versions found")

        print("=" * 60)


def integrate_with_batch_trainer() -> None:
    """Show how to integrate HF Model Manager with the batch trainer."""
    print(
        """
Integration with Batch Trainer:

1. Import in batch_trainer.py:
   from hf_model_manager import HFModelManager

2. Add to BatchTrainingManager.__init__():
   self.hf_manager = HFModelManager()

3. Replace local model saving in run_batch_training():
   # Instead of: model.save_pretrained(checkpoint_path)
   version_tag = f"batch-{self.batch_count + 1}"
   self.hf_manager.upload_model(
       model_path=checkpoint_path,
       version_tag=version_tag,
       training_results=training_results
   )

4. Add model loading capability:
   def load_best_model(self):
       return self.hf_manager.load_model_and_tokenizer("main")
"""
    )


if __name__ == "__main__":
    # Example usage
    manager = HFModelManager()
    manager.print_model_status()
    integrate_with_batch_trainer()
