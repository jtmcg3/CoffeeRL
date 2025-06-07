#!/usr/bin/env python3
"""Test script to verify QLoRA setup without downloading models."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.platform_config import (  # noqa: E402
    get_device_map,
    get_gradient_accumulation_steps,
    get_optimal_qwen_model,
    get_training_batch_size,
    is_cloud_environment,
    print_platform_summary,
)


def test_platform_detection() -> bool:
    """Test platform detection functionality."""
    print("=== Testing Platform Detection ===")

    model = get_optimal_qwen_model()
    device_map = get_device_map()
    batch_size = get_training_batch_size()
    grad_accum = get_gradient_accumulation_steps()
    is_cloud = is_cloud_environment()

    print(f"✅ Model selection: {model}")
    print(f"✅ Device mapping: {device_map}")
    print(f"✅ Batch size: {batch_size}")
    print(f"✅ Gradient accumulation: {grad_accum}")
    print(f"✅ Cloud environment: {is_cloud}")
    return True


def test_qlora_config() -> bool:
    """Test QLoRA configuration setup."""
    print("\n=== Testing QLoRA Configuration ===")

    try:
        from qlora_config import setup_qlora_config, setup_training_arguments

        # Test LoRA config
        lora_config = setup_qlora_config()
        print(f"✅ LoRA rank: {lora_config.r}")
        print(f"✅ LoRA alpha: {lora_config.lora_alpha}")
        print(f"✅ Target modules: {len(lora_config.target_modules)} modules")
        print(f"✅ Dropout: {lora_config.lora_dropout}")

        # Test training arguments
        training_args = setup_training_arguments()
        print(f"✅ Training batch size: {training_args.per_device_train_batch_size}")
        print(f"✅ Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"✅ Learning rate: {training_args.learning_rate}")
        print(f"✅ Epochs: {training_args.num_train_epochs}")
        return True

    except Exception as e:
        print(f"❌ QLoRA config test failed: {e}")
        return False


def test_docker_config() -> bool:
    """Test Docker configuration files."""
    print("\n=== Testing Docker Configuration ===")

    dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
    compose_path = Path(__file__).parent.parent / "docker-compose.yml"

    if dockerfile_path.exists():
        print("✅ Dockerfile exists")
        with open(dockerfile_path) as f:
            content = f.read()
            if "qlora" in content.lower():
                print("✅ Dockerfile contains QLoRA configuration")
            else:
                print("⚠️  Dockerfile missing QLoRA configuration")
    else:
        print("❌ Dockerfile not found")

    if compose_path.exists():
        print("✅ docker-compose.yml exists")
        with open(compose_path) as f:
            content = f.read()
            if "qlora" in content.lower():
                print("✅ docker-compose.yml contains QLoRA services")
            else:
                print("⚠️  docker-compose.yml missing QLoRA services")
    else:
        print("❌ docker-compose.yml not found")

    return True


def main() -> int:
    """Run all tests."""
    print("🧪 Testing QLoRA Setup for CoffeeRL-Lite\n")

    # Print platform summary
    print_platform_summary()
    print()

    # Run tests
    tests = [
        test_platform_detection,
        test_qlora_config,
        test_docker_config,
    ]

    results: list[bool] = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            results.append(False)

    # Summary
    print("\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All {total} tests passed!")
        print("\n✅ QLoRA setup is ready for training!")
        print("\nNext steps:")
        print("1. Prepare your training data")
        print("2. Run: python src/train_qlora.py")
        print("3. Or use Docker: docker-compose --profile qlora-cpu up")
        return 0
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("Please check the failed tests above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
