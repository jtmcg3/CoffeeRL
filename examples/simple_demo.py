#!/usr/bin/env python3
"""Simple working demo of ExperimentGenerator."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.experiment_generator import ExperimentGenerator
    from src.kdtree_explorer import SimpleKDTreeExplorer
    from src.parameter_space import BrewingParameters, BrewMethod
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)


def print_params(params: BrewingParameters, index: int = None):
    """Print brewing parameters nicely."""
    prefix = f"{index + 1}. " if index is not None else ""
    print(
        f"  {prefix}{params.brew_method.value}: {params.water_temp}°C, "
        f"{params.coffee_dose}g coffee, {params.water_amount}g water, "
        f"{params.grind_size.value} grind, {params.brew_time / 60:.1f}min"
    )
    if params.bloom_time:
        print(f"     Bloom: {params.bloom_time}s")


def main():
    print("🧪 Simple ExperimentGenerator Demo")
    print("=" * 50)

    # Create generator with k-d tree
    generator = ExperimentGenerator(kdtree_explorer=SimpleKDTreeExplorer())

    print("✅ Created ExperimentGenerator")

    # Create some sample pour-over experiments (consistent dimensions)
    sample_experiments = [
        {
            "water_temp": 90.0,
            "coffee_dose": 18.0,
            "water_amount": 300.0,
            "grind_size": "medium",
            "brew_time": 240.0,
            "brew_method": "pour_over",
            "bloom_time": 30.0,
        },
        {
            "water_temp": 92.0,
            "coffee_dose": 20.0,
            "water_amount": 320.0,
            "grind_size": "medium_fine",
            "brew_time": 210.0,
            "brew_method": "pour_over",
            "bloom_time": 45.0,
        },
    ]

    print(f"\n📚 Loading {len(sample_experiments)} sample experiments...")

    # Generate new experiments using sparsity-guided exploration
    print("\n🎯 Generating sparsity-guided experiments:")
    experiments = generator.generate_exploration_based_experiments(
        num_experiments=3,
        brew_method=BrewMethod.POUR_OVER,
        exploration_strategy="sparsity_guided",
        existing_experiments=sample_experiments,
    )

    for i, params in enumerate(experiments):
        print_params(params, i)

    # Show exploration stats
    stats = generator.get_exploration_stats()
    print(
        f"\n📊 Stats: {stats['total_experiments']} experiments, "
        f"coverage: {stats.get('coverage_estimate', 0):.2f}"
    )

    # Generate using random strategy
    print("\n🎲 Generating random k-d tree experiments:")
    random_experiments = generator.generate_exploration_based_experiments(
        num_experiments=2,
        brew_method=BrewMethod.POUR_OVER,
        exploration_strategy="random",
    )

    for i, params in enumerate(random_experiments):
        print_params(params, i)

    print("\n✨ Demo complete! The ExperimentGenerator is working.")
    print("💡 It can intelligently suggest brewing experiments in unexplored regions.")


if __name__ == "__main__":
    main()
