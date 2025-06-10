#!/usr/bin/env python3
"""
Create sample experiments that match the interface cards.
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.database import DatabaseManager  # noqa: E402
from src.experiment_tracker import ExperimentTracker  # noqa: E402


def create_sample_experiments():
    """Create sample experiments that match the interface cards."""
    print("🧪 Creating sample experiments...")

    # Use SQLite database
    sqlite_url = "sqlite:///cofferl.db"
    db_manager = DatabaseManager(sqlite_url)

    # Create tables if they don't exist
    db_manager.create_tables()

    tracker = ExperimentTracker(db_manager)

    # Sample experiments matching the interface cards
    experiments = [
        {
            "user_id": "demo_user_001",
            "brew_method": "V60",
            "parameters_json": {
                "coffee_origin": "Ethiopian Yirgacheffe",
                "roast_level": "Light",
                "grinder": "Comandante C40",
                "filter": "Hario V60 White",
                "difficulty": "Easy",
                "reward_points": 50,
                "rationale": "Light roast Ethiopian beans showcase bright acidity and floral notes when brewed with precise temperature control.",
            },
            "description": "Explore the bright, floral notes of Ethiopian Yirgacheffe with precise V60 brewing. Focus on temperature control and even extraction.",
            "status": "pending",
        },
        {
            "user_id": "demo_user_001",
            "brew_method": "V60",
            "parameters_json": {
                "coffee_origin": "Colombian Geisha",
                "roast_level": "Medium-Light",
                "grinder": "Fellow Ode",
                "filter": "Origami Dripper",
                "difficulty": "Medium",
                "reward_points": 75,
                "rationale": "Geisha varieties require careful extraction to highlight their unique jasmine and bergamot characteristics.",
            },
            "description": "Master the delicate extraction of Colombian Geisha. Experiment with grind size and pour technique to unlock complex flavors.",
            "status": "pending",
        },
        {
            "user_id": "demo_user_001",
            "brew_method": "V60",
            "parameters_json": {
                "coffee_origin": "Jamaican Blue Mountain",
                "roast_level": "Medium",
                "grinder": "Baratza Forte",
                "filter": "Kalita Wave",
                "difficulty": "Hard",
                "reward_points": 100,
                "rationale": "Blue Mountain coffee's subtle complexity demands expert-level brewing precision and attention to detail.",
            },
            "description": "Challenge yourself with the legendary Jamaican Blue Mountain. Perfect your technique to capture its renowned balance and smoothness.",
            "status": "pending",
        },
    ]

    created_count = 0
    for exp_data in experiments:
        try:
            # Check if experiment already exists
            existing = tracker.get_experiments_by_user(exp_data["user_id"])
            if len(existing) >= len(experiments):
                print(f"   ⚠️  Experiments already exist for {exp_data['user_id']}")
                continue

            experiment = tracker.create_experiment(
                user_id=exp_data["user_id"],
                brew_method=exp_data["brew_method"],
                parameters_json=exp_data["parameters_json"],
                description=exp_data["description"],
                status=exp_data["status"],
            )

            if experiment:
                created_count += 1
                print(
                    f"   ✅ Created experiment {experiment.id}: {exp_data['parameters_json']['coffee_origin']}"
                )
            else:
                print(f"   ❌ Failed to create experiment: {exp_data['description']}")

        except Exception as e:
            print(f"   ❌ Error creating experiment: {e}")

    print(f"🎉 Created {created_count} sample experiments!")

    # Show current experiments
    print("\n📋 Current experiments in database:")
    with db_manager.get_session() as session:
        from src.database import Experiment

        experiments = session.query(Experiment).all()
        for exp in experiments:
            print(f"   ID: {exp.id} | User: {exp.user_id} | Method: {exp.brew_method}")

    print(f"\n✨ Total experiments: {len(experiments)}")


if __name__ == "__main__":
    create_sample_experiments()
