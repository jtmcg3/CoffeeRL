#!/usr/bin/env python3
"""
Demo script to demonstrate the complete CoffeeRL experiment flow and data collection.

This script shows:
1. How experiments are created and stored
2. How users interact with experiments
3. How results are submitted and captured
4. How data flows through the system
"""

import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.experiment_tracker import ExperimentTracker  # noqa: E402


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-' * 40}")
    print(f"  {title}")
    print(f"{'-' * 40}")


def demonstrate_experiment_creation():
    """Demonstrate creating experiments in the system."""
    print_section("1. EXPERIMENT CREATION & STORAGE")

    tracker = ExperimentTracker()

    # Create a sample experiment
    print("Creating a new experiment...")
    experiment = tracker.create_experiment(
        user_id="demo_user_001",
        brew_method="V60",
        coffee_dose=18.0,
        water_amount=300.0,
        water_temperature=92.0,
        grind_size="medium-fine",
        brew_time=240,
        bloom_time=30,
        predicted_score=8.2,
        uncertainty_score=0.3,
        parameters_json={
            "coffee_origin": "Ethiopian Yirgacheffe",
            "roast_level": "Light",
            "grinder": "Comandante C40",
            "filter": "Hario V60 White",
        },
    )

    print(f"✅ Experiment created with ID: {experiment.id}")
    print(f"   User ID: {experiment.user_id}")
    print(f"   Method: {experiment.brew_method}")
    print(f"   Status: {experiment.status}")
    print(f"   Created: {experiment.created_at}")
    print(f"   Parameters: {experiment.parameters_json}")

    return experiment.id


def demonstrate_user_interactions(experiment_id: int):
    """Demonstrate user interactions with experiments."""
    print_section("2. USER INTERACTIONS & TRACKING")

    tracker = ExperimentTracker()

    # Log various user interactions
    interactions = [
        ("view", {"source": "web_interface", "device": "desktop"}),
        ("start", {"start_time": datetime.utcnow().isoformat()}),
        ("pause", {"reason": "coffee_grinder_adjustment"}),
        ("resume", {"resume_time": datetime.utcnow().isoformat()}),
    ]

    print("Logging user interactions...")
    for interaction_type, data in interactions:
        interaction = tracker.log_user_interaction(
            experiment_id=experiment_id,
            user_id="demo_user_001",
            interaction_type=interaction_type,
            interaction_data=data,
        )
        print(f"   📝 {interaction_type}: {data}")

    # Retrieve and display interactions
    print("\nRetrieving interaction history...")
    user_interactions = tracker.get_user_interactions(
        experiment_id=experiment_id, user_id="demo_user_001"
    )

    print(f"✅ Found {len(user_interactions)} interactions:")
    for interaction in user_interactions:
        print(f"   - {interaction.interaction_type} at {interaction.timestamp}")


def demonstrate_result_submission(experiment_id: int):
    """Demonstrate submitting experiment results."""
    print_section("3. RESULT SUBMISSION & DATA CAPTURE")

    tracker = ExperimentTracker()

    # Submit experiment results (simulating what happens in the web form)
    print("Submitting experiment results...")
    result = tracker.record_experiment_result(
        experiment_id=experiment_id,
        user_id="demo_user_001",
        taste_score=8.5,
        extraction_yield=22.3,
        tds=1.35,
        brew_ratio=16.7,
        notes="Excellent balance with bright acidity. Clean finish with floral notes.",
        tags=["fruity", "bright", "clean", "floral"],
        actual_brew_time=245,
        temperature_profile={
            "initial_temp": 92.0,
            "final_temp": 88.5,
            "consistency_rating": 4,
            "temperature_drop": 3.5,
        },
        is_successful=True,
    )

    print(f"✅ Result submitted with ID: {result.id}")
    print(f"   Taste Score: {result.taste_score}/10")
    print(f"   Extraction Yield: {result.extraction_yield}%")
    print(f"   TDS: {result.tds}")
    print(f"   Actual Brew Time: {result.actual_brew_time}s")
    print(f"   Notes: {result.notes}")
    print(f"   Tags: {result.tags}")
    print(f"   Temperature Profile: {result.temperature_profile}")

    # Complete the experiment
    print("\nCompleting the experiment...")
    success = tracker.complete_experiment(experiment_id, "demo_user_001")
    if success:
        print("✅ Experiment marked as completed")

    return result.id


def demonstrate_data_retrieval(experiment_id: int):
    """Demonstrate retrieving and analyzing collected data."""
    print_section("4. DATA RETRIEVAL & ANALYSIS")

    tracker = ExperimentTracker()

    # Get comprehensive experiment summary
    print("Retrieving comprehensive experiment data...")
    summary = tracker.get_experiment_summary(experiment_id)

    if summary:
        exp_data = summary["experiment"]
        interactions_data = summary["interactions"]
        results_data = summary["results"]
        summary_stats = summary["summary"]

        print_subsection("Experiment Details")
        print(f"   ID: {exp_data['id']}")
        print(f"   User: {exp_data['user_id']}")
        print(f"   Method: {exp_data['brew_method']}")
        print(f"   Coffee: {exp_data['coffee_dose']}g")
        print(f"   Water: {exp_data['water_amount']}g")
        print(f"   Temperature: {exp_data['water_temperature']}°C")
        print(f"   Grind: {exp_data['grind_size']}")
        print(f"   Status: {exp_data['status']}")

        print_subsection("Interaction Timeline")
        for interaction in interactions_data:
            print(f"   {interaction['timestamp']}: {interaction['interaction_type']}")

        print_subsection("Results Data")
        for result in results_data:
            print(f"   Taste Score: {result['taste_score']}")
            print(f"   Extraction: {result['extraction_yield']}%")
            print(f"   TDS: {result['tds']}")
            print(f"   Notes: {result['notes']}")

        print_subsection("Summary Statistics")
        print(f"   Total Interactions: {summary_stats['total_interactions']}")
        print(f"   Total Results: {summary_stats['total_results']}")
        print(f"   Duration: {summary_stats['duration_seconds']}s")
        print(f"   Has Results: {summary_stats['has_results']}")
        print(f"   Latest Score: {summary_stats['latest_taste_score']}")


def demonstrate_user_statistics():
    """Demonstrate user statistics and analytics."""
    print_section("5. USER ANALYTICS & STATISTICS")

    tracker = ExperimentTracker()

    # Get user statistics
    print("Retrieving user statistics...")
    stats = tracker.get_user_experiment_stats("demo_user_001")

    print("✅ User Statistics for 'demo_user_001':")
    print(f"   Total Experiments: {stats['total_experiments']}")
    print(f"   Completed Experiments: {stats['completed_experiments']}")
    print(f"   Completion Rate: {stats['completion_rate']:.1f}%")
    print(
        f"   Average Taste Score: {stats['average_taste_score']:.1f}/10"
        if stats["average_taste_score"]
        else "   Average Taste Score: N/A"
    )
    print(f"   Total Results: {stats['total_results']}")

    print("\n   Status Distribution:")
    for status, count in stats["status_distribution"].items():
        percentage = (
            (count / stats["total_experiments"]) * 100
            if stats["total_experiments"] > 0
            else 0
        )
        print(f"     - {status.title()}: {count} ({percentage:.1f}%)")


def demonstrate_web_interface_flow():
    """Demonstrate how the web interface integrates with the data system."""
    print_section("6. WEB INTERFACE INTEGRATION")

    print("The Gradio web interface at http://127.0.0.1:7861 provides:")
    print("\n📱 EXPERIMENT LAB TAB:")
    print("   1. Experiment Cards - Display available experiments with:")
    print("      • Parameters (coffee dose, water amount, temperature, etc.)")
    print("      • Scientific rationale")
    print("      • Difficulty level and reward points")
    print("      • Status indicators")

    print("\n📝 RESULT SUBMISSION FORM:")
    print("   2. When you click 'Submit Results', you can enter:")
    print("      • Experiment ID (1, 2, or 3 from the sample data)")
    print("      • Your User ID (use 'demo_user_001' to see the data we just created)")
    print("      • Taste Score (1-10 slider)")
    print("      • Extraction Notes (required text field)")
    print("      • Actual Brew Time (in seconds)")
    print("      • Temperature Consistency (dropdown)")
    print("      • Overall Satisfaction (1-10)")
    print("      • Additional Notes (optional)")

    print("\n📊 HISTORY & STATISTICS:")
    print("   3. Enter your User ID to view:")
    print("      • Complete experiment history table")
    print("      • Filter by status (All, Pending, Completed, etc.)")
    print("      • User statistics and performance metrics")
    print("      • Completion rates and average scores")

    print("\n🔄 DATA FLOW:")
    print("   • Form submissions → ExperimentTracker.record_experiment_result()")
    print("   • Results stored in → experiment_results table")
    print("   • Interactions logged in → user_interactions table")
    print("   • Statistics calculated from → aggregated data")
    print("   • History retrieved from → database queries")


def main():
    """Run the complete demonstration."""
    print_section("COFFEERL EXPERIMENT FLOW DEMONSTRATION")
    print("This demo shows the complete data collection and experiment flow.")
    print("The Gradio interface is running at: http://127.0.0.1:7861")

    try:
        # Run through the complete flow
        experiment_id = demonstrate_experiment_creation()
        demonstrate_user_interactions(experiment_id)
        result_id = demonstrate_result_submission(experiment_id)
        demonstrate_data_retrieval(experiment_id)
        demonstrate_user_statistics()
        demonstrate_web_interface_flow()

        print_section("NEXT STEPS")
        print("1. Open http://127.0.0.1:7861 in your browser")
        print("2. Click on the 'Experiment Lab' tab")
        print("3. Try submitting results using:")
        print("   • User ID: demo_user_001")
        print(
            f"   • Experiment ID: {experiment_id} (or 1, 2, 3 for sample experiments)"
        )
        print("4. View your experiment history in the same tab")
        print("5. Check the database to see how data is stored")

        print("\n✅ Demo completed successfully!")
        print(f"   Created experiment ID: {experiment_id}")
        print(f"   Submitted result ID: {result_id}")
        print("   User: demo_user_001")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
