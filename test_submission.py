#!/usr/bin/env python3
"""
Test the experiment submission functionality.
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.database import DatabaseManager  # noqa: E402
from src.experiment_tracker import ExperimentTracker  # noqa: E402


def test_submission():
    """Test submitting an experiment result."""
    print("🧪 Testing experiment submission...")

    # Use SQLite database
    sqlite_url = "sqlite:///cofferl.db"
    db_manager = DatabaseManager(sqlite_url)
    tracker = ExperimentTracker(db_manager)

    # Test data matching what user might submit
    test_data = {
        "experiment_id": 1,
        "user_id": "jtmc3",
        "taste_score": 8.3,
        "actual_brew_time": 249,
        "notes": "Extraction notes: floral notes\nAdditional notes: excellent cup",
        "tags": ["web_submission", "temp_consistency_fair"],
        "temperature_profile": {"consistency_rating": 3},
        "is_successful": True,
    }

    try:
        # Submit the result
        print(f"   Attempting to submit: {test_data}")
        result = tracker.record_experiment_result(**test_data)

        if result:
            print(f"   ✅ Successfully submitted result ID: {result.id}")

            # Log interaction
            tracker.log_user_interaction(
                experiment_id=test_data["experiment_id"],
                user_id=test_data["user_id"],
                interaction_type="submit_result",
                interaction_data={
                    "taste_score": test_data["taste_score"],
                    "overall_satisfaction": 9,
                    "submission_method": "test_script",
                },
            )
            print("   ✅ Logged user interaction")

            # Verify the submission
            results = tracker.get_experiment_results(
                experiment_id=test_data["experiment_id"], user_id=test_data["user_id"]
            )
            print(f"   📊 Found {len(results)} results for user {test_data['user_id']}")

            return True
        else:
            print("   ❌ Failed to submit result - returned None")
            return False

    except Exception as e:
        print(f"   ❌ Error during submission: {e}")
        import traceback

        traceback.print_exc()
        return False


def show_results():
    """Show all experiment results."""
    print("\n📊 Current experiment results:")
    print("=" * 60)

    sqlite_url = "sqlite:///cofferl.db"
    db_manager = DatabaseManager(sqlite_url)

    with db_manager.get_session_context() as session:
        from src.database import ExperimentResult

        results = session.query(ExperimentResult).all()

        if results:
            for result in results:
                print(
                    f"   ID: {result.id} | Exp: {result.experiment_id} | User: {result.user_id} | Score: {result.taste_score}"
                )
        else:
            print("   No results found")


if __name__ == "__main__":
    success = test_submission()
    show_results()

    if success:
        print("\n✅ Submission test passed! The form should work properly.")
    else:
        print("\n❌ Submission test failed. Check the error messages above.")
