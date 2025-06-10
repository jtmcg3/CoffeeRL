#!/usr/bin/env python3
"""
Simple demo to show experiment result submission and data capture.
"""

import sqlite3


def show_database_contents():
    """Show current database contents."""
    print("📊 Current Database Contents:")
    print("=" * 50)

    conn = sqlite3.connect("cofferl.db")
    cursor = conn.cursor()

    # Show experiments
    print("\n🧪 EXPERIMENTS:")
    cursor.execute(
        "SELECT id, user_id, brew_method, coffee_dose, water_amount, status, created_at FROM experiments"
    )
    experiments = cursor.fetchall()
    if experiments:
        for exp in experiments:
            print(
                f"   ID: {exp[0]} | User: {exp[1]} | Method: {exp[2]} | Coffee: {exp[3]}g | Water: {exp[4]}g | Status: {exp[5]}"
            )
    else:
        print("   No experiments found")

    # Show results
    print("\n📝 EXPERIMENT RESULTS:")
    cursor.execute(
        "SELECT id, experiment_id, user_id, taste_score, notes, recorded_at FROM experiment_results"
    )
    results = cursor.fetchall()
    if results:
        for result in results:
            print(
                f"   ID: {result[0]} | Exp: {result[1]} | User: {result[2]} | Score: {result[3]} | Notes: {result[4][:50]}..."
            )
    else:
        print("   No results found")

    # Show interactions
    print("\n👤 USER INTERACTIONS:")
    cursor.execute(
        "SELECT id, experiment_id, user_id, interaction_type, timestamp FROM user_interactions"
    )
    interactions = cursor.fetchall()
    if interactions:
        for interaction in interactions:
            print(
                f"   ID: {interaction[0]} | Exp: {interaction[1]} | User: {interaction[2]} | Type: {interaction[3]} | Time: {interaction[4]}"
            )
    else:
        print("   No interactions found")

    conn.close()


def main():
    """Main demonstration."""
    print("🎯 CoffeeRL Experiment Flow Demo")
    print("=" * 50)
    print()
    print("The Gradio interface is running at: http://127.0.0.1:7861")
    print()
    print("📋 STEP-BY-STEP INSTRUCTIONS:")
    print()
    print("1. Open http://127.0.0.1:7861 in your browser")
    print("2. Click on the 'Experiment Lab' tab")
    print("3. Scroll down and click 'Submit Results' button")
    print("4. Fill out the form with these values:")
    print("   • Experiment ID: 1")
    print("   • User ID: demo_user_001")
    print("   • Taste Score: 8.5")
    print("   • Extraction Notes: 'Bright acidity, floral notes, clean finish'")
    print("   • Actual Brew Time: 240")
    print("   • Temperature Consistency: Good")
    print("   • Overall Satisfaction: 9")
    print("   • Additional Notes: 'Excellent cup, would brew again'")
    print("5. Click 'Submit Experiment Result'")
    print("6. You should see a success message")
    print("7. Enter 'demo_user_001' in the User ID field in the History section")
    print("8. Click 'Get History' to see your submitted result")
    print()
    print("🔍 CURRENT DATABASE STATE:")
    show_database_contents()
    print()
    print("💡 AFTER SUBMITTING THROUGH THE WEB INTERFACE:")
    print("   Run this script again to see the new data!")
    print(
        "   Or check manually with: sqlite3 cofferl.db 'SELECT * FROM experiment_results;'"
    )


if __name__ == "__main__":
    main()
