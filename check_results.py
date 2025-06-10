#!/usr/bin/env python3
"""
Quick script to check for new experiment results.
"""

import sqlite3


def check_results():
    """Check for experiment results in the database."""
    conn = sqlite3.connect("cofferl.db")
    cursor = conn.cursor()

    print("🔍 Checking for experiment results...")
    print("=" * 50)

    # Check results
    cursor.execute(
        """
        SELECT
            r.id,
            r.experiment_id,
            r.user_id,
            r.taste_score,
            r.notes,
            r.actual_brew_time,
            r.recorded_at
        FROM experiment_results r
        ORDER BY r.recorded_at DESC
    """
    )

    results = cursor.fetchall()

    if results:
        print(f"✅ Found {len(results)} experiment result(s):")
        for result in results:
            print(f"   📝 Result ID: {result[0]}")
            print(f"      Experiment: {result[1]} | User: {result[2]}")
            print(f"      Taste Score: {result[3]}/10")
            print(f"      Brew Time: {result[5]}s")
            print(f"      Notes: {result[4]}")
            print(f"      Submitted: {result[6]}")
            print()
    else:
        print("❌ No experiment results found yet.")
        print("   Submit a result through the web interface first!")

    # Check total interactions
    cursor.execute("SELECT COUNT(*) FROM user_interactions")
    interaction_count = cursor.fetchone()[0]
    print(f"📊 Total user interactions: {interaction_count}")

    conn.close()


if __name__ == "__main__":
    check_results()
