#!/usr/bin/env python3
"""
Complete workflow for processing community contribution data.
Run this script after collecting CSV responses from Google Forms or web form.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from community_collection import CommunityDataProcessor
import json

def main():
    """Process community CSV data into training examples."""
    
    # Initialize processor
    processor = CommunityDataProcessor()
    print("🚀 Community Data Processor initialized!")
    
    # Check for CSV file
    csv_file = "data/community/responses.csv"
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("📝 Please place your Google Forms CSV export at this location.")
        print("💡 You can use data/community/example_responses.csv as a template.")
        return
    
    try:
        # Process CSV responses
        print(f"📊 Processing CSV file: {csv_file}")
        examples = processor.process_csv_responses(csv_file)
        print(f"✅ Processed {len(examples)} raw examples")
        
        # Validate and filter examples
        print("🔍 Validating examples...")
        valid_examples = processor.validate_and_filter(examples)
        print(f"✅ {len(valid_examples)} examples passed validation")
        
        if len(valid_examples) == 0:
            print("❌ No valid examples found. Check your CSV format.")
            return
        
        # Save processed examples
        print("💾 Saving processed examples...")
        output_file = processor.save_examples(valid_examples)
        print(f"✅ Saved to: {output_file}")
        
        # Show summary
        print("\n📈 Summary:")
        print(f"  • Raw examples: {len(examples)}")
        print(f"  • Valid examples: {len(valid_examples)}")
        print(f"  • Success rate: {len(valid_examples)/len(examples)*100:.1f}%")
        
        # Show sample output
        if valid_examples:
            print("\n🔍 Sample processed example:")
            print(json.dumps(valid_examples[0], indent=2))
        
        print(f"\n🎯 Next steps:")
        print(f"  1. Review the output file: {output_file}")
        print(f"  2. Merge with existing data if needed")
        print(f"  3. Continue collecting until you reach 200 examples!")
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        print("💡 Check your CSV format matches the example_responses.csv")

if __name__ == "__main__":
    main() 