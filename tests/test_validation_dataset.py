#!/usr/bin/env python3
"""
Tests for the validation dataset creation script.

This module tests all functionality of the create_validation_dataset.py script
including data loading, categorization, balancing, and saving.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import pandas as pd
from datasets import Dataset

# Import the functions we want to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from create_validation_dataset import (
    load_all_data_sources,
    categorize_by_extraction,
    select_high_quality_examples,
    create_balanced_validation_set,
    save_datasets,
    analyze_dataset_balance
)


class TestLoadAllDataSources:
    """Test data loading functionality."""
    
    def test_load_raw_dataset_only(self, tmp_path):
        """Test loading only raw dataset."""
        # Create test data
        test_data = [
            {
                "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
                "output": {
                    "grind_change": "coarser_2",
                    "reasoning": "Bitter taste indicates over-extraction",
                    "expected_time": "3:30",
                    "extraction": "good",
                    "confidence": 0.9
                }
            }
        ]
        
        # Create test file
        raw_file = tmp_path / "coffee_dataset_raw.json"
        with open(raw_file, 'w') as f:
            json.dump(test_data, f)
        
        # Test loading
        result = load_all_data_sources(tmp_path)
        assert len(result) == 1
        assert result[0]["input"] == test_data[0]["input"]
    
    def test_load_all_data_sources(self, tmp_path):
        """Test loading all data sources."""
        # Create test data for each source
        raw_data = [{"input": "raw_test", "output": {"extraction": "good"}}]
        manual_data = [{"input": "manual_test", "output": {"extraction": "under"}}]
        community_data = [{"input": "community_test", "output": {"extraction": "over"}}]
        
        # Create test files
        with open(tmp_path / "coffee_dataset_raw.json", 'w') as f:
            json.dump(raw_data, f)
        
        with open(tmp_path / "manual_examples.json", 'w') as f:
            json.dump(manual_data, f)
        
        community_dir = tmp_path / "community"
        community_dir.mkdir()
        with open(community_dir / "community_examples_test.json", 'w') as f:
            json.dump(community_data, f)
        
        # Test loading
        result = load_all_data_sources(tmp_path)
        assert len(result) == 3
        
        # Check that all sources are included
        inputs = [ex["input"] for ex in result]
        assert "raw_test" in inputs
        assert "manual_test" in inputs
        assert "community_test" in inputs
    
    def test_load_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        result = load_all_data_sources(tmp_path)
        assert result == []


class TestCategorizeByExtraction:
    """Test extraction categorization functionality."""
    
    def test_categorize_explicit_extraction(self):
        """Test categorization with explicit extraction labels."""
        examples = [
            {"output": {"extraction": "under"}},
            {"output": {"extraction": "good"}},
            {"output": {"extraction": "over"}},
            {"output": {"extraction": "good"}},
        ]
        
        result = categorize_by_extraction(examples)
        
        assert len(result["under"]) == 1
        assert len(result["good"]) == 2
        assert len(result["over"]) == 1
    
    def test_categorize_from_grind_change(self):
        """Test categorization using grind change fallback."""
        examples = [
            {"output": {"grind_change": "finer_2", "reasoning": "test"}},
            {"output": {"grind_change": "coarser_1", "reasoning": "test"}},
            {"output": {"grind_change": "none", "reasoning": "test"}},
        ]
        
        result = categorize_by_extraction(examples)
        
        assert len(result["under"]) == 1  # finer grind
        assert len(result["over"]) == 1   # coarser grind
        assert len(result["good"]) == 1   # no change
    
    def test_categorize_from_reasoning(self):
        """Test categorization using reasoning fallback."""
        examples = [
            {"output": {"grind_change": "unknown", "reasoning": "under-extraction detected"}},
            {"output": {"grind_change": "unknown", "reasoning": "over-extraction problem"}},
            {"output": {"grind_change": "unknown", "reasoning": "perfect balance"}},
        ]
        
        result = categorize_by_extraction(examples)
        
        assert len(result["under"]) == 1
        assert len(result["over"]) == 1
        assert len(result["good"]) == 1


class TestSelectHighQualityExamples:
    """Test high-quality example selection."""
    
    def test_select_by_confidence(self):
        """Test selection prioritizes high confidence examples."""
        categories = {
            "good": [
                {"input": "test1", "output": {"confidence": 0.9}},
                {"input": "test2", "output": {"confidence": 0.7}},
                {"input": "test3", "output": {"confidence": 0.95}},
            ]
        }
        
        result = select_high_quality_examples(categories)
        
        # Should be sorted by confidence (descending)
        confidences = [ex["output"]["confidence"] for ex in result]
        assert confidences == [0.95, 0.9, 0.7]
    
    def test_remove_duplicates(self):
        """Test duplicate removal based on input similarity."""
        categories = {
            "good": [
                {"input": "V60, 20g coffee, 300g water", "output": {"confidence": 0.9}},
                {"input": "V60,20gcoffee,300gwater", "output": {"confidence": 0.8}},  # Duplicate
                {"input": "V60, 25g coffee, 350g water", "output": {"confidence": 0.85}},
            ]
        }
        
        result = select_high_quality_examples(categories)
        
        # Should remove the duplicate
        assert len(result) == 2
        assert result[0]["output"]["confidence"] == 0.9  # Higher confidence kept


class TestCreateBalancedValidationSet:
    """Test balanced validation set creation."""
    
    def test_balanced_selection(self):
        """Test that validation set is balanced across categories."""
        examples = []
        
        # Create 50 examples of each category
        for i in range(50):
            examples.extend([
                {"input": f"under_{i}", "output": {"extraction": "under", "confidence": 0.8}},
                {"input": f"good_{i}", "output": {"extraction": "good", "confidence": 0.8}},
                {"input": f"over_{i}", "output": {"extraction": "over", "confidence": 0.8}},
            ])
        
        validation, training = create_balanced_validation_set(examples, validation_size=99)
        
        # Check validation set size
        assert len(validation) == 99
        
        # Check balance (should be roughly 33 each)
        val_categories = categorize_by_extraction(validation)
        assert len(val_categories["under"]) >= 30
        assert len(val_categories["good"]) >= 30
        assert len(val_categories["over"]) >= 30
        
        # Check no overlap
        val_inputs = {ex["input"] for ex in validation}
        train_inputs = {ex["input"] for ex in training}
        assert len(val_inputs.intersection(train_inputs)) == 0
    
    def test_small_dataset_handling(self):
        """Test handling of datasets smaller than validation size."""
        examples = [
            {"input": "test1", "output": {"extraction": "good", "confidence": 0.8}},
            {"input": "test2", "output": {"extraction": "under", "confidence": 0.8}},
        ]
        
        validation, training = create_balanced_validation_set(examples, validation_size=100)
        
        # Should use all available examples for validation
        assert len(validation) == 2
        assert len(training) == 0


class TestSaveDatasets:
    """Test dataset saving functionality."""
    
    def test_save_json_files(self, tmp_path):
        """Test saving JSON files."""
        validation_examples = [{"input": "val_test", "output": {"extraction": "good"}}]
        training_examples = [{"input": "train_test", "output": {"extraction": "under"}}]
        
        save_datasets(validation_examples, training_examples, tmp_path)
        
        # Check JSON files exist
        assert (tmp_path / "validation_examples.json").exists()
        assert (tmp_path / "training_examples.json").exists()
        
        # Check content
        with open(tmp_path / "validation_examples.json") as f:
            saved_val = json.load(f)
        assert saved_val == validation_examples
        
        with open(tmp_path / "training_examples.json") as f:
            saved_train = json.load(f)
        assert saved_train == training_examples
    
    def test_save_hf_datasets(self, tmp_path):
        """Test saving Hugging Face datasets."""
        validation_examples = [{"input": "val_test", "output": {"extraction": "good"}}]
        training_examples = [{"input": "train_test", "output": {"extraction": "under"}}]
        
        save_datasets(validation_examples, training_examples, tmp_path)
        
        # Check HF dataset directories exist
        assert (tmp_path / "coffee_validation_dataset").exists()
        assert (tmp_path / "coffee_training_dataset").exists()
        
        # Load and verify datasets
        val_dataset = Dataset.load_from_disk(str(tmp_path / "coffee_validation_dataset"))
        train_dataset = Dataset.load_from_disk(str(tmp_path / "coffee_training_dataset"))
        
        assert len(val_dataset) == 1
        assert len(train_dataset) == 1


class TestAnalyzeDatasetBalance:
    """Test dataset analysis functionality."""
    
    def test_analyze_extraction_categories(self, capsys):
        """Test analysis of extraction categories."""
        examples = [
            {"output": {"extraction": "under", "confidence": 0.8, "grind_change": "finer_1"}},
            {"output": {"extraction": "good", "confidence": 0.9, "grind_change": "none"}},
            {"output": {"extraction": "over", "confidence": 0.7, "grind_change": "coarser_2"}},
        ]
        
        analyze_dataset_balance(examples, "Test Dataset")
        
        captured = capsys.readouterr()
        assert "Test Dataset Analysis" in captured.out
        assert "Under extraction: 1" in captured.out
        assert "Good extraction: 1" in captured.out
        assert "Over extraction: 1" in captured.out
        assert "Average confidence: 0.800" in captured.out
    
    def test_analyze_grind_changes(self, capsys):
        """Test analysis of grind change distribution."""
        examples = [
            {"output": {"grind_change": "finer_1", "confidence": 0.8, "extraction": "good"}},
            {"output": {"grind_change": "finer_1", "confidence": 0.9, "extraction": "good"}},
            {"output": {"grind_change": "coarser_2", "confidence": 0.7, "extraction": "good"}},
        ]
        
        analyze_dataset_balance(examples, "Test Dataset")
        
        captured = capsys.readouterr()
        assert "finer_1: 2" in captured.out
        assert "coarser_2: 1" in captured.out


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_end_to_end_workflow(self, tmp_path):
        """Test the complete validation dataset creation workflow."""
        # Create comprehensive test data
        test_data = []
        
        # Create diverse examples across categories
        for i in range(30):
            test_data.extend([
                {
                    "input": f"V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes sour {i}",
                    "output": {
                        "grind_change": "finer_2",
                        "reasoning": "Sour taste indicates under-extraction",
                        "expected_time": "2:45",
                        "extraction": "under",
                        "confidence": 0.8 + (i % 3) * 0.05
                    }
                },
                {
                    "input": f"V60, 20g coffee, 300g water, medium grind, 3:30 brew time, tastes balanced {i}",
                    "output": {
                        "grind_change": "none",
                        "reasoning": "Perfect balance achieved",
                        "expected_time": "3:30",
                        "extraction": "good",
                        "confidence": 0.85 + (i % 3) * 0.05
                    }
                },
                {
                    "input": f"V60, 20g coffee, 300g water, medium grind, 4:00 brew time, tastes bitter {i}",
                    "output": {
                        "grind_change": "coarser_2",
                        "reasoning": "Bitter taste indicates over-extraction",
                        "expected_time": "4:30",
                        "extraction": "over",
                        "confidence": 0.75 + (i % 3) * 0.05
                    }
                }
            ])
        
        # Save test data
        with open(tmp_path / "coffee_dataset_raw.json", 'w') as f:
            json.dump(test_data, f)
        
        # Run the workflow
        all_examples = load_all_data_sources(tmp_path)
        validation, training = create_balanced_validation_set(all_examples, validation_size=30)
        
        output_dir = tmp_path / "output"
        save_datasets(validation, training, output_dir)
        
        # Verify results
        assert len(all_examples) == 90  # 30 * 3 categories
        assert len(validation) == 30
        assert len(training) == 60
        
        # Check balance
        val_categories = categorize_by_extraction(validation)
        assert len(val_categories["under"]) == 10
        assert len(val_categories["good"]) == 10
        assert len(val_categories["over"]) == 10
        
        # Check files were created
        assert (output_dir / "validation_examples.json").exists()
        assert (output_dir / "training_examples.json").exists()
        assert (output_dir / "coffee_validation_dataset").exists()
        assert (output_dir / "coffee_training_dataset").exists()


# Fixtures for pytest
@pytest.fixture
def sample_examples():
    """Provide sample examples for testing."""
    return [
        {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes bitter",
            "output": {
                "grind_change": "coarser_2",
                "reasoning": "Bitter taste indicates over-extraction",
                "expected_time": "3:30",
                "extraction": "over",
                "confidence": 0.9
            }
        },
        {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes sour",
            "output": {
                "grind_change": "finer_2",
                "reasoning": "Sour taste indicates under-extraction",
                "expected_time": "2:45",
                "extraction": "under",
                "confidence": 0.85
            }
        },
        {
            "input": "V60, 20g coffee, 300g water, medium grind, 3:15 brew time, tastes balanced",
            "output": {
                "grind_change": "none",
                "reasoning": "Perfect extraction achieved",
                "expected_time": "3:15",
                "extraction": "good",
                "confidence": 0.95
            }
        }
    ]


if __name__ == "__main__":
    pytest.main([__file__]) 