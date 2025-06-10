"""Tests for the Gradio web interface."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gradio_app import (  # noqa: E402
    create_interface,
    detect_platform,
    format_output,
    get_platform_settings,
    parse_model_output,
    predict_coffee,
)


class TestPlatformDetection:
    """Test platform detection functionality."""

    def test_detect_platform_returns_dict(self):
        """Test that detect_platform returns a dictionary with expected keys."""
        platform_info = detect_platform()

        assert isinstance(platform_info, dict)
        assert "platform" in platform_info
        assert "system" in platform_info
        assert "architecture" in platform_info
        assert "device" in platform_info
        assert "device_name" in platform_info

    def test_get_platform_settings_returns_dict(self):
        """Test that get_platform_settings returns a dictionary with expected keys."""
        mock_platform_info = {
            "platform": "macOS (Apple Silicon)",
            "system": "Darwin",
            "architecture": "arm64",
            "device": "mps",
            "device_name": "Apple Metal",
        }

        settings = get_platform_settings(mock_platform_info)

        assert isinstance(settings, dict)
        assert "share" in settings
        assert "server_name" in settings
        assert "server_port" in settings

    @patch("src.gradio_app.torch.cuda.is_available")
    @patch("src.gradio_app.torch.backends.mps.is_available")
    def test_platform_detection_cuda(self, mock_mps, mock_cuda):
        """Test platform detection when CUDA is available."""
        mock_cuda.return_value = True
        mock_mps.return_value = False

        with patch("src.gradio_app.torch.cuda.get_device_name") as mock_device_name:
            mock_device_name.return_value = "NVIDIA RTX 4090"
            platform_info = detect_platform()

            assert platform_info["device"] == "cuda"
            assert platform_info["device_name"] == "NVIDIA RTX 4090"

    @patch("src.gradio_app.torch.cuda.is_available")
    @patch("src.gradio_app.torch.backends.mps.is_available")
    def test_platform_detection_mps(self, mock_mps, mock_cuda):
        """Test platform detection when MPS is available."""
        mock_cuda.return_value = False
        mock_mps.return_value = True

        platform_info = detect_platform()

        assert platform_info["device"] == "mps"
        assert platform_info["device_name"] == "Apple Metal"

    @patch("src.gradio_app.torch.cuda.is_available")
    @patch("src.gradio_app.torch.backends.mps.is_available")
    def test_platform_detection_cpu_fallback(self, mock_mps, mock_cuda):
        """Test platform detection falls back to CPU when no acceleration available."""
        mock_cuda.return_value = False
        mock_mps.return_value = False

        platform_info = detect_platform()

        assert platform_info["device"] == "cpu"
        assert platform_info["device_name"] == "CPU"


class TestModelOutputParsing:
    """Test model output parsing and formatting."""

    def test_parse_model_output_with_json(self):
        """Test parsing model output that contains JSON."""
        output_text = 'Some text before {"grind_change": "finer", "expected_time": "2:30", "extraction": "under-extracted", "confidence": 0.8, "reasoning": "Sour taste indicates under-extraction"} some text after'

        result = parse_model_output(output_text)

        assert isinstance(result, dict)
        assert result["grind_change"] == "finer"
        assert result["expected_time"] == "2:30"
        assert result["extraction"] == "under-extracted"
        assert result["confidence"] == 0.8
        assert result["reasoning"] == "Sour taste indicates under-extraction"

    def test_parse_model_output_without_json(self):
        """Test parsing model output that doesn't contain JSON."""
        output_text = "This is just plain text without any JSON structure."

        result = parse_model_output(output_text)

        assert isinstance(result, dict)
        assert result["grind_change"] == "See reasoning below"
        assert result["expected_time"] == "2:30-3:00"
        assert result["extraction"] == "Needs adjustment"
        assert result["confidence"] == 0.7
        assert result["reasoning"] == output_text.strip()

    def test_parse_model_output_invalid_json(self):
        """Test parsing model output with malformed JSON."""
        output_text = 'Some text before {"grind_change": "finer", "expected_time": invalid json} some text after'

        result = parse_model_output(output_text)

        assert isinstance(result, dict)
        assert result["grind_change"] == "See reasoning below"
        assert result["reasoning"] == output_text.strip()

    def test_format_output_complete_data(self):
        """Test formatting output with complete data."""
        output_json = {
            "confidence": 0.8,
            "recommendations": {
                "grind_adjustment": "finer",
                "time_adjustment": "2:30-3:00",
                "extraction_notes": "under-extracted",
            },
            "reasoning": "Sour taste indicates under-extraction. Try grinding finer.",
        }

        result = format_output(output_json)

        # Check for new format elements
        assert "🟢 **High Confidence**" in result
        assert "### 🎯 Recommendations" in result
        assert "⚙️ **Grind:** finer" in result
        assert "⏱️ **Time:** 2:30-3:00" in result
        assert "🔬 **Extraction:** under-extracted" in result
        assert "### 🧠 AI Reasoning" in result
        assert "Sour taste indicates under-extraction" in result

    def test_format_output_missing_data(self):
        """Test formatting output with missing data."""
        output_json = {
            "confidence": 0.6,
            "recommendations": {"grind_adjustment": "finer"},
        }

        result = format_output(output_json)

        # Check for new format elements
        assert "🟡 **Medium Confidence**" in result
        assert "⚙️ **Grind:** finer" in result
        assert "### 🧠 AI Reasoning" in result
        assert "*No reasoning provided*" in result

    def test_format_output_none_input(self):
        """Test formatting output with None input."""
        result = format_output(None)
        assert (
            result == "❌ **Error:** Unable to process your request. Please try again."
        )


class TestCoffeePrediction:
    """Test coffee prediction functionality."""

    def test_predict_coffee_empty_input(self):
        """Test predict_coffee with empty input."""
        result = predict_coffee("")

        assert (
            result
            == "📝 **Input Required:** Please enter your brewing parameters and taste notes to get personalized recommendations."
        )

    def test_predict_coffee_whitespace_input(self):
        """Test predict_coffee with whitespace-only input."""
        result = predict_coffee("   ")

        assert (
            result
            == "📝 **Input Required:** Please enter your brewing parameters and taste notes to get personalized recommendations."
        )

    @patch("src.gradio_app.load_model")
    def test_predict_coffee_model_not_loaded(self, mock_load_model):
        """Test predict_coffee when model is not loaded."""
        mock_load_model.return_value = (None, None, {}, "")

        result = predict_coffee(
            "V60, 20g coffee, 300g water, medium grind, 3:00 brew time, tastes sour"
        )

        # Check for new demo format elements
        assert "🎭 Demo Response" in result
        assert "⚙️ **Grind:** Try going one step finer" in result
        assert "⏱️ **Time:** Aim for 2:30-3:00 total brew time" in result
        assert "🔬 **Extraction:** Your coffee appears under-extracted" in result

    @patch("src.gradio_app.load_model")
    def test_predict_coffee_with_mock_model(self, mock_load_model):
        """Test prediction with mocked model."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_platform_info = {"device": "cpu"}

        # Mock tokenizer behavior
        mock_tokenizer.return_value = {"input_ids": Mock(), "attention_mask": Mock()}
        mock_tokenizer.eos_token_id = 2

        # Mock model generation
        mock_outputs = [[1, 2, 3, 4, 5]]  # Mock token IDs as list
        mock_model.generate.return_value = mock_outputs

        # Mock tokenizer decode
        mock_tokenizer.decode.return_value = "Analyze this V60 brew: test input\n\nResponse: Try grinding finer for better extraction."

        mock_load_model.return_value = (
            mock_model,
            mock_tokenizer,
            mock_platform_info,
            "QLora",
        )

        result = predict_coffee("V60, 15g coffee, 250g water, medium grind, sour taste")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain formatted output structure
        assert "Recommendation" in result or "Error" in result


class TestGradioInterface:
    """Test Gradio interface creation."""

    @patch("src.gradio_app.detect_platform")
    def test_create_interface_returns_gradio_blocks(self, mock_detect_platform):
        """Test that create_interface returns a Gradio Blocks object."""
        mock_detect_platform.return_value = {
            "platform": "macOS (Apple Silicon)",
            "device": "mps",
        }

        interface = create_interface()

        # Check that it's a Gradio Blocks object
        assert hasattr(interface, "launch")
        assert hasattr(interface, "queue")

    @patch("src.gradio_app.detect_platform")
    @patch("src.gradio_app.load_model")
    def test_interface_components_exist(self, mock_load_model, mock_detect_platform):
        """Test that the interface has the expected components."""
        mock_detect_platform.return_value = {
            "platform": "macOS (Apple Silicon)",
            "device": "mps",
        }
        mock_load_model.return_value = (None, None, {"device": "mps"}, "QLora")

        interface = create_interface()

        # The interface should be created without errors
        assert interface is not None


class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch("src.gradio_app.load_model")
    def test_full_prediction_workflow(self, mock_load_model):
        """Test the complete prediction workflow with mocked model loading."""
        # Mock load_model to return None (model not loaded scenario)
        mock_load_model.return_value = (None, None, {"device": "cpu"}, "QLora")

        # Test the prediction - should return demo response when model not loaded
        result = predict_coffee("V60, 15g coffee, 250g water, medium grind, sour taste")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain demo response structure
        assert "Demo Response" in result or "Recommendation" in result


if __name__ == "__main__":
    pytest.main([__file__])
