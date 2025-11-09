import sys
import os
import pytest

# --- Ensure Python can find the src/ package ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# --- Import inference functions ---
from src.inference import load_model, predict


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Fixture to load the model and tokenizer once for all tests."""
    # Dynamically locate the model folder relative to the test file
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../final_model"))
    if not os.path.exists(model_path):
        pytest.skip(f"⚠️ Skipping tests: model folder not found at {model_path}")
    model, tokenizer = load_model(model_path)
    return model, tokenizer


def test_inference_with_dummy_input(model_and_tokenizer):
    """Test prediction on a single text input."""
    model, tokenizer = model_and_tokenizer
    sample_text = "This movie was fantastic and inspiring!"
    results = predict(sample_text, model, tokenizer)

    assert isinstance(results, list)
    assert len(results) == 1

    output = results[0]
    assert "text" in output and "label" in output and "score" in output
    assert output["label"] in ["positive", "negative"]
    assert 0 <= output["score"] <= 1


def test_inference_batch_mode(model_and_tokenizer):
    """Test prediction on multiple text inputs."""
    model, tokenizer = model_and_tokenizer
    texts = ["Loved it!", "It was awful."]
    results = predict(texts, model, tokenizer)

    assert isinstance(results, list)
    assert len(results) == len(texts)

    for r in results:
        assert "label" in r and "score" in r


def test_empty_input_handling(model_and_tokenizer):
    """Test handling of empty input strings."""
    model, tokenizer = model_and_tokenizer
    results = predict("", model, tokenizer)

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["text"] == ""


def test_invalid_input_type(model_and_tokenizer):
    """Test handling of invalid (non-string) input."""
    model, tokenizer = model_and_tokenizer
    try:
        predict(None, model, tokenizer)
    except Exception as e:
        pytest.fail(f"Inference raised an unexpected exception: {e}")
