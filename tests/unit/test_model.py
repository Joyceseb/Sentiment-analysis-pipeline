# tests/unit/test_model.py

import torch
import pytest
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load pretrained model and tokenizer once for all tests."""
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model, tokenizer


def test_model_instantiation(model_and_tokenizer):
    """Ensure the model and tokenizer load correctly."""
    model, tokenizer = model_and_tokenizer
    assert model is not None
    assert tokenizer is not None
    assert hasattr(model, "forward")


def test_forward_pass_output_shape(model_and_tokenizer):
    """Run a dummy batch through the model and verify logits shape."""
    model, tokenizer = model_and_tokenizer
    dummy_texts = ["I love this!", "This was terrible."]
    inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True)

    outputs = model(**inputs)
    logits = outputs.logits

    # Expect 2 samples × 2 labels
    assert logits.shape == (2, 2)
    assert isinstance(logits, torch.Tensor)


def test_loss_computation(model_and_tokenizer):
    """Validate loss computation using dummy labels."""
    model, tokenizer = model_and_tokenizer
    dummy_texts = ["Good movie!", "Awful film."]
    dummy_labels = torch.tensor([1, 0])

    inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs, labels=dummy_labels)

    assert hasattr(outputs, "loss")
    assert isinstance(outputs.loss.item(), float)


def test_no_runtime_errors(model_and_tokenizer):
    """Ensure no runtime errors occur during forward pass."""
    model, tokenizer = model_and_tokenizer
    dummy_texts = ["Just a quick test."]
    inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True)

    try:
        _ = model(**inputs)
    except Exception as e:
        pytest.fail(f"Model forward pass raised an exception: {e}")