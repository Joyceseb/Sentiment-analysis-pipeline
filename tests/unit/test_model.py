# tests/unit/test_model.py

import torch
import pytest
import sys
import importlib
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load pretrained model and tokenizer once for all tests."""
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model, tokenizer


def test_model_instantiation(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    assert model is not None
    assert tokenizer is not None
    assert hasattr(model, "forward")


def test_forward_pass_output_shape(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    dummy_texts = ["I love this!", "This was terrible."]
    inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True)

    outputs = model(**inputs)
    logits = outputs.logits

    assert logits.shape == (2, 2)
    assert isinstance(logits, torch.Tensor)


def test_loss_computation(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    dummy_texts = ["Good movie!", "Awful film."]
    dummy_labels = torch.tensor([1, 0])

    inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs, labels=dummy_labels)

    assert hasattr(outputs, "loss")
    assert isinstance(outputs.loss.item(), float)


def test_no_runtime_errors(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    dummy_texts = ["Just a quick test."]
    inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True)

    try:
        _ = model(**inputs)
    except Exception as e:
        pytest.fail(f"Model forward pass raised an exception: {e}")


def test_model_training_flow():
    """Test full training flow using mocks."""

    mock_modules = {
        "datasets": MagicMock(),
        "evaluate": MagicMock(),
        "transformers": MagicMock(),
        "torch": MagicMock(),
        "os": MagicMock(),
    }

    with patch.dict(sys.modules, mock_modules):

        # Force clean import
        if "src.model" in sys.modules:
            del sys.modules["src.model"]

        import src.model
        importlib.reload(src.model)

        # Mock dataset.map output
        src.model.load_dataset.return_value.map.return_value = {
            "train": MagicMock(),
            "test": MagicMock(),
        }

        # Mock Trainer
        src.model.Trainer.return_value.train = MagicMock()

        # Mock trainer.model (the actual model saved)
        src.model.Trainer.return_value.model = MagicMock()
        src.model.Trainer.return_value.model.save_pretrained = MagicMock()

        # Mock tokenizer save_pretrained
        src.model.BertTokenizer.from_pretrained.return_value.save_pretrained = MagicMock()

        # Ensure "./final_model" is considered missing
        src.model.os.path.exists.return_value = False

        # ACT
        src.model.main()

        # ASSERTIONS
        src.model.load_dataset.assert_called_with("imdb")
        src.model.Trainer.assert_called()
        src.model.Trainer.return_value.train.assert_called()

        # Assert model saving
        src.model.Trainer.return_value.model.save_pretrained.assert_called_with("./final_model")

        # Assert tokenizer saving
        src.model.BertTokenizer.from_pretrained.return_value.save_pretrained.assert_called_with("./final_model")


def test_model_training_error():
    """Ensure main raises and logs on dataset loading error."""

    mock_modules = {
        "datasets": MagicMock(),
        "evaluate": MagicMock(),
        "transformers": MagicMock(),
    }

    with patch.dict(sys.modules, mock_modules):

        if "src.model" in sys.modules:
            del sys.modules["src.model"]

        import src.model
        importlib.reload(src.model)

        # Force error
        src.model.load_dataset.side_effect = ValueError("Test Error")

        with patch("src.model.logger") as mock_logger:
            with pytest.raises(ValueError, match="Test Error"):
                src.model.main()
            mock_logger.error.assert_called()
