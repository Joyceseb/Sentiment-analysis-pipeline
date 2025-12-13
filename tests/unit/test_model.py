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


def test_model_training_flow():
    """Test the full training script flow using mocks."""
    # Mock datasets and evaluate using patch.dict on sys.modules
    # We also mock transformers to prevent real imports, but we must be careful not to break other tests
    # Use patch.dict to safely mock sys.modules for this scope only
    mock_modules = {
        "datasets": MagicMock(),
        "evaluate": MagicMock(),
        "transformers": MagicMock(),
        "torch": MagicMock(),
    }
    
    with patch.dict(sys.modules, mock_modules):
        # Now it is safe to import src.model
        # We need to ensure src.model is re-imported if it was already loaded, 
        # or imported fresh. Since we are inside a patch.dict, the imports inside src.model 
        # will use the mocks in sys.modules.
        
        # Note: if src.model depends on 'transformers', it will get the MagicMock.
        if "src.model" in sys.modules:
            del sys.modules["src.model"]
            
        import src.model
        importlib.reload(src.model) 
        
        # Configure the mocks that are now part of src.model
        # src.model imported transformers, so src.model.BertTokenizer is the Mock
        
        # Setup mocks on the imported module (which are already MagicMocks)
        src.model.load_dataset.return_value.map.return_value = {"train": MagicMock(), "test": MagicMock()}
        
        # Run main
        src.model.main()
        
        # Verify calls
        src.model.load_dataset.assert_called_with("imdb")
        src.model.Trainer.assert_called()
        src.model.Trainer.return_value.train.assert_called()
        src.model.Trainer.return_value.save_model.assert_called_with("./final_model")


def test_model_training_error():
    """Test error handling in main."""
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
        
        # Make load_dataset raise generic error
        src.model.load_dataset.side_effect = ValueError("Test Error")
        
        # We need to patch logger specifically because it's usually set up at module level
        with patch("src.model.logger") as mock_logger:
            with pytest.raises(ValueError, match="Test Error"):
                src.model.main()
            mock_logger.error.assert_called()