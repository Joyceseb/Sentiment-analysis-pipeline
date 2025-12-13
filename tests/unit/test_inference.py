import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import torch

# --- Ensure Python can find the src/ package ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# --- Import inference functions ---
from src.inference import load_model, predict, tokenize_input, run_inference, decode_predictions, main

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    # Mock behavior of tokenizer call
    tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 2023, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    return tokenizer

@pytest.fixture
def mock_model():
    model = MagicMock()
    # Mock output logits
    model.return_value.logits = torch.tensor([[-2.0, 3.5]]) # Class 1 (Positive)
    return model

def test_load_model_success():
    """Test loading model successfully with mocks."""
    with patch("src.inference.os.path.exists", return_value=True), \
         patch("src.inference.BertTokenizer.from_pretrained") as mock_bert_tok, \
         patch("src.inference.BertForSequenceClassification.from_pretrained") as mock_bert_model:
        
        model, tokenizer = load_model("dummy_path")
        
        assert model is not None
        assert tokenizer is not None
        mock_bert_tok.assert_called_with("dummy_path")
        mock_bert_model.assert_called_with("dummy_path")
        model.eval.assert_called_once()

def test_load_model_file_not_found():
    """Test that FileNotFoundError is raised when path doesn't exist."""
    with patch("src.inference.os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_model("non_existent_path")

def test_tokenize_input(mock_tokenizer):
    """Test tokenization helper function."""
    texts = ["Test sentence"]
    output = tokenize_input(texts, mock_tokenizer)
    
    # Check that tokenizer was called with specific args
    mock_tokenizer.assert_called_with(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    assert "input_ids" in output

def test_run_inference(mock_model):
    """Test inference helper function."""
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    logits = run_inference(mock_model, inputs)
    
    assert torch.equal(logits, torch.tensor([[-2.0, 3.5]]))
    mock_model.assert_called_with(**inputs)

def test_decode_predictions():
    """Test decoding logic."""
    # Logits favoring class 1
    logits_pos = torch.tensor([[-2.0, 3.5]])
    decoded = decode_predictions(logits_pos)
    assert decoded == ["Positive 😊"]
    
    # Logits favoring class 0
    logits_neg = torch.tensor([[3.5, -2.0]])
    decoded = decode_predictions(logits_neg)
    assert decoded == ["Negative 😡"]

def test_predict_flow(mock_model, mock_tokenizer):
    """Test full predict flow."""
    text = "Great movie"
    
    # Mocking internal helpers via patches isn't strictly necessary if we pass mocks,
    # but we want to ensure predict calls them correctly.
    # Since predict calls tokenize_input and run_inference directly, 
    # passing the mocks is sufficient because we are unit testing `predict`'s wiring.
    
    results = predict(text, mock_model, mock_tokenizer)
    
    assert results == ["Positive 😊"]
    mock_tokenizer.assert_called()
    mock_model.assert_called()

def test_predict_list_input(mock_model, mock_tokenizer):
    """Test predict with list input."""
    texts = ["Good", "Bad"]
    
    # Adjust mocks for batch size 2
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1], [2]]),
        "attention_mask": torch.tensor([[1], [1]])
    }
    mock_model.return_value.logits = torch.tensor([[-1.0, 2.0], [2.0, -1.0]]) # Pos, Neg
    
    results = predict(texts, mock_model, mock_tokenizer)
    
    assert results == ["Positive 😊", "Negative 😡"]

def test_main_loop_quit():
    """Test the main loop exits on 'quit'."""
    with patch("builtins.input", side_effect=["quit"]), \
         patch("src.inference.load_model", return_value=(MagicMock(), MagicMock())), \
         patch("src.inference.logger"):
        
        try:
            main()
        except SystemExit:
            pass # Should hit break and exit cleanly

def test_main_loop_inference():
    """Test the main loop runs inference once then quits."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    
    with patch("builtins.input", side_effect=["test text", "quit"]), \
         patch("src.inference.load_model", return_value=(mock_model, mock_tokenizer)), \
         patch("src.inference.predict", return_value=["Positive 😊"]), \
         patch("builtins.print") as mock_print, \
         patch("src.inference.logger"):
            
        main()
        
        # Verify predict was called
        mock_print.assert_any_call("Sentiment: Positive 😊")

def test_main_error_handling():
    """Test that main logs error and raises exception on failure."""
    with patch("src.inference.load_model", side_effect=Exception("Load error")), \
         patch("src.inference.logger") as mock_logger:
        
        with pytest.raises(Exception, match="Load error"):
            main()
        
        mock_logger.error.assert_called()
