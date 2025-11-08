# tests/unit/test_data_preprocessing.py

import pytest
import pandas as pd
from src.data_processing import clean_text, preprocess_data


def test_clean_text_basic_cleaning():
    """Test that URLs, mentions, and hashtags are removed."""
    raw_text = "Visit http://example.com @user #coolStuff Great movie!!!"
    cleaned = clean_text(raw_text)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "great movie!!!" in cleaned


def test_clean_text_lowercase_and_whitespace():
    """Ensure lowercase conversion and space normalization work correctly."""
    raw_text = "  HELLO   WORLD!!  "
    cleaned = clean_text(raw_text)
    assert cleaned == "hello world!!"


def test_preprocess_data_tokenization_and_split():
    """Check that tokenization and train/validation split are correct."""
    # Mock dataset
    df = pd.DataFrame({
        "text": [
            "I loved this movie!",
            "Terrible film...",
            "It was just okay.",
            "Amazing performance!",
            "Bad script but great actors."
        ],
        "label": [1, 0, 1, 1, 0]
    })

    train_df, val_df, enc_train, enc_val = preprocess_data(df)

    # 1. Split ratio (default 80/20)
    total = len(df)
    expected_train = int(total * 0.8)
    expected_val = total - expected_train
    assert len(train_df) == expected_train
    assert len(val_df) == expected_val

    # 2. Tokenization outputs
    assert "input_ids" in enc_train
    assert "attention_mask" in enc_train
    assert len(enc_train["input_ids"]) == len(train_df)
    assert len(enc_val["input_ids"]) == len(val_df)

    # 3. Check token IDs (integers, not empty)
    first_token_list = enc_train["input_ids"][0]
    assert isinstance(first_token_list[0], int)
    assert len(first_token_list) > 0


def test_preprocess_data_handles_missing_texts():
    """Ensure rows with missing text are dropped."""
    df = pd.DataFrame({
        "text": ["Good movie", None, "Bad movie"],
        "label": [1, 0, 0]
    })
    train_df, val_df, _, _ = preprocess_data(df)
    assert train_df["clean_text"].isna().sum() == 0
    assert val_df["clean_text"].isna().sum() == 0