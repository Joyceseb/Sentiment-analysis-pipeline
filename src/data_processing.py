import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def clean_text(text: str) -> str:
    """
    Clean the input text using a Kaggle-inspired approach:
    1. Remove URLs, mentions, and special characters
    2. Convert text to lowercase
    3. Normalize whitespace
    4. Keep basic punctuation (.,!?)
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove @mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove numbers and special characters (keep letters and punctuation)
    text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9.,!?;:()'\s]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_data(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Complete text preprocessing and tokenization pipeline.

    Steps:
    1. Clean text using clean_text()
    2. Tokenize using Hugging Face AutoTokenizer (bert-base-uncased)
    3. Split into training and validation sets

    Returns:
        train_df, val_df, train_encodings, val_encodings
    """
    # Drop missing text rows
    df = df.dropna(subset=[text_column]).reset_index(drop=True)

    # Clean the text column
    df["clean_text"] = df[text_column].apply(clean_text)

    # Split data (stratified if label column available)
    stratify = df[label_column] if label_column in df.columns else None
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify if stratify is not None else None,
    )

    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize datasets
    train_encodings = tokenizer(
        train_df["clean_text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )
    val_encodings = tokenizer(
        val_df["clean_text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )

    return train_df, val_df, train_encodings, val_encodings


if __name__ == "__main__":
    # Example for manual testing
    sample_data = {
        "text": [
            "I LOVED this movie!!! So much better than expected :)",
            "Terrible... I want my time back.",
            "An average film, nothing special.",
            "Check this out: https://imdb.com",
        ],
        "label": [1, 0, 1, 1],
    }

    df = pd.DataFrame(sample_data)
    train_df, val_df, train_enc, val_enc = preprocess_data(df)

    print("✅ Sample cleaning & tokenization complete!")
    print("Train size:", len(train_df), "Validation size:", len(val_df))
    print("First cleaned text:", train_df['clean_text'].iloc[0])