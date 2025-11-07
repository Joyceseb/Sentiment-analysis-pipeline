import pandas as pd 
import numpy as np
import os 

def load_data(file_path: str) -> pd.DataFrame | None:
    """
    Load a CSV file into a pandas DataFrame.
    Handles errors such as missing files, permission issues, or invalid formats.

    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from '{file_path}'.")
        print(f"Data shape: {df.shape}")
        return df

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied for file '{file_path}'.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: The file '{file_path}' has an invalid format (parse error).")
    except Exception as e:
        print(f"Error: Unexpected error while reading '{file_path}': {e}")

    return None

if __name__ == "__main__":
    file_path = "dataset (1).csv"  
    df = load_data(file_path)
