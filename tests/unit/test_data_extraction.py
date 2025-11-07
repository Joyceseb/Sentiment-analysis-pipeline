import unittest
import pandas as pd
import os
import tempfile
import shutil
from unittest import mock
from src.data_extraction import load_data  # Remplace par le bon import


class TestLoadData(unittest.TestCase):
    """Unit tests for load_data function."""

    def setUp(self):
        """Create temporary test directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary directory after each test."""
        shutil.rmtree(self.test_dir)

    def test_load_data_success(self):
        """Test 1: CSV file loads successfully."""
        file_path = os.path.join(self.test_dir, "test_data.csv")
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'C']
        })
        test_data.to_csv(file_path, index=False)

        result = load_data(file_path)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (3, 2))
        self.assertListEqual(list(result.columns), ['col1', 'col2'])

    def test_load_data_file_not_found(self):
        """Test 2: File not found should return None."""
        file_path = "non_existent_file.csv"

        with mock.patch("builtins.print") as mock_print:
            result = load_data(file_path)

        self.assertIsNone(result)
        mock_print.assert_any_call(f"Error: File '{file_path}' does not exist.")

    def test_load_data_empty_file(self):
        """Test 3: Empty file should return None."""
        file_path = os.path.join(self.test_dir, "empty.csv")
        open(file_path, 'w').close()  # Create empty file

        with mock.patch("builtins.print") as mock_print:
            result = load_data(file_path)

        self.assertIsNone(result)
        mock_print.assert_any_call(f"Error: The file '{file_path}' is empty.")

    def test_load_data_invalid_format(self):
        """Test 4: Invalid CSV format should return None."""
        file_path = os.path.join(self.test_dir, "invalid.csv")
        with open(file_path, "w") as f:
            f.write("col1,col2\n1,2\ninvalid,data,extra,column\n3,4")

        with mock.patch("builtins.print") as mock_print:
            result = load_data(file_path)

        self.assertIsNone(result)
        mock_print.assert_any_call(
            f"Error: The file '{file_path}' has an invalid format (parse error)."
        )

    def test_load_data_permission_error(self):
        """Test 5: Permission error should return None."""
        file_path = os.path.join(self.test_dir, "no_permission.csv")
        pd.DataFrame({"col1": [1, 2, 3]}).to_csv(file_path, index=False)

        def mock_read_csv(_):
            raise PermissionError("Permission denied")

        with mock.patch("pandas.read_csv", side_effect=mock_read_csv), \
             mock.patch("builtins.print") as mock_print:
            result = load_data(file_path)

        self.assertIsNone(result)
        mock_print.assert_any_call(f"Error: Permission denied for file '{file_path}'.")

    def test_load_data_with_expected_columns(self):
        """Test 6: File contains expected columns."""
        file_path = os.path.join(self.test_dir, "specific_columns.csv")
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        test_data.to_csv(file_path, index=False)

        result = load_data(file_path)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), ['id', 'name', 'age'])
        self.assertEqual(len(result.columns), 3)


if __name__ == "__main__":
    unittest.main()
