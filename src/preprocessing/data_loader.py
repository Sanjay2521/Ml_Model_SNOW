"""
Data Loading Module
Handles loading data from various formats (CSV, Excel, JSON)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple
import yaml


class DataLoader:
    """Data loading and initial processing"""

    def __init__(self, config_path: str = None):
        """
        Initialize DataLoader

        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path) if config_path else {}
        self.data_path = self.config.get('data', {}).get('raw_path', 'data/raw/')

    @staticmethod
    def load_config(config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file

        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Pandas DataFrame
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            print(f"Successfully loaded CSV: {file_path}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"Error loading CSV file {file_path}: {str(e)}")
            raise

    def load_excel(self, file_path: str, sheet_name: Union[str, int, List] = 0, **kwargs) -> Union[pd.DataFrame, dict]:
        """
        Load data from Excel file

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name, index, or list of sheets
            **kwargs: Additional arguments for pd.read_excel

        Returns:
            Pandas DataFrame or dict of DataFrames (if multiple sheets)
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            print(f"Successfully loaded Excel: {file_path}")

            if isinstance(df, dict):
                print(f"Loaded {len(df)} sheets")
                for sheet, data in df.items():
                    print(f"  Sheet '{sheet}': {data.shape}")
            else:
                print(f"Shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")

            return df
        except Exception as e:
            print(f"Error loading Excel file {file_path}: {str(e)}")
            raise

    def load_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from JSON file

        Args:
            file_path: Path to JSON file
            **kwargs: Additional arguments for pd.read_json

        Returns:
            Pandas DataFrame
        """
        try:
            df = pd.read_json(file_path, **kwargs)
            print(f"Successfully loaded JSON: {file_path}")
            print(f"Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {str(e)}")
            raise

    def load_incident_data(self, csv_path: str = None, excel_path: str = None) -> Tuple[pd.DataFrame, dict]:
        """
        Load ServiceNow incident data from CSV and Excel files

        Args:
            csv_path: Path to incident CSV file
            excel_path: Path to incident KPI Excel file

        Returns:
            Tuple of (incident_df, kpi_dict)
        """
        # Use config paths if not provided
        if csv_path is None:
            csv_path = Path(self.data_path) / self.config.get('data', {}).get('incident_csv', 'incidents.csv')
        if excel_path is None:
            excel_path = Path(self.data_path) / self.config.get('data', {}).get('kpi_excel', 'kpis.xlsx')

        # Load incident CSV
        incident_df = None
        if Path(csv_path).exists():
            incident_df = self.load_csv(csv_path)
        else:
            print(f"Warning: Incident CSV not found at {csv_path}")

        # Load KPI Excel (all sheets)
        kpi_data = None
        if Path(excel_path).exists():
            kpi_data = self.load_excel(excel_path, sheet_name=None)
        else:
            print(f"Warning: KPI Excel not found at {excel_path}")

        return incident_df, kpi_data

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive information about the dataframe

        Args:
            df: Input dataframe

        Returns:
            Dictionary with data information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }

        return info

    def print_data_summary(self, df: pd.DataFrame, name: str = "Dataset"):
        """Print summary of dataframe"""
        print(f"\n{'='*50}")
        print(f"{name} Summary")
        print(f"{'='*50}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nData Types:")
        print(df.dtypes)
        print(f"\nNull Values:")
        print(df.isnull().sum())
        print(f"\nDuplicate Rows: {df.duplicated().sum()}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"{'='*50}\n")

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets

        Args:
            df: Input dataframe
            target_col: Target column name
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion
            random_state: Random seed
            stratify: Whether to stratify split based on target

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        assert abs(train_size + val_size + test_size - 1.0) < 1e-5, "Sizes must sum to 1"

        # First split: separate test set
        stratify_col = df[target_col] if stratify and target_col in df.columns else None

        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        # Second split: separate train and validation
        val_size_adjusted = val_size / (train_size + val_size)
        stratify_col_train = train_val_df[target_col] if stratify and target_col in train_val_df.columns else None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_col_train
        )

        print(f"Train set: {train_df.shape}")
        print(f"Validation set: {val_df.shape}")
        print(f"Test set: {test_df.shape}")

        return train_df, val_df, test_df

    def save_processed_data(
        self,
        df: pd.DataFrame,
        file_name: str,
        output_path: str = None,
        format: str = 'csv'
    ):
        """
        Save processed data to file

        Args:
            df: Dataframe to save
            file_name: Output file name
            output_path: Output directory path
            format: File format ('csv', 'pickle', 'parquet')
        """
        if output_path is None:
            output_path = self.config.get('data', {}).get('processed_path', 'data/processed/')

        Path(output_path).mkdir(parents=True, exist_ok=True)
        file_path = Path(output_path) / file_name

        if format == 'csv':
            df.to_csv(file_path, index=False)
        elif format == 'pickle':
            df.to_pickle(file_path)
        elif format == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Saved processed data to: {file_path}")


def load_sample_data() -> pd.DataFrame:
    """
    Create sample incident data for testing

    Returns:
        Sample DataFrame
    """
    data = {
        'number': ['INC0001', 'INC0002', 'INC0003', 'INC0004', 'INC0005'],
        'short_description': [
            'Unable to access email',
            'Printer not working',
            'VPN connection issue',
            'Password reset required',
            'Application crash error'
        ],
        'description': [
            'User cannot login to email application. Error message: Connection timeout',
            'Office printer on 3rd floor is not printing. Paper jam error showing',
            'Cannot connect to VPN from home. Authentication fails',
            'User forgot password and needs immediate reset',
            'CRM application crashes when opening reports module'
        ],
        'priority': ['2', '3', '1', '2', '1'],
        'impact': ['2', '3', '1', '2', '1'],
        'urgency': ['2', '3', '1', '2', '1'],
        'category': ['Email', 'Hardware', 'Network', 'Access', 'Software'],
        'assignment_group': ['IT Support', 'Hardware Team', 'Network Team', 'IT Support', 'Application Team']
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()

    # Create sample data
    sample_df = load_sample_data()
    loader.print_data_summary(sample_df, "Sample Incident Data")

    # Save sample data
    loader.save_processed_data(sample_df, 'sample_incidents.csv', 'data/sample/')
    print("Sample data saved successfully!")
