from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    tickers: list
    start_date: str
    end_date: str
    local_data_dir: Path
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    apple_data_dir: Path
    amazon_data_dir: Path
    google_data_dir: Path
    microsoft_data_dir: Path
    transformed_apple_data_dir: Path
    transformed_amazon_data_dir: Path
    transformed_google_data_dir: Path
    transformed_microsoft_data_dir: Path