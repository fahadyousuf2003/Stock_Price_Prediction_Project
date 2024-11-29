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
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    apple_transformed_data_dir: Path
    amazon_transformed_data_dir: Path
    google_transformed_data_dir: Path
    microsoft_transformed_data_dir: Path
    apple_model_name: Path
    amazon_model_name: Path
    google_model_name: Path
    microsoft_model_name: Path
    layer1: int
    layer2: int
    layer3: int
    layer4: int
    optimizer: str
    loss: str
    batch_size: int
    epochs: int
    
    
@dataclass(frozen=True)
class EvaluationConfig:
    apple_model_dir:Path
    amazon_model_dir:Path
    google_model_dir:Path
    microsoft_model_dir:Path
    apple_raw_data_dir:Path
    amazon_raw_data_dir:Path
    google_raw_data_dir:Path
    microsoft_raw_data_dir:Path
    
    