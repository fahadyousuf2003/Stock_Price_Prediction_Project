from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    itickers: list
    ptickers: list
    istart_date: str
    iend_date: str
    pstart_date: str
    pend_date: str
    internationalStocks_local_data_dir: Path
    pakistanStocks_local_data_dir: Path
    
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
    silk_data_dir: Path
    pace_data_dir: Path
    fauji_data_dir: Path
    punjab_data_dir: Path
    transformed_silk_data_dir: Path
    transformed_pace_data_dir: Path
    transformed_fauji_data_dir: Path
    transformed_punjab_data_dir: Path
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    apple_transformed_data_dir: Path
    amazon_transformed_data_dir: Path
    google_transformed_data_dir: Path
    microsoft_transformed_data_dir: Path
    silk_transformed_data_dir: Path
    pace_transformed_data_dir:Path
    fauji_transformed_data_dir:Path
    punjab_transformed_data_dir:Path
    apple_model_name: Path
    amazon_model_name: Path
    google_model_name: Path
    microsoft_model_name: Path
    silk_model_name: Path
    pace_model_name: Path
    fauji_model_name: Path
    punjab_model_name: Path
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
    silk_model_dir:Path
    pace_model_dir:Path
    fauji_model_dir:Path
    punjab_model_dir:Path
    apple_raw_data_dir:Path
    amazon_raw_data_dir:Path
    google_raw_data_dir:Path
    microsoft_raw_data_dir:Path
    silk_raw_data_dir:Path
    pace_raw_data_dir:Path
    fauji_raw_data_dir:Path
    punjab_raw_data_dir:Path
    
    