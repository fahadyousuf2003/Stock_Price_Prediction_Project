from smPredictor.constants import *
from smPredictor.utils.common import read_yaml, create_directories
from smPredictor.entity.config_entity import (DataIngestionConfig,DataTransformationConfig,ModelTrainerConfig,EvaluationConfig)
from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        # Ensure tickers is a list
        if not isinstance(config.itickers, list):
            raise ValueError("Pakistani Tickers should be a list in the configuration file.")
        if not isinstance(config.ptickers, list):
            raise ValueError("Pakistani Tickers should be a list in the configuration file.")

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            itickers=config.itickers,
            ptickers=config.ptickers,
            istart_date=config.istart_date,
            iend_date=config.iend_date,
            pstart_date=config.pstart_date,
            pend_date=config.pend_date,
            internationalStocks_local_data_dir=config.internationalStocks_local_data_dir,
            pakistanStocks_local_data_dir=config.pakistanStocks_local_data_dir
        )
        
        return data_ingestion_config

    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            apple_data_dir=config.apple_data_dir,
            amazon_data_dir=config.apple_data_dir,
            google_data_dir=config.apple_data_dir,
            microsoft_data_dir=config.apple_data_dir,
            transformed_apple_data_dir= config.transformed_apple_data_dir,
            transformed_amazon_data_dir=config.transformed_amazon_data_dir,
            transformed_google_data_dir=config.transformed_google_data_dir,
            transformed_microsoft_data_dir=config.transformed_microsoft_data_dir,
            silk_data_dir=config.silk_data_dir,
            pace_data_dir=config.pace_data_dir,
            fauji_data_dir=config.fauji_data_dir,
            punjab_data_dir=config.punjab_data_dir,
            transformed_silk_data_dir=config.transformed_silk_data_dir,
            transformed_pace_data_dir=config.transformed_pace_data_dir,
            transformed_fauji_data_dir=config.transformed_fauji_data_dir,
            transformed_punjab_data_dir=config.transformed_punjab_data_dir
            
        )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params
        
        create_directories([config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            apple_transformed_data_dir=config.apple_transformed_data_dir,
            amazon_transformed_data_dir=config.amazon_transformed_data_dir,
            google_transformed_data_dir=config.google_transformed_data_dir,
            microsoft_transformed_data_dir=config.microsoft_transformed_data_dir,
            silk_transformed_data_dir=config.silk_transformed_data_dir,
            pace_transformed_data_dir=config.pace_transformed_data_dir,
            fauji_transformed_data_dir=config.fauji_transformed_data_dir,
            punjab_transformed_data_dir=config.punjab_transformed_data_dir,
            apple_model_name=config.apple_model_name,
            amazon_model_name=config.amazon_model_name,
            google_model_name=config.google_model_name,
            microsoft_model_name=config.microsoft_model_name,
            silk_model_name=config.silk_model_name,
            pace_model_name=config.pace_model_name,
            fauji_model_name=config.fauji_model_name,
            punjab_model_name=config.punjab_model_name,
            layer1=params.layer1,
            layer2=params.layer2,
            layer3=params.layer3,
            layer4=params.layer4,
            optimizer=params.optimizer,
            loss=params.loss,
            batch_size=params.batch_size,
            epochs=params.epochs
        )
        
        return model_trainer_config
    
    def get_validation_config(self) -> EvaluationConfig:
        eval_config= EvaluationConfig(
            apple_model_dir=Path("artifacts/model_trainer/apple_model.keras"),
            amazon_model_dir=Path("artifacts/model_trainer/amazon_model.keras"),
            google_model_dir=Path("artifacts/model_trainer/google_model.keras"),
            microsoft_model_dir=Path("artifacts/model_trainer/microsoft_model.keras"),
            silk_model_dir=Path("artifacts/model_trainer/silk_model.keras"),
            pace_model_dir=Path("artifacts/model_trainer/pace_model.keras"),
            fauji_model_dir=Path("artifacts/model_trainer/fauji_model.keras"),
            punjab_model_dir=Path("artifacts/model_trainer/punjab_model.keras"),
            apple_raw_data_dir=Path("artifacts/data_ingestion/international_stocks/raw_data/AAPL.csv"),
            amazon_raw_data_dir=Path("artifacts/data_ingestion/international_stocks/raw_data/AMZN.csv"),
            google_raw_data_dir=Path("artifacts/data_ingestion/international_stocks/raw_data/GOOG.csv"),
            microsoft_raw_data_dir=Path("artifacts/data_ingestion/international_stocks/raw_data/MSFT.csv"),
            silk_raw_data_dir=Path("artifacts/data_ingestion/pakistan_stocks/raw_data/SILK.csv"),
            pace_raw_data_dir=Path("artifacts/data_ingestion/pakistan_stocks/raw_data/PACE.csv"),
            fauji_raw_data_dir=Path("artifacts/data_ingestion/pakistan_stocks/raw_data/FFL.csv"),
            punjab_raw_data_dir=Path("artifacts/data_ingestion/pakistan_stocks/raw_data/BOP.csv")
            
        )
        return eval_config