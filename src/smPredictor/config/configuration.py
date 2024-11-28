from smPredictor.constants import *
from smPredictor.utils.common import read_yaml, create_directories
from smPredictor.entity.config_entity import DataIngestionConfig


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
        if not isinstance(config.tickers, list):
            raise ValueError("Tickers should be a list in the configuration file.")

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            tickers=config.tickers,  # No change in how tickers is passed
            start_date=config.start_date,
            end_date=config.end_date,
            local_data_dir=config.local_data_dir
        )
        
        return data_ingestion_config
