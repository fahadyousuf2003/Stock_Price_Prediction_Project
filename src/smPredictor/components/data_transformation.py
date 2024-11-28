import pandas as pd
from smPredictor import logger
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from smPredictor.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_data(self):
        # Transformations for Apple data
        apple_data = pd.read_csv(self.config.apple_data_dir)
        apple_data = apple_data.iloc[1:]
        apple_data.rename(columns={"Price": "Date"}, inplace=True)
        apple_data = apple_data.drop(apple_data.index[0])

        # Keep only 'Close' column and scale the data
        apple_data = apple_data.filter(['Close'])
        apple_dataset = apple_data.values
        apple_training_data_len = int(np.ceil(len(apple_dataset) * 0.95))
        scaler = MinMaxScaler(feature_range=(0, 1))
        apple_scaled_data = scaler.fit_transform(apple_dataset)
        apple_train_data = apple_scaled_data[:apple_training_data_len]

        # Convert to DataFrame
        apple_train_data_df = pd.DataFrame(apple_train_data)

        # Ensure output directory exists
        os.makedirs(self.config.transformed_apple_data_dir, exist_ok=True)

        # Save the transformed Apple data
        apple_train_data_df.to_csv(
            os.path.join(self.config.transformed_apple_data_dir, "transformed_apple.csv"), 
            index=False
        )
        logger.info("Transformed Apple Data saved successfully")
        print(f"Apple data shape: {apple_train_data_df.shape}")

        # Repeat for other datasets (Amazon, Google, Microsoft)
        for company, data_dir, save_dir, save_name in [
            ("Amazon", self.config.amazon_data_dir, self.config.transformed_amazon_data_dir, "transformed_amazon.csv"),
            ("Google", self.config.google_data_dir, self.config.transformed_google_data_dir, "transformed_google.csv"),
            ("Microsoft", self.config.microsoft_data_dir, self.config.transformed_microsoft_data_dir, "transformed_microsoft.csv"),
        ]:
            data = pd.read_csv(data_dir)
            data = data.iloc[1:]
            data.rename(columns={"Price": "Date"}, inplace=True)
            data = data.drop(data.index[0])
            data = data.filter(['Close'])
            dataset = data.values
            training_data_len = int(np.ceil(len(dataset) * 0.95))
            scaled_data = scaler.fit_transform(dataset)
            train_data = scaled_data[:training_data_len]
            train_data_df = pd.DataFrame(train_data)
            os.makedirs(save_dir, exist_ok=True)
            train_data_df.to_csv(os.path.join(save_dir, save_name), index=False)
            logger.info(f"Transformed {company} Data saved successfully")
            print(f"{company} data shape: {train_data_df.shape}")
