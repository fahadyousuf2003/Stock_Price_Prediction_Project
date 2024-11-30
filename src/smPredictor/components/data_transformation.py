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
        apple_data = apple_data.filter(['Close'])
        apple_dataset = apple_data.values
        apple_training_data_len = int(np.ceil(len(apple_dataset) * 0.95))
        scaler = MinMaxScaler(feature_range=(0, 1))
        apple_scaled_data = scaler.fit_transform(apple_dataset)
        apple_train_data = apple_scaled_data[:apple_training_data_len]
        apple_train_data_df = pd.DataFrame(apple_train_data)
        os.makedirs(self.config.transformed_apple_data_dir, exist_ok=True)
        apple_train_data_df.to_csv(
            os.path.join(self.config.transformed_apple_data_dir, "transformed_apple.csv"), 
            index=False
        )
        logger.info("Transformed Apple Data saved successfully")
        print(f"Apple data shape: {apple_train_data_df.shape}")
        
        # Transformation for Amazon Data
        amazon_data = pd.read_csv(self.config.amazon_data_dir)
        amazon_data = amazon_data.iloc[1:]
        amazon_data.rename(columns={"Price": "Date"}, inplace=True)
        amazon_data = amazon_data.drop(amazon_data.index[0])
        amazon_data = amazon_data.filter(['Close'])
        amazon_dataset = amazon_data.values
        amazon_training_data_len = int(np.ceil(len(amazon_dataset) * 0.95))
        scaler = MinMaxScaler(feature_range=(0, 1))
        amazon_scaled_data = scaler.fit_transform(amazon_dataset)
        amazon_train_data = amazon_scaled_data[:amazon_training_data_len]
        amazon_train_data_df = pd.DataFrame(amazon_train_data)
        os.makedirs(self.config.transformed_amazon_data_dir, exist_ok=True)
        amazon_train_data_df.to_csv(
            os.path.join(self.config.transformed_amazon_data_dir, "transformed_amazon.csv"), 
            index=False
        )
        logger.info("Transformed Amazon Data saved successfully")
        print(f"Amazon data shape: {amazon_train_data_df.shape}")
        
        # Transformation for Google Data
        google_data = pd.read_csv(self.config.google_data_dir)
        google_data = google_data.iloc[1:]
        google_data.rename(columns={"Price": "Date"}, inplace=True)
        google_data = google_data.drop(google_data.index[0])
        google_data = google_data.filter(['Close'])
        google_dataset = google_data.values
        google_training_data_len = int(np.ceil(len(google_dataset) * 0.95))
        scaler = MinMaxScaler(feature_range=(0, 1))
        google_scaled_data = scaler.fit_transform(google_dataset)
        google_train_data = google_scaled_data[:google_training_data_len]
        google_train_data_df = pd.DataFrame(google_train_data)
        os.makedirs(self.config.transformed_google_data_dir, exist_ok=True)
        google_train_data_df.to_csv(
            os.path.join(self.config.transformed_google_data_dir, "transformed_google.csv"), 
            index=False
        )
        logger.info("Transformed Google Data saved successfully")
        print(f"Google data shape: {google_train_data_df.shape}")
        
        # Transformation for Microsoft Data
        microsoft_data = pd.read_csv(self.config.microsoft_data_dir)
        microsoft_data = microsoft_data.iloc[1:]
        microsoft_data.rename(columns={"Price": "Date"}, inplace=True)
        microsoft_data = microsoft_data.drop(microsoft_data.index[0])
        microsoft_data = microsoft_data.filter(['Close'])
        microsoft_dataset = microsoft_data.values
        microsoft_training_data_len = int(np.ceil(len(microsoft_dataset) * 0.95))
        scaler = MinMaxScaler(feature_range=(0, 1))
        microsoft_scaled_data = scaler.fit_transform(microsoft_dataset)
        microsoft_train_data = microsoft_scaled_data[:microsoft_training_data_len]
        microsoft_train_data_df = pd.DataFrame(microsoft_train_data)
        os.makedirs(self.config.transformed_microsoft_data_dir, exist_ok=True)
        microsoft_train_data_df.to_csv(
            os.path.join(self.config.transformed_microsoft_data_dir, "transformed_microsoft.csv"), 
            index=False
        )
        logger.info("Transformed Microsoft Data saved successfully")
        print(f"Microsoft data shape: {microsoft_train_data_df.shape}")
        
        # Transformation for Silk Bank Data
        silk_data = pd.read_csv(self.config.silk_data_dir)
        silk_data = silk_data.filter(['Close'])
        silk_dataset = silk_data.values
        silk_training_data_len = int(np.ceil(len(silk_dataset) * 0.95))
        scaler = MinMaxScaler(feature_range=(0, 1))
        silk_scaled_data = scaler.fit_transform(silk_dataset)
        silk_train_data = silk_scaled_data[:silk_training_data_len]
        silk_train_data_df = pd.DataFrame(silk_train_data)
        os.makedirs(self.config.transformed_silk_data_dir, exist_ok=True)
        silk_train_data_df.to_csv(
            os.path.join(self.config.transformed_silk_data_dir, "transformed_silk.csv"), 
            index=False
        )
        logger.info("Transformed Silk Data saved successfully")
        print(f"Silk data shape: {silk_train_data_df.shape}")
        
        # Transformation for Pace Pakistan Data
        pace_data = pd.read_csv(self.config.pace_data_dir)
        pace_data = pace_data.filter(['Close'])
        pace_dataset = pace_data.values
        pace_training_data_len = int(np.ceil(len(pace_dataset) * 0.95))
        scaler = MinMaxScaler(feature_range=(0, 1))
        pace_scaled_data = scaler.fit_transform(pace_dataset)
        pace_train_data = pace_scaled_data[:pace_training_data_len]
        pace_train_data_df = pd.DataFrame(pace_train_data)
        os.makedirs(self.config.transformed_pace_data_dir, exist_ok=True)
        pace_train_data_df.to_csv(
            os.path.join(self.config.transformed_pace_data_dir, "transformed_pace.csv"), 
            index=False
        )
        logger.info("Transformed Pace Data saved successfully")
        print(f"Pace data shape: {pace_train_data_df.shape}")
        
        # Transformation for Fauji Foods Limited Data
        fauji_data = pd.read_csv(self.config.fauji_data_dir)
        fauji_data = fauji_data.filter(['Close'])
        fauji_dataset = fauji_data.values
        fauji_training_data_len = int(np.ceil(len(fauji_dataset) * 0.95))
        scaler = MinMaxScaler(feature_range=(0, 1))
        fauji_scaled_data = scaler.fit_transform(fauji_dataset)
        fauji_train_data = fauji_scaled_data[:fauji_training_data_len]
        fauji_train_data_df = pd.DataFrame(fauji_train_data)
        os.makedirs(self.config.transformed_fauji_data_dir, exist_ok=True)
        fauji_train_data_df.to_csv(
            os.path.join(self.config.transformed_fauji_data_dir, "transformed_fauji.csv"), 
            index=False
        )
        logger.info("Transformed Fauji Data saved successfully")
        print(f"Fauji data shape: {fauji_train_data_df.shape}")
        
        # Transformation for Bank of Punjab Data
        punjab_data = pd.read_csv(self.config.punjab_data_dir)
        punjab_data = punjab_data.filter(['Close'])
        punjab_dataset = punjab_data.values
        punjab_training_data_len = int(np.ceil(len(punjab_dataset) * 0.95))
        scaler = MinMaxScaler(feature_range=(0, 1))
        punjab_scaled_data = scaler.fit_transform(punjab_dataset)
        punjab_train_data = punjab_scaled_data[:punjab_training_data_len]
        punjab_train_data_df = pd.DataFrame(punjab_train_data)
        os.makedirs(self.config.transformed_punjab_data_dir, exist_ok=True)
        punjab_train_data_df.to_csv(
            os.path.join(self.config.transformed_punjab_data_dir, "transformed_punjab.csv"), 
            index=False
        )
        logger.info("Transformed Punjab Data saved successfully")
        print(f"Punjab data shape: {punjab_train_data_df.shape}")


        
        

       