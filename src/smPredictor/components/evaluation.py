import tensorflow as tf
from pathlib import Path
import pandas as pd
from smPredictor import logger
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import os
from smPredictor import logger
from smPredictor.entity.config_entity import EvaluationConfig
from smPredictor.utils.common import save_json

class Evaluation:
    def __init__(self, config):
        self.config = config
    
    def create_rmse_directory(self):
        # Create the model_rmse directory if it doesn't exist
        rmse_dir = Path("artifacts/model_rmse")
        rmse_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self):
        #Load Models
        apple_model = tf.keras.models.load_model("artifacts/model_trainer/apple_model.keras")
        amazon_model = tf.keras.models.load_model("artifacts/model_trainer/amazon_model.keras")
        google_model = tf.keras.models.load_model("artifacts/model_trainer/google_model.keras")
        microsoft_model = tf.keras.models.load_model("artifacts/model_trainer/microsoft_model.keras")

        # Evaluate Apple Model on Test Data
        apple_scaler = MinMaxScaler(feature_range=(0, 1))
        apple_raw_data = pd.read_csv(self.config.apple_raw_data_dir)
        apple_raw_data = apple_raw_data.iloc[1:]
        apple_raw_data.rename(columns={"Price": "Date"}, inplace=True)
        apple_raw_data = apple_raw_data.filter(['Close'])
        apple_raw_data['Close'] = pd.to_numeric(apple_raw_data['Close'], errors='coerce')
        apple_raw_data = apple_raw_data.dropna()
        apple_dataset = apple_raw_data.values
        apple_training_data_len = int(np.ceil(len(apple_dataset) * .95))
        apple_scaled_data = apple_scaler.fit_transform(apple_dataset)
        apple_test_data = apple_scaled_data[apple_training_data_len - 60:, :]
        apple_x_test = []
        apple_y_test = apple_dataset[apple_training_data_len:, :]
        for i in range(60, len(apple_test_data)):
            apple_x_test.append(apple_test_data[i - 60:i, 0])
        apple_x_test = np.array(apple_x_test)
        apple_x_test = np.reshape(apple_x_test, (apple_x_test.shape[0], apple_x_test.shape[1], 1))
        apple_predictions = apple_model.predict(apple_x_test)
        apple_predictions = apple_scaler.inverse_transform(apple_predictions)
        apple_rmse = np.sqrt(np.mean((apple_predictions - apple_y_test) ** 2))
        save_json(path=Path("artifacts/model_rmse/apple_rmse.json"), data={"RMSE": apple_rmse})
        
        # Evaluate Amazon Model on Test Data
        amazon_scaler = MinMaxScaler(feature_range=(0, 1))
        amazon_raw_data = pd.read_csv(self.config.amazon_raw_data_dir)
        amazon_raw_data = amazon_raw_data.iloc[1:]
        amazon_raw_data.rename(columns={"Price": "Date"}, inplace=True)
        amazon_raw_data = amazon_raw_data.filter(['Close'])
        amazon_raw_data['Close'] = pd.to_numeric(amazon_raw_data['Close'], errors='coerce')
        amazon_raw_data = amazon_raw_data.dropna()
        amazon_dataset = amazon_raw_data.values
        amazon_training_data_len = int(np.ceil(len(amazon_dataset) * .95))
        amazon_scaled_data = amazon_scaler.fit_transform(amazon_dataset)
        amazon_test_data = amazon_scaled_data[amazon_training_data_len - 60:, :]
        amazon_x_test = []
        amazon_y_test = amazon_dataset[amazon_training_data_len:, :]
        for i in range(60, len(amazon_test_data)):
            amazon_x_test.append(amazon_test_data[i - 60:i, 0])
        amazon_x_test = np.array(amazon_x_test)
        amazon_x_test = np.reshape(amazon_x_test, (amazon_x_test.shape[0], amazon_x_test.shape[1], 1))
        amazon_predictions = amazon_model.predict(amazon_x_test)
        amazon_predictions = amazon_scaler.inverse_transform(amazon_predictions)
        amazon_rmse = np.sqrt(np.mean((amazon_predictions - amazon_y_test) ** 2))
        save_json(path=Path("artifacts/model_rmse/amazon_rmse.json"), data={"RMSE": amazon_rmse})
        
        # Evaluate Google Model on Test Data
        google_scaler = MinMaxScaler(feature_range=(0, 1))
        google_raw_data = pd.read_csv(self.config.google_raw_data_dir)
        google_raw_data = google_raw_data.iloc[1:]
        google_raw_data.rename(columns={"Price": "Date"}, inplace=True)
        google_raw_data = google_raw_data.filter(['Close'])
        google_raw_data['Close'] = pd.to_numeric(google_raw_data['Close'], errors='coerce')
        google_raw_data = google_raw_data.dropna()
        google_dataset = google_raw_data.values
        google_training_data_len = int(np.ceil(len(google_dataset) * .95))
        google_scaled_data = google_scaler.fit_transform(google_dataset)
        google_test_data = google_scaled_data[google_training_data_len - 60:, :]
        google_x_test = []
        google_y_test = google_dataset[google_training_data_len:, :]
        for i in range(60, len(google_test_data)):
            google_x_test.append(google_test_data[i - 60:i, 0])
        google_x_test = np.array(google_x_test)
        google_x_test = np.reshape(google_x_test, (google_x_test.shape[0], google_x_test.shape[1], 1))
        google_predictions = google_model.predict(google_x_test)
        google_predictions = google_scaler.inverse_transform(google_predictions)
        google_rmse = np.sqrt(np.mean((google_predictions - google_y_test) ** 2))
        save_json(path=Path("artifacts/model_rmse/google_rmse.json"), data={"RMSE": google_rmse})
        
        # Evaluate Microsoft Model on Test Data
        microsoft_scaler = MinMaxScaler(feature_range=(0, 1))
        microsoft_raw_data = pd.read_csv(self.config.microsoft_raw_data_dir)
        microsoft_raw_data = microsoft_raw_data.iloc[1:]
        microsoft_raw_data.rename(columns={"Price": "Date"}, inplace=True)
        microsoft_raw_data = microsoft_raw_data.filter(['Close'])
        microsoft_raw_data['Close'] = pd.to_numeric(microsoft_raw_data['Close'], errors='coerce')
        microsoft_raw_data = microsoft_raw_data.dropna()
        microsoft_dataset = microsoft_raw_data.values
        microsoft_training_data_len = int(np.ceil(len(microsoft_dataset) * .95))
        microsoft_scaled_data = microsoft_scaler.fit_transform(microsoft_dataset)
        microsoft_test_data = microsoft_scaled_data[microsoft_training_data_len - 60:, :]
        microsoft_x_test = []
        microsoft_y_test = microsoft_dataset[microsoft_training_data_len:, :]
        for i in range(60, len(microsoft_test_data)):
            microsoft_x_test.append(microsoft_test_data[i - 60:i, 0])
        microsoft_x_test = np.array(microsoft_x_test)
        microsoft_x_test = np.reshape(microsoft_x_test, (microsoft_x_test.shape[0], microsoft_x_test.shape[1], 1))
        microsoft_predictions = microsoft_model.predict(microsoft_x_test)
        microsoft_predictions = microsoft_scaler.inverse_transform(microsoft_predictions)
        microsoft_rmse = np.sqrt(np.mean((microsoft_predictions - microsoft_y_test) ** 2))
        save_json(path=Path("artifacts/model_rmse/microsoft_rmse.json"), data={"RMSE": microsoft_rmse})





