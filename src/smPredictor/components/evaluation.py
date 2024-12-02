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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from smPredictor.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def create_rmse_directory(self):
        # Create the model_rmse directory if it doesn't exist
        rmse_dir = Path("artifacts/model_metrics")
        rmse_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self):
        # Load Models
        apple_model = tf.keras.models.load_model("artifacts/model_trainer/apple_model.keras")
        amazon_model = tf.keras.models.load_model("artifacts/model_trainer/amazon_model.keras")
        google_model = tf.keras.models.load_model("artifacts/model_trainer/google_model.keras")
        microsoft_model = tf.keras.models.load_model("artifacts/model_trainer/microsoft_model.keras")
        silk_model = tf.keras.models.load_model("artifacts/model_trainer/silk_model.keras")
        pace_model = tf.keras.models.load_model("artifacts/model_trainer/pace_model.keras")
        fauji_model = tf.keras.models.load_model("artifacts/model_trainer/fauji_model.keras")
        punjab_model = tf.keras.models.load_model("artifacts/model_trainer/punjab_model.keras")

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
        # Calculate evaluation metrics
        apple_mae = mean_absolute_error(apple_y_test, apple_predictions)
        apple_mse = mean_squared_error(apple_y_test, apple_predictions)
        apple_rmse = np.sqrt(apple_mse)
        apple_mape = np.mean(np.abs((apple_y_test - apple_predictions) / apple_y_test)) * 100
        apple_accuracy = 100 - apple_mape
        save_json(path=Path("artifacts/model_metrics/apple_metrics.json"), data={"MAE": apple_mae, "MSE": apple_mae, "RMSE": apple_rmse , "MAPE": apple_mape, "Accuracy": apple_accuracy})
        
        
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
        amazon_mae = mean_absolute_error(amazon_y_test, amazon_predictions)
        amazon_mse = mean_squared_error(amazon_y_test, amazon_predictions)
        amazon_rmse = np.sqrt(amazon_mse)
        amazon_mape = np.mean(np.abs((amazon_y_test - amazon_predictions) / amazon_y_test)) * 100
        amazon_accuracy = 100 - amazon_mape
        save_json(path=Path("artifacts/model_metrics/amazon_metrics.json"), data={"MAE": amazon_mae, "MSE": amazon_mae, "RMSE": amazon_rmse , "MAPE": amazon_mape, "Accuracy": amazon_accuracy})

        
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
        google_mae = mean_absolute_error(google_y_test, google_predictions)
        google_mse = mean_squared_error(google_y_test, google_predictions)
        google_rmse = np.sqrt(google_mse)
        google_mape = np.mean(np.abs((google_y_test - google_predictions) / google_y_test)) * 100
        google_accuracy = 100 - google_mape
        save_json(path=Path("artifacts/model_metrics/google_metrics.json"), data={"MAE": google_mae, "MSE": google_mae, "RMSE": google_rmse , "MAPE": google_mape, "Accuracy": google_accuracy})

        
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
        microsoft_mae = mean_absolute_error(microsoft_y_test, microsoft_predictions)
        microsoft_mse = mean_squared_error(microsoft_y_test, microsoft_predictions)
        microsoft_rmse = np.sqrt(microsoft_mse)
        microsoft_mape = np.mean(np.abs((microsoft_y_test - microsoft_predictions) / microsoft_y_test)) * 100
        microsoft_accuracy = 100 - microsoft_mape
        save_json(path=Path("artifacts/model_metrics/microsoft_metrics.json"), data={"MAE": microsoft_mae, "MSE": microsoft_mae, "RMSE": microsoft_rmse , "MAPE": microsoft_mape, "Accuracy": microsoft_accuracy})

        
        # Evaluate Silk Model on Test Data
        silk_scaler = MinMaxScaler(feature_range=(0, 1))
        silk_raw_data = pd.read_csv(self.config.silk_raw_data_dir)
        silk_raw_data = silk_raw_data.filter(['Close'])
        silk_raw_data['Close'] = pd.to_numeric(silk_raw_data['Close'], errors='coerce')
        silk_raw_data = silk_raw_data.dropna()
        silk_dataset = silk_raw_data.values
        silk_training_data_len = int(np.ceil(len(silk_dataset) * .95))
        silk_scaled_data = silk_scaler.fit_transform(silk_dataset)
        silk_test_data = silk_scaled_data[silk_training_data_len - 60:, :]
        silk_x_test = []
        silk_y_test = silk_dataset[silk_training_data_len:, :]
        for i in range(60, len(silk_test_data)):
            silk_x_test.append(silk_test_data[i - 60:i, 0])
        silk_x_test = np.array(silk_x_test)
        silk_x_test = np.reshape(silk_x_test, (silk_x_test.shape[0], silk_x_test.shape[1], 1))
        silk_predictions = silk_model.predict(silk_x_test)
        silk_predictions = silk_scaler.inverse_transform(silk_predictions)
        silk_mae = mean_absolute_error(silk_y_test, silk_predictions)
        silk_mse = mean_squared_error(silk_y_test, silk_predictions)
        silk_rmse = np.sqrt(silk_mse)
        silk_mape = np.mean(np.abs((silk_y_test - silk_predictions) / silk_y_test)) * 100
        silk_accuracy = 100 - silk_mape
        save_json(path=Path("artifacts/model_metrics/silk_metrics.json"), data={"MAE": silk_mae, "MSE": silk_mae, "RMSE": silk_rmse , "MAPE": silk_mape, "Accuracy": silk_accuracy})

        
        # Evaluate Pace Model on Test Data
        pace_scaler = MinMaxScaler(feature_range=(0, 1))
        pace_raw_data = pd.read_csv(self.config.pace_raw_data_dir)
        pace_raw_data = pace_raw_data.filter(['Close'])
        pace_raw_data['Close'] = pd.to_numeric(pace_raw_data['Close'], errors='coerce')
        pace_raw_data = pace_raw_data.dropna()
        pace_dataset = pace_raw_data.values
        pace_training_data_len = int(np.ceil(len(pace_dataset) * .95))
        pace_scaled_data = pace_scaler.fit_transform(pace_dataset)
        pace_test_data = pace_scaled_data[pace_training_data_len - 60:, :]
        pace_x_test = []
        pace_y_test = pace_dataset[pace_training_data_len:, :]
        for i in range(60, len(pace_test_data)):
            pace_x_test.append(pace_test_data[i - 60:i, 0])
        pace_x_test = np.array(pace_x_test)
        pace_x_test = np.reshape(pace_x_test, (pace_x_test.shape[0], pace_x_test.shape[1], 1))
        pace_predictions = pace_model.predict(pace_x_test)
        pace_predictions = pace_scaler.inverse_transform(pace_predictions)
        pace_mae = mean_absolute_error(pace_y_test, pace_predictions)
        pace_mse = mean_squared_error(pace_y_test, pace_predictions)
        pace_rmse = np.sqrt(pace_mse)
        pace_mape = np.mean(np.abs((pace_y_test - pace_predictions) / pace_y_test)) * 100
        pace_accuracy = 100 - pace_mape
        save_json(path=Path("artifacts/model_metrics/pace_metrics.json"), data={"MAE": pace_mae, "MSE": pace_mae, "RMSE": pace_rmse , "MAPE": pace_mape, "Accuracy": pace_accuracy})

        
        # Evaluate Fauji Model on Test Data
        fauji_scaler = MinMaxScaler(feature_range=(0, 1))
        fauji_raw_data = pd.read_csv(self.config.fauji_raw_data_dir)
        fauji_raw_data = fauji_raw_data.filter(['Close'])
        fauji_raw_data['Close'] = pd.to_numeric(fauji_raw_data['Close'], errors='coerce')
        fauji_raw_data = fauji_raw_data.dropna()
        fauji_dataset = fauji_raw_data.values
        fauji_training_data_len = int(np.ceil(len(fauji_dataset) * .95))
        fauji_scaled_data = fauji_scaler.fit_transform(fauji_dataset)
        fauji_test_data = fauji_scaled_data[fauji_training_data_len - 60:, :]
        fauji_x_test = []
        fauji_y_test = fauji_dataset[fauji_training_data_len:, :]
        for i in range(60, len(fauji_test_data)):
            fauji_x_test.append(fauji_test_data[i - 60:i, 0])
        fauji_x_test = np.array(fauji_x_test)
        fauji_x_test = np.reshape(fauji_x_test, (fauji_x_test.shape[0], fauji_x_test.shape[1], 1))
        fauji_predictions = fauji_model.predict(fauji_x_test)
        fauji_predictions = fauji_scaler.inverse_transform(fauji_predictions)
        fauji_mae = mean_absolute_error(fauji_y_test, fauji_predictions)
        fauji_mse = mean_squared_error(fauji_y_test, fauji_predictions)
        fauji_rmse = np.sqrt(fauji_mse)
        fauji_mape = np.mean(np.abs((fauji_y_test - fauji_predictions) / fauji_y_test)) * 100
        fauji_accuracy = 100 - fauji_mape
        save_json(path=Path("artifacts/model_metrics/fauji_metrics.json"), data={"MAE": fauji_mae, "MSE": fauji_mae, "RMSE": fauji_rmse , "MAPE": fauji_mape, "Accuracy": fauji_accuracy})

        
        # Evaluate Punjab Model on Test Data
        punjab_scaler = MinMaxScaler(feature_range=(0, 1))
        punjab_raw_data = pd.read_csv(self.config.punjab_raw_data_dir)
        punjab_raw_data = punjab_raw_data.filter(['Close'])
        punjab_raw_data['Close'] = pd.to_numeric(punjab_raw_data['Close'], errors='coerce')
        punjab_raw_data = punjab_raw_data.dropna()
        punjab_dataset = punjab_raw_data.values
        punjab_training_data_len = int(np.ceil(len(punjab_dataset) * .95))
        punjab_scaled_data = punjab_scaler.fit_transform(punjab_dataset)
        punjab_test_data = punjab_scaled_data[punjab_training_data_len - 60:, :]
        punjab_x_test = []
        punjab_y_test = punjab_dataset[punjab_training_data_len:, :]
        for i in range(60, len(punjab_test_data)):
            punjab_x_test.append(punjab_test_data[i - 60:i, 0])
        punjab_x_test = np.array(punjab_x_test)
        punjab_x_test = np.reshape(punjab_x_test, (punjab_x_test.shape[0], punjab_x_test.shape[1], 1))
        punjab_predictions = punjab_model.predict(punjab_x_test)
        punjab_predictions = punjab_scaler.inverse_transform(punjab_predictions)
        punjab_mae = mean_absolute_error(punjab_y_test, punjab_predictions)
        punjab_mse = mean_squared_error(punjab_y_test, punjab_predictions)
        punjab_rmse = np.sqrt(punjab_mse)
        punjab_mape = np.mean(np.abs((punjab_y_test - punjab_predictions) / punjab_y_test)) * 100
        punjab_accuracy = 100 - punjab_mape
        save_json(path=Path("artifacts/model_metrics/punjab_metrics.json"), data={"MAE": punjab_mae, "MSE": punjab_mae, "RMSE": punjab_rmse , "MAPE": punjab_mape, "Accuracy": punjab_accuracy})






