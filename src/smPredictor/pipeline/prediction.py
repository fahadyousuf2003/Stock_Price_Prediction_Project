import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from smPredictor.entity.config_entity import EvaluationConfig
from datetime import timedelta
import tensorflow as tf



class PredictionPipeline:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def predict(self,days,category,stock):
        # Load Models
        apple_model = tf.keras.models.load_model("artifacts/model_trainer/apple_model.keras")
        amazon_model = tf.keras.models.load_model("artifacts/model_trainer/amazon_model.keras")
        google_model = tf.keras.models.load_model("artifacts/model_trainer/google_model.keras")
        microsoft_model = tf.keras.models.load_model("artifacts/model_trainer/microsoft_model.keras")
        silk_model = tf.keras.models.load_model("artifacts/model_trainer/silk_model.keras")
        pace_model = tf.keras.models.load_model("artifacts/model_trainer/pace_model.keras")
        fauji_model = tf.keras.models.load_model("artifacts/model_trainer/fauji_model.keras")
        punjab_model = tf.keras.models.load_model("artifacts/model_trainer/punjab_model.keras")
        
        if category == "international":
            if stock == "apple":
                # Apple Future Closing Price Predictions
                apple_raw_data = pd.read_csv(self.config.apple_raw_data_dir)
                apple_raw_data = apple_raw_data.iloc[1:]
                apple_raw_data.rename(columns={"Price": "Date"}, inplace=True)
                apple_raw_data = apple_raw_data.drop(apple_raw_data.index[0])
                apple_data = apple_raw_data.filter(['Close'])
                apple_scaler = MinMaxScaler(feature_range=(0, 1))
                apple_scaler.fit(apple_data)
                apple_scaled_data = apple_scaler.transform(apple_data)
                apple_train_size = int(len(apple_scaled_data) * 0.8)
                apple_test_data = apple_scaled_data[apple_train_size:]
                apple_X_test = []
                apple_y_test = []
                for i in range(60, len(apple_test_data)):
                    apple_X_test.append(apple_test_data[i-60:i])
                    apple_y_test.append(apple_test_data[i])
                apple_X_test = np.array(apple_X_test)
                apple_y_test = np.array(apple_y_test)
                apple_X_test = apple_X_test.reshape(apple_X_test.shape[0], apple_X_test.shape[1], 1)
                apple_num_days = days
                apple_last_60_days = apple_scaled_data[-60:]
                apple_X_future = [apple_last_60_days]
                apple_X_future = np.array(apple_X_future).reshape(1, 60, 1)
                apple_future_predictions = []
                for _ in range(apple_num_days):
                    apple_predicted_price = apple_model.predict(apple_X_future)
                    apple_future_predictions.append(apple_predicted_price[0][0])
                    apple_next_input = np.append(apple_X_future[0, 1:, 0], apple_predicted_price[0][0])
                    apple_X_future = np.array([apple_next_input]).reshape(1, 60, 1)
                apple_future_predictions = apple_scaler.inverse_transform(np.array(apple_future_predictions).reshape(-1, 1))
                apple_last_date = pd.to_datetime(apple_raw_data['Date'].iloc[-1])
                apple_future_dates = [apple_last_date + timedelta(days=i+1) for i in range(apple_num_days)]
                apple_future_dates_formatted = [date.strftime('%m/%d/%Y') for date in apple_future_dates]
                apple_future_df = pd.DataFrame({'Date': apple_future_dates_formatted, 'Predictions': apple_future_predictions.flatten()})
                return apple_future_df
            
        
            elif stock == "amazon":
                # Amazon Future Closing Price Predictions
                amazon_raw_data = pd.read_csv(self.config.amazon_raw_data_dir)
                amazon_raw_data = amazon_raw_data.iloc[1:]
                amazon_raw_data.rename(columns={"Price": "Date"}, inplace=True)
                amazon_raw_data = amazon_raw_data.drop(amazon_raw_data.index[0])
                amazon_data = amazon_raw_data.filter(['Close'])
                amazon_scaler = MinMaxScaler(feature_range=(0, 1))
                amazon_scaler.fit(amazon_data)
                amazon_scaled_data = amazon_scaler.transform(amazon_data)
                amazon_train_size = int(len(amazon_scaled_data) * 0.8)
                amazon_test_data = amazon_scaled_data[amazon_train_size:]
                amazon_X_test = []
                amazon_y_test = []
                for i in range(60, len(amazon_test_data)):
                    amazon_X_test.append(amazon_test_data[i-60:i])
                    amazon_y_test.append(amazon_test_data[i])
                amazon_X_test = np.array(amazon_X_test)
                amazon_y_test = np.array(amazon_y_test)
                amazon_X_test = amazon_X_test.reshape(amazon_X_test.shape[0], amazon_X_test.shape[1], 1)
                amazon_num_days = days
                amazon_last_60_days = amazon_scaled_data[-60:]
                amazon_X_future = [amazon_last_60_days]
                amazon_X_future = np.array(amazon_X_future).reshape(1, 60, 1)
                amazon_future_predictions = []
                for _ in range(amazon_num_days):
                    amazon_predicted_price = amazon_model.predict(amazon_X_future)
                    amazon_future_predictions.append(amazon_predicted_price[0][0])
                    amazon_next_input = np.append(amazon_X_future[0, 1:, 0], amazon_predicted_price[0][0])
                    amazon_X_future = np.array([amazon_next_input]).reshape(1, 60, 1)
                amazon_future_predictions = amazon_scaler.inverse_transform(np.array(amazon_future_predictions).reshape(-1, 1))
                amazon_last_date = pd.to_datetime(amazon_raw_data['Date'].iloc[-1])
                amazon_future_dates = [amazon_last_date + timedelta(days=i+1) for i in range(amazon_num_days)]
                amazon_future_dates_formatted = [date.strftime('%m/%d/%Y') for date in amazon_future_dates]
                amazon_future_df = pd.DataFrame({'Date': amazon_future_dates_formatted, 'Predictions': amazon_future_predictions.flatten()})
                return amazon_future_df
            
            elif stock == "google":
                # Google Future Closing Price Predictions
                google_raw_data = pd.read_csv(self.config.google_raw_data_dir)
                google_raw_data = google_raw_data.iloc[1:]
                google_raw_data.rename(columns={"Price": "Date"}, inplace=True)
                google_raw_data = google_raw_data.drop(google_raw_data.index[0])
                google_data = google_raw_data.filter(['Close'])
                google_scaler = MinMaxScaler(feature_range=(0, 1))
                google_scaler.fit(google_data)
                google_scaled_data = google_scaler.transform(google_data)
                google_train_size = int(len(google_scaled_data) * 0.8)
                google_test_data = google_scaled_data[google_train_size:]
                google_X_test = []
                google_y_test = []
                for i in range(60, len(google_test_data)):
                    google_X_test.append(google_test_data[i-60:i])
                    google_y_test.append(google_test_data[i])
                google_X_test = np.array(google_X_test)
                google_y_test = np.array(google_y_test)
                google_X_test = google_X_test.reshape(google_X_test.shape[0], google_X_test.shape[1], 1)
                google_num_days = days
                google_last_60_days = google_scaled_data[-60:]
                google_X_future = [google_last_60_days]
                google_X_future = np.array(google_X_future).reshape(1, 60, 1)
                google_future_predictions = []
                for _ in range(google_num_days):
                    google_predicted_price = google_model.predict(google_X_future)
                    google_future_predictions.append(google_predicted_price[0][0])
                    google_next_input = np.append(google_X_future[0, 1:, 0], google_predicted_price[0][0])
                    google_X_future = np.array([google_next_input]).reshape(1, 60, 1)
                google_future_predictions = google_scaler.inverse_transform(np.array(google_future_predictions).reshape(-1, 1))
                google_last_date = pd.to_datetime(google_raw_data['Date'].iloc[-1])
                google_future_dates = [google_last_date + timedelta(days=i+1) for i in range(google_num_days)]
                google_future_dates_formatted = [date.strftime('%m/%d/%Y') for date in google_future_dates]
                google_future_df = pd.DataFrame({'Date': google_future_dates_formatted, 'Predictions': google_future_predictions.flatten()})
                return google_future_df
            
            elif stock == "microsoft":
                # Microsoft Future Closing Price Predictions
                microsoft_raw_data = pd.read_csv(self.config.microsoft_raw_data_dir)
                microsoft_raw_data = microsoft_raw_data.iloc[1:]
                microsoft_raw_data.rename(columns={"Price": "Date"}, inplace=True)
                microsoft_raw_data = microsoft_raw_data.drop(microsoft_raw_data.index[0])
                microsoft_data = microsoft_raw_data.filter(['Close'])
                microsoft_scaler = MinMaxScaler(feature_range=(0, 1))
                microsoft_scaler.fit(microsoft_data)
                microsoft_scaled_data = microsoft_scaler.transform(microsoft_data)
                microsoft_train_size = int(len(microsoft_scaled_data) * 0.8)
                microsoft_test_data = microsoft_scaled_data[microsoft_train_size:]
                microsoft_X_test = []
                microsoft_y_test = []
                for i in range(60, len(microsoft_test_data)):
                    microsoft_X_test.append(microsoft_test_data[i-60:i])
                    microsoft_y_test.append(microsoft_test_data[i])
                microsoft_X_test = np.array(microsoft_X_test)
                microsoft_y_test = np.array(microsoft_y_test)
                microsoft_X_test = microsoft_X_test.reshape(microsoft_X_test.shape[0], microsoft_X_test.shape[1], 1)
                microsoft_num_days = days
                microsoft_last_60_days = microsoft_scaled_data[-60:]
                microsoft_X_future = [microsoft_last_60_days]
                microsoft_X_future = np.array(microsoft_X_future).reshape(1, 60, 1)
                microsoft_future_predictions = []
                for _ in range(microsoft_num_days):
                    microsoft_predicted_price = microsoft_model.predict(microsoft_X_future)
                    microsoft_future_predictions.append(microsoft_predicted_price[0][0])
                    microsoft_next_input = np.append(microsoft_X_future[0, 1:, 0], microsoft_predicted_price[0][0])
                    microsoft_X_future = np.array([microsoft_next_input]).reshape(1, 60, 1)
                microsoft_future_predictions = microsoft_scaler.inverse_transform(np.array(microsoft_future_predictions).reshape(-1, 1))
                microsoft_last_date = pd.to_datetime(microsoft_raw_data['Date'].iloc[-1])
                microsoft_future_dates = [microsoft_last_date + timedelta(days=i+1) for i in range(microsoft_num_days)]
                microsoft_future_dates_formatted = [date.strftime('%m/%d/%Y') for date in microsoft_future_dates]
                microsoft_future_df = pd.DataFrame({'Date': microsoft_future_dates_formatted, 'Predictions': microsoft_future_predictions.flatten()})
                return microsoft_future_df
            
        elif category == "pakistani":
            if stock == "silk":
                # Silk Bank Future Closing Price Predictions
                silk_raw_data = pd.read_csv(self.config.silk_raw_data_dir)
                silk_data = silk_raw_data.filter(['Close'])
                silk_scaler = MinMaxScaler(feature_range=(0, 1))
                silk_scaler.fit(silk_data)
                silk_scaled_data = silk_scaler.transform(silk_data)
                silk_train_size = int(len(silk_scaled_data) * 0.8)
                silk_test_data = silk_scaled_data[silk_train_size:]
                silk_X_test = []
                silk_y_test = []
                for i in range(60, len(silk_test_data)):
                    silk_X_test.append(silk_test_data[i-60:i])
                    silk_y_test.append(silk_test_data[i])
                silk_X_test = np.array(silk_X_test)
                silk_y_test = np.array(silk_y_test)
                silk_X_test = silk_X_test.reshape(silk_X_test.shape[0], silk_X_test.shape[1], 1)
                silk_num_days = days
                silk_last_60_days = silk_scaled_data[-60:]
                silk_X_future = [silk_last_60_days]
                silk_X_future = np.array(silk_X_future).reshape(1, 60, 1)
                silk_future_predictions = []
                for _ in range(silk_num_days):
                    silk_predicted_price = silk_model.predict(silk_X_future)
                    silk_future_predictions.append(silk_predicted_price[0][0])
                    silk_next_input = np.append(silk_X_future[0, 1:, 0], silk_predicted_price[0][0])
                    silk_X_future = np.array([silk_next_input]).reshape(1, 60, 1)
                silk_future_predictions = silk_scaler.inverse_transform(np.array(silk_future_predictions).reshape(-1, 1))
                silk_last_date = pd.to_datetime(silk_raw_data['Date'].iloc[-1])
                silk_future_dates = [silk_last_date + timedelta(days=i+1) for i in range(silk_num_days)]
                silk_future_dates_formatted = [date.strftime('%m/%d/%Y') for date in silk_future_dates]
                silk_future_df = pd.DataFrame({'Date': silk_future_dates_formatted, 'Predictions': silk_future_predictions.flatten()})
                return silk_future_df
            
            elif stock== "pace":
                # Pace Pakistan Future Closing Price Predictions
                pace_raw_data = pd.read_csv(self.config.pace_raw_data_dir)
                pace_data = pace_raw_data.filter(['Close'])
                pace_scaler = MinMaxScaler(feature_range=(0, 1))
                pace_scaler.fit(pace_data)
                pace_scaled_data = pace_scaler.transform(pace_data)
                pace_train_size = int(len(pace_scaled_data) * 0.8)
                pace_test_data = pace_scaled_data[pace_train_size:]
                pace_X_test = []
                pace_y_test = []
                for i in range(60, len(pace_test_data)):
                    pace_X_test.append(pace_test_data[i-60:i])
                    pace_y_test.append(pace_test_data[i])
                pace_X_test = np.array(pace_X_test)
                pace_y_test = np.array(pace_y_test)
                pace_X_test = pace_X_test.reshape(pace_X_test.shape[0], pace_X_test.shape[1], 1)
                pace_num_days = days
                pace_last_60_days = pace_scaled_data[-60:]
                pace_X_future = [pace_last_60_days]
                pace_X_future = np.array(pace_X_future).reshape(1, 60, 1)
                pace_future_predictions = []
                for _ in range(pace_num_days):
                    pace_predicted_price = pace_model.predict(pace_X_future)
                    pace_future_predictions.append(pace_predicted_price[0][0])
                    pace_next_input = np.append(pace_X_future[0, 1:, 0], pace_predicted_price[0][0])
                    pace_X_future = np.array([pace_next_input]).reshape(1, 60, 1)
                pace_future_predictions = pace_scaler.inverse_transform(np.array(pace_future_predictions).reshape(-1, 1))
                pace_last_date = pd.to_datetime(pace_raw_data['Date'].iloc[-1])
                pace_future_dates = [pace_last_date + timedelta(days=i+1) for i in range(pace_num_days)]
                pace_future_dates_formatted = [date.strftime('%m/%d/%Y') for date in pace_future_dates]
                pace_future_df = pd.DataFrame({'Date': pace_future_dates_formatted, 'Predictions': pace_future_predictions.flatten()})
                return pace_future_df
            
            elif stock=="fauji":
                # Fauji Foods Future Closing Price Predictions
                fauji_raw_data = pd.read_csv(self.config.fauji_raw_data_dir)
                fauji_data = fauji_raw_data.filter(['Close'])
                fauji_scaler = MinMaxScaler(feature_range=(0, 1))
                fauji_scaler.fit(fauji_data)
                fauji_scaled_data = fauji_scaler.transform(fauji_data)
                fauji_train_size = int(len(fauji_scaled_data) * 0.8)
                fauji_test_data = fauji_scaled_data[fauji_train_size:]
                fauji_X_test = []
                fauji_y_test = []
                for i in range(60, len(fauji_test_data)):
                    fauji_X_test.append(fauji_test_data[i-60:i])
                    fauji_y_test.append(fauji_test_data[i])
                fauji_X_test = np.array(fauji_X_test)
                fauji_y_test = np.array(fauji_y_test)
                fauji_X_test = fauji_X_test.reshape(fauji_X_test.shape[0], fauji_X_test.shape[1], 1)
                fauji_num_days = days
                fauji_last_60_days = fauji_scaled_data[-60:]
                fauji_X_future = [fauji_last_60_days]
                fauji_X_future = np.array(fauji_X_future).reshape(1, 60, 1)
                fauji_future_predictions = []
                for _ in range(fauji_num_days):
                    fauji_predicted_price = fauji_model.predict(fauji_X_future)
                    fauji_future_predictions.append(fauji_predicted_price[0][0])
                    fauji_next_input = np.append(fauji_X_future[0, 1:, 0], fauji_predicted_price[0][0])
                    fauji_X_future = np.array([fauji_next_input]).reshape(1, 60, 1)
                fauji_future_predictions = fauji_scaler.inverse_transform(np.array(fauji_future_predictions).reshape(-1, 1))
                fauji_last_date = pd.to_datetime(fauji_raw_data['Date'].iloc[-1])
                fauji_future_dates = [fauji_last_date + timedelta(days=i+1) for i in range(fauji_num_days)]
                fauji_future_dates_formatted = [date.strftime('%m/%d/%Y') for date in fauji_future_dates]
                fauji_future_df = pd.DataFrame({'Date': fauji_future_dates_formatted, 'Predictions': fauji_future_predictions.flatten()})
                return fauji_future_df
            
            elif stock=="punjab":
                # Bank of Punjab Future Closing Price Predictions
                punjab_raw_data = pd.read_csv(self.config.punjab_raw_data_dir)
                punjab_data = punjab_raw_data.filter(['Close'])
                punjab_scaler = MinMaxScaler(feature_range=(0, 1))
                punjab_scaler.fit(punjab_data)
                punjab_scaled_data = punjab_scaler.transform(punjab_data)
                punjab_train_size = int(len(punjab_scaled_data) * 0.8)
                punjab_test_data = punjab_scaled_data[punjab_train_size:]
                punjab_X_test = []
                punjab_y_test = []
                for i in range(60, len(punjab_test_data)):
                    punjab_X_test.append(punjab_test_data[i-60:i])
                    punjab_y_test.append(punjab_test_data[i])
                punjab_X_test = np.array(punjab_X_test)
                punjab_y_test = np.array(punjab_y_test)
                punjab_X_test = punjab_X_test.reshape(punjab_X_test.shape[0], punjab_X_test.shape[1], 1)
                punjab_num_days = days
                punjab_last_60_days = punjab_scaled_data[-60:]
                punjab_X_future = [punjab_last_60_days]
                punjab_X_future = np.array(punjab_X_future).reshape(1, 60, 1)
                punjab_future_predictions = []
                for _ in range(punjab_num_days):
                    punjab_predicted_price = punjab_model.predict(punjab_X_future)
                    punjab_future_predictions.append(punjab_predicted_price[0][0])
                    punjab_next_input = np.append(punjab_X_future[0, 1:, 0], punjab_predicted_price[0][0])
                    punjab_X_future = np.array([punjab_next_input]).reshape(1, 60, 1)
                punjab_future_predictions = punjab_scaler.inverse_transform(np.array(punjab_future_predictions).reshape(-1, 1))
                punjab_last_date = pd.to_datetime(punjab_raw_data['Date'].iloc[-1])
                punjab_future_dates = [punjab_last_date + timedelta(days=i+1) for i in range(punjab_num_days)]
                punjab_future_dates_formatted = [date.strftime('%m/%d/%Y') for date in punjab_future_dates]
                punjab_future_df = pd.DataFrame({'Date': punjab_future_dates_formatted, 'Predictions': punjab_future_predictions.flatten()})
                return punjab_future_df


# Example usage
if __name__ == "__main__":
    pipeline = PredictionPipeline()
    predictions = pipeline.predict(days=30, category="international", stock="apple")
    print(predictions)
