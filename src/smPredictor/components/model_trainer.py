import numpy as np
import pandas as pd
from smPredictor import logger
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import save_model
from smPredictor.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Training of Apple Model
        apple_df = pd.read_csv(self.config.apple_transformed_data_dir)
        apple_df = apple_df.astype(float)  # Ensure numeric data
        apple_x_train = []
        apple_y_train = []
        for i in range(60, len(apple_df)):
            apple_x_train.append(apple_df.iloc[i-60:i, 0].values)
            apple_y_train.append(apple_df.iloc[i, 0])
            if i <= 61:
                print(apple_x_train)
                print(apple_y_train)
                print()
        apple_x_train, apple_y_train = np.array(apple_x_train), np.array(apple_y_train)
        apple_x_train = np.reshape(apple_x_train, (apple_x_train.shape[0], apple_x_train.shape[1], 1))
        apple_model = Sequential()
        apple_model.add(LSTM(self.config.layer1, return_sequences=True, input_shape=(apple_x_train.shape[1], 1)))
        apple_model.add(LSTM(self.config.layer2, return_sequences=False))
        apple_model.add(Dense(self.config.layer3))
        apple_model.add(Dense(self.config.layer4))
        apple_model.compile(optimizer=self.config.optimizer, loss=self.config.loss)
        apple_model.fit(apple_x_train, apple_y_train, batch_size=self.config.batch_size, epochs=self.config.epochs)
        apple_model.save(os.path.join(self.config.root_dir, self.config.apple_model_name))  # Save model

        # Training of Amazon Model
        amazon_df = pd.read_csv(self.config.amazon_transformed_data_dir)
        amazon_df = amazon_df.astype(float)  # Ensure numeric data
        amazon_x_train = []
        amazon_y_train = []
        for i in range(60, len(amazon_df)):
            amazon_x_train.append(amazon_df.iloc[i-60:i, 0].values)
            amazon_y_train.append(amazon_df.iloc[i, 0])
            if i <= 61:
                print(amazon_x_train)
                print(amazon_y_train)
                print()
        amazon_x_train, amazon_y_train = np.array(amazon_x_train), np.array(amazon_y_train)
        amazon_x_train = np.reshape(amazon_x_train, (amazon_x_train.shape[0], amazon_x_train.shape[1], 1))
        amazon_model = Sequential()
        amazon_model.add(LSTM(self.config.layer1, return_sequences=True, input_shape=(amazon_x_train.shape[1], 1)))
        amazon_model.add(LSTM(self.config.layer2, return_sequences=False))
        amazon_model.add(Dense(self.config.layer3))
        amazon_model.add(Dense(self.config.layer4))
        amazon_model.compile(optimizer=self.config.optimizer, loss=self.config.loss)
        amazon_model.fit(amazon_x_train, amazon_y_train, batch_size=self.config.batch_size, epochs=self.config.epochs)
        amazon_model.save(os.path.join(self.config.root_dir, self.config.amazon_model_name))  # Save model

        # Training of Google Model
        google_df = pd.read_csv(self.config.google_transformed_data_dir)
        google_df = google_df.astype(float)  # Ensure numeric data
        google_x_train = []
        google_y_train = []
        for i in range(60, len(google_df)):
            google_x_train.append(google_df.iloc[i-60:i, 0].values)
            google_y_train.append(google_df.iloc[i, 0])
            if i <= 61:
                print(google_x_train)
                print(google_y_train)
                print()
        google_x_train, google_y_train = np.array(google_x_train), np.array(google_y_train)
        google_x_train = np.reshape(google_x_train, (google_x_train.shape[0], google_x_train.shape[1], 1))
        google_model = Sequential()
        google_model.add(LSTM(self.config.layer1, return_sequences=True, input_shape=(google_x_train.shape[1], 1)))
        google_model.add(LSTM(self.config.layer2, return_sequences=False))
        google_model.add(Dense(self.config.layer3))
        google_model.add(Dense(self.config.layer4))
        google_model.compile(optimizer=self.config.optimizer, loss=self.config.loss)
        google_model.fit(google_x_train, google_y_train, batch_size=self.config.batch_size, epochs=self.config.epochs)
        google_model.save(os.path.join(self.config.root_dir, self.config.google_model_name))  # Save model

        # Training of Microsoft Model
        microsoft_df = pd.read_csv(self.config.microsoft_transformed_data_dir)
        microsoft_df = microsoft_df.astype(float)  # Ensure numeric data
        microsoft_x_train = []
        microsoft_y_train = []
        for i in range(60, len(microsoft_df)):
            microsoft_x_train.append(microsoft_df.iloc[i-60:i, 0].values)
            microsoft_y_train.append(microsoft_df.iloc[i, 0])
            if i <= 61:
                print(microsoft_x_train)
                print(microsoft_y_train)
                print()
        microsoft_x_train, microsoft_y_train = np.array(microsoft_x_train), np.array(microsoft_y_train)
        microsoft_x_train = np.reshape(microsoft_x_train, (microsoft_x_train.shape[0], microsoft_x_train.shape[1], 1))
        microsoft_model = Sequential()
        microsoft_model.add(LSTM(self.config.layer1, return_sequences=True, input_shape=(microsoft_x_train.shape[1], 1)))
        microsoft_model.add(LSTM(self.config.layer2, return_sequences=False))
        microsoft_model.add(Dense(self.config.layer3))
        microsoft_model.add(Dense(self.config.layer4))
        microsoft_model.compile(optimizer=self.config.optimizer, loss=self.config.loss)
        microsoft_model.fit(microsoft_x_train, microsoft_y_train, batch_size=self.config.batch_size, epochs=self.config.epochs)
        microsoft_model.save(os.path.join(self.config.root_dir, self.config.microsoft_model_name))  # Save model
