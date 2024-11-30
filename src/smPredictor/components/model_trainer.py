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
        
        # Training of Silk Bank Model
        silk_df = pd.read_csv(self.config.silk_transformed_data_dir)
        silk_df = silk_df.astype(float)  # Ensure numeric data
        silk_x_train = []
        silk_y_train = []
        for i in range(60, len(silk_df)):
            silk_x_train.append(silk_df.iloc[i-60:i, 0].values)
            silk_y_train.append(silk_df.iloc[i, 0])
            if i <= 61:
                print(silk_x_train)
                print(silk_y_train)
                print()
        silk_x_train, silk_y_train = np.array(silk_x_train), np.array(silk_y_train)
        silk_x_train = np.reshape(silk_x_train, (silk_x_train.shape[0], silk_x_train.shape[1], 1))
        silk_model = Sequential()
        silk_model.add(LSTM(self.config.layer1, return_sequences=True, input_shape=(silk_x_train.shape[1], 1)))
        silk_model.add(LSTM(self.config.layer2, return_sequences=False))
        silk_model.add(Dense(self.config.layer3))
        silk_model.add(Dense(self.config.layer4))
        silk_model.compile(optimizer=self.config.optimizer, loss=self.config.loss)
        silk_model.fit(silk_x_train, silk_y_train, batch_size=self.config.batch_size, epochs=self.config.epochs)
        silk_model.save(os.path.join(self.config.root_dir, self.config.silk_model_name))  # Save model
        
        # Training of Pace Pakistan Model
        pace_df = pd.read_csv(self.config.pace_transformed_data_dir)
        pace_df = pace_df.astype(float)  # Ensure numeric data
        pace_x_train = []
        pace_y_train = []
        for i in range(60, len(pace_df)):
            pace_x_train.append(pace_df.iloc[i-60:i, 0].values)
            pace_y_train.append(pace_df.iloc[i, 0])
            if i <= 61:
                print(pace_x_train)
                print(pace_y_train)
                print()
        pace_x_train, pace_y_train = np.array(pace_x_train), np.array(pace_y_train)
        pace_x_train = np.reshape(pace_x_train, (pace_x_train.shape[0], pace_x_train.shape[1], 1))
        pace_model = Sequential()
        pace_model.add(LSTM(self.config.layer1, return_sequences=True, input_shape=(pace_x_train.shape[1], 1)))
        pace_model.add(LSTM(self.config.layer2, return_sequences=False))
        pace_model.add(Dense(self.config.layer3))
        pace_model.add(Dense(self.config.layer4))
        pace_model.compile(optimizer=self.config.optimizer, loss=self.config.loss)
        pace_model.fit(pace_x_train, pace_y_train, batch_size=self.config.batch_size, epochs=self.config.epochs)
        pace_model.save(os.path.join(self.config.root_dir, self.config.pace_model_name))  # Save model
        
        # Training of Fauji Foods Model
        fauji_df = pd.read_csv(self.config.fauji_transformed_data_dir)
        fauji_df = fauji_df.astype(float)  # Ensure numeric data
        fauji_x_train = []
        fauji_y_train = []
        for i in range(60, len(fauji_df)):
            fauji_x_train.append(fauji_df.iloc[i-60:i, 0].values)
            fauji_y_train.append(fauji_df.iloc[i, 0])
            if i <= 61:
                print(fauji_x_train)
                print(fauji_y_train)
                print()
        fauji_x_train, fauji_y_train = np.array(fauji_x_train), np.array(fauji_y_train)
        fauji_x_train = np.reshape(fauji_x_train, (fauji_x_train.shape[0], fauji_x_train.shape[1], 1))
        fauji_model = Sequential()
        fauji_model.add(LSTM(self.config.layer1, return_sequences=True, input_shape=(fauji_x_train.shape[1], 1)))
        fauji_model.add(LSTM(self.config.layer2, return_sequences=False))
        fauji_model.add(Dense(self.config.layer3))
        fauji_model.add(Dense(self.config.layer4))
        fauji_model.compile(optimizer=self.config.optimizer, loss=self.config.loss)
        fauji_model.fit(fauji_x_train, fauji_y_train, batch_size=self.config.batch_size, epochs=self.config.epochs)
        fauji_model.save(os.path.join(self.config.root_dir, self.config.fauji_model_name))  # Save model
        
        # Training of Bank of Punjab Model
        punjab_df = pd.read_csv(self.config.punjab_transformed_data_dir)
        punjab_df = punjab_df.astype(float)  # Ensure numeric data
        punjab_x_train = []
        punjab_y_train = []
        for i in range(60, len(punjab_df)):
            punjab_x_train.append(punjab_df.iloc[i-60:i, 0].values)
            punjab_y_train.append(punjab_df.iloc[i, 0])
            if i <= 61:
                print(punjab_x_train)
                print(punjab_y_train)
                print()
        punjab_x_train, punjab_y_train = np.array(punjab_x_train), np.array(punjab_y_train)
        punjab_x_train = np.reshape(punjab_x_train, (punjab_x_train.shape[0], punjab_x_train.shape[1], 1))
        punjab_model = Sequential()
        punjab_model.add(LSTM(self.config.layer1, return_sequences=True, input_shape=(punjab_x_train.shape[1], 1)))
        punjab_model.add(LSTM(self.config.layer2, return_sequences=False))
        punjab_model.add(Dense(self.config.layer3))
        punjab_model.add(Dense(self.config.layer4))
        punjab_model.compile(optimizer=self.config.optimizer, loss=self.config.loss)
        punjab_model.fit(punjab_x_train, punjab_y_train, batch_size=self.config.batch_size, epochs=self.config.epochs)
        punjab_model.save(os.path.join(self.config.root_dir, self.config.punjab_model_name))  # Save model





