import os
import yfinance as yf
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
from datetime import datetime
from smPredictor import logger
from smPredictor.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def get_data(self):
        # Ensure the local data directory and all subdirectories exist
        os.makedirs(self.config.local_data_dir, exist_ok=True)

        # Parse tickers, start_date, and end_date from the config
        tickers = self.config.tickers
        
        # Validate tickers to ensure it is a list
        if not isinstance(tickers, list):
            raise ValueError("Tickers should be a list.")

        start_date = self.config.start_date
        
        # Convert 'now' to current date if 'end_date' is 'now'
        end_date = (
            datetime.now().strftime("%Y-%m-%d") 
            if self.config.end_date.lower() == "now" 
            else self.config.end_date
        )

        # Download stock data for each ticker
        for ticker in tickers:
            try:
                print(f"Downloading data for {ticker}...")
                # Fetch the stock data using yfinance
                data = yf.download(ticker, start=start_date, end=end_date)

                # Check if data is empty
                if data.empty:
                    print(f"No data found for {ticker}. Skipping.")
                    continue

                # Ensure the directory for each stock exists
                file_path = os.path.join(self.config.local_data_dir, f"{ticker}.csv")
                
                # Save the data to a CSV file
                data.to_csv(file_path)
                print(f"Saved data for {ticker} to {file_path}")

            except Exception as e:
                print(f"Failed to download data for {ticker}: {e}")
