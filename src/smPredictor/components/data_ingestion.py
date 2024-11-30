import os
import yfinance as yf
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
from datetime import datetime, date
from smPredictor import logger
from psx import stocks
from smPredictor.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def get_data(self):
        os.makedirs(self.config.internationalStocks_local_data_dir, exist_ok=True)
        os.makedirs(self.config.pakistanStocks_local_data_dir, exist_ok=True)

        itickers = self.config.itickers
        ptickers = self.config.ptickers
        
        # Validate tickers to ensure they are lists
        if not isinstance(itickers, list):
            raise ValueError("International Tickers should be a list.")
        if not isinstance(ptickers, list):
            raise ValueError("Pakistan Tickers should be a list.")

        istart_date = self.config.istart_date
        iend_date = (
            datetime.now().strftime("%Y-%m-%d") 
            if isinstance(self.config.iend_date, str) and self.config.iend_date.lower() == "now"
            else self.config.iend_date
        )

        # Download stock data for International tickers
        for iticker in itickers:
            try:
                print(f"Downloading data for {iticker}...")
                idata = yf.download(iticker, start=istart_date, end=iend_date)
                if idata.empty:
                    print(f"No data found for {iticker}. Skipping.")
                    continue
                ifile_path = os.path.join(self.config.internationalStocks_local_data_dir, f"{iticker}.csv")
                idata.to_csv(ifile_path)
                print(f"Saved data for {iticker} to {ifile_path}")
            except Exception as e:
                print(f"Failed to download data for {iticker}: {e}")
        
        # Handle date parsing for Pakistani tickers
        pstart_date = (
            self.config.pstart_date
            if isinstance(self.config.pstart_date, date)
            else datetime.strptime(self.config.pstart_date, "%Y-%m-%d").date()
        )
        pend_date = (
            date.today()
            if isinstance(self.config.pend_date, str) and self.config.pend_date.lower() == "now"
            else self.config.pend_date
            if isinstance(self.config.pend_date, date)
            else datetime.strptime(self.config.pend_date, "%Y-%m-%d").date()
        )

        # Download stock data for Pakistani tickers
        for pticker in ptickers:
            try:
                print(f"Downloading data for {pticker}...")
                pdata = stocks(pticker, start=pstart_date, end=pend_date)
                if pdata.empty:
                    print(f"No data found for {pticker}. Skipping.")
                    continue
                pfile_path = os.path.join(self.config.pakistanStocks_local_data_dir, f"{pticker}.csv")
                pdata.to_csv(pfile_path)
                print(f"Saved data for {pticker} to {pfile_path}")
            except Exception as e:
                print(f"Failed to download data for {pticker}: {e}")
