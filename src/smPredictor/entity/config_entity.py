from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    tickers: list
    start_date: str
    end_date: str
    local_data_dir: Path
    
    