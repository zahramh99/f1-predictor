import fastf1
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path
from config import settings
from tqdm import tqdm

class DataProcessor:
    def __init__(self):
        fastf1.Cache.enable_cache(settings.CACHE_DIR)
        
    def load_session_data(self, year: int, race_name: str, session_type: str = "R") -> fastf1.core.Session:
        """Load session data from FastF1"""
        try:
            session = fastf1.get_session(year, race_name, session_type)
            session.load()
            return session
        except Exception as e:
            raise ValueError(f"Error loading session data: {str(e)}")
    
    def process_lap_data(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Process lap data into a clean DataFrame"""
        laps = session.laps.copy()
        
        # Select relevant columns
        cols = ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", 
                "Compound", "Position", "TyreLife", "PitOutTime", "PitInTime"]
        laps = laps[[c for c in cols if c in laps.columns]]
        
        # Convert times to seconds
        time_cols = [c for c in laps.columns if "Time" in c]
        for col in time_cols:
            laps[f"{col} (s)"] = laps[col].dt.total_seconds()
        
        # Drop original time columns
        laps.drop(columns=time_cols, inplace=True)
        
        # Add session info
        laps["Year"] = session.event.year
        laps["Event"] = session.event.EventName
        laps["Session"] = session.name
        
        return laps.dropna(subset=["LapTime (s)"])
    
    def create_driver_mapping(self, drivers: list) -> Dict[str, str]:
        """Create mapping from driver names to 3-letter codes"""
        return {d: d[:3].upper() for d in drivers}
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to CSV"""
        path = settings.DATA_DIR / "processed" / filename
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)