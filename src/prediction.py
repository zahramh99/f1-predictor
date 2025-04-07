import pandas as pd
from typing import Dict, List
from joblib import load
from pathlib import Path
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RacePredictor:
    def __init__(self, model_path: str = None):
        if model_path:
            self.model = load(Path(model_path))
        else:
            self.model = None
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        self.model = load(Path(model_path))
    
    def predict_race(self, qualifying_data: pd.DataFrame, sector_data: pd.DataFrame) -> pd.DataFrame:
        """Predict race results based on qualifying and sector times"""
        if self.model is None:
            raise ValueError("No model loaded for predictions")
        
        # Merge data
        merged = qualifying_data.merge(
            sector_data, 
            left_on="DriverCode", 
            right_on="Driver", 
            how="left"
        ).fillna(0)
        
        # Prepare features
        features = ["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
        X = merged[features]
        
        # Make predictions
        merged["PredictedRaceTime (s)"] = self.model.predict(X)
        
        # Rank drivers
        result = merged.sort_values(by="PredictedRaceTime (s)")[
            ["Driver", "PredictedRaceTime (s)"]
        ]
        
        return result.reset_index(drop=True)
    
    def generate_report(self, predictions: pd.DataFrame) -> str:
        """Generate a printable race prediction report"""
        report = ["🏁 Formula 1 Race Predictions 🏁", "="*40]
        
        for i, (_, row) in enumerate(predictions.iterrows(), 1):
            report.append(f"{i}. {row['Driver']}: {row['PredictedRaceTime (s)']:.3f}s")
        
        return "\n".join(report)