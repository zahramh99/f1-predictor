from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {
            "GradientBoosting": GradientBoostingRegressor(**settings.DEFAULT_MODEL_PARAMS["GradientBoostingRegressor"]),
            "RandomForest": RandomForestRegressor(**settings.DEFAULT_MODEL_PARAMS["RandomForestRegressor"])
        }
        self.scaler = StandardScaler()
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target"""
        X = df[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "TyreLife"]]
        y = df["LapTime (s)"]
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_name: str = "GradientBoosting") -> Tuple[object, Dict]:
        """Train the selected model"""
        model = self.models[model_name]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        logger.info(f"{model_name} trained with MAE: {metrics['MAE']:.3f}, RMSE: {metrics['RMSE']:.3f}")
        
        return pipeline, metrics
    
    def save_model(self, model: object, filename: str):
        """Save trained model to disk"""
        path = settings.MODEL_DIR / filename
        path.parent.mkdir(exist_ok=True)
        dump(model, path)
        logger.info(f"Model saved to {path}")