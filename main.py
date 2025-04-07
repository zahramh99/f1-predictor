import click
from pathlib import Path
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.prediction import RacePredictor
from config import settings
import pandas as pd

@click.command()
@click.option('--year', type=int, help='Year of historical race to analyze')
@click.option('--race', type=str, help='Name of the race circuit')
@click.option('--train', is_flag=True, help='Train a new model')
@click.option('--predict-2025', is_flag=True, help='Predict 2025 race results')
@click.option('--model-path', type=str, default=None, help='Path to trained model')
def main(year, race, train, predict_2025, model_path):
    """Formula 1 Race Predictor CLI"""
    
    processor = DataProcessor()
    trainer = ModelTrainer()
    predictor = RacePredictor(model_path)
    
    if train:
        # Training mode
        click.echo(f"Training model on {year} {race} data...")
        session = processor.load_session_data(year, race)
        laps = processor.process_lap_data(session)
        X, y = trainer.prepare_data(laps)
        model, metrics = trainer.train_model(X, y)
        trainer.save_model(model, f"f1_model_{year}_{race.replace(' ', '_')}.pkl")
        click.echo(f"Model trained with MAE: {metrics['MAE']:.3f}")
    
    if predict_2025:
        # Prediction mode
        if not model_path:
            model_path = settings.MODEL_DIR / "f1_model_2024_China.pkl"
            predictor.load_model(model_path)
        
        click.echo(f"Predicting 2025 {race} results...")
        
        # Example 2025 qualifying data (in a real app, this would come from an API or file)
        qualifying_2025 = pd.DataFrame({
            "Driver": ["Verstappen", "Hamilton", "Norris", "Leclerc", "Perez"],
            "DriverCode": ["VER", "HAM", "NOR", "LEC", "PER"],
            "QualifyingTime (s)": [90.123, 90.456, 90.789, 91.012, 91.345]
        })
        
        # Get historical sector times
        session = processor.load_session_data(year, race)
        laps = processor.process_lap_data(session)
        sector_times = laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
        
        # Make predictions
        predictions = predictor.predict_race(qualifying_2025, sector_times)
        report = predictor.generate_report(predictions)
        
        click.echo("\n" + report)

if __name__ == "__main__":
    main()