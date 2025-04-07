# f1-predictor
This repository contains a comprehensive machine learning pipeline for predicting Formula 1 race results. 
## 🚀 Project Overview
This repository contains a comprehensive machine learning pipeline for predicting Formula 1 race results. The system leverages:
Advanced feature engineering (sector times, qualifying performance, tire data)
Multiple ML models (Gradient Boosting, Random Forest) with hyperparameter tuning
Modular architecture for easy maintenance and extension
Professional CI/CD-ready code structure

## Key innovations:
Hybrid prediction system combining qualifying and historical race data
Automated data processing pipeline
Model performance tracking
Interactive CLI interface

## 🏁 How It Works
1-Data Ingestion
Automatic session loading from FastF1
Custom data integration
Time conversion and normalization
2-Feature Engineering
Sector time analysis
Tire performance metrics
Driver-specific baselines
3-Model Training
Gradient Boosting (primary)
Random Forest (comparison)
Hyperparameter tuning
4-Prediction
Race time estimation
Driver ranking
Performance reporting

## Dependencies
Core Requirements
fastf1>=3.0.0
scikit-learn>=1.0.0
pandas>=1.3.0

Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

Infrastructure
python-dotenv>=0.19.0
joblib>=1.1.0

## 📌 Future Improvements
1-Near-Term
Weather condition integration
Pit stop strategy simulation
Driver form tracking
2-Long-Term
Neural network approaches
Real-time prediction engine
Probabilistic outcome modeling

## 🔗 Project Attribution
This prediction model builds upon the foundational work from @mar-antaya/2025_f1_predictions with significant enhancements:
1-Key Improvements Added:
Architecture Upgrade
Implemented modular OOP design (DataProcessor, ModelTrainer, RacePredictor classes)
Added configuration management system
Introduced proper logging
2-Feature Expansion
Incorporated tire life metrics
Added pit stop timing analysis
Implemented weather data integration framework
3-Technical Advancements
Multi-model comparison system
Automated hyperparameter tuning
Model persistence with joblib

## Acknowledgement Statement:
"This project originated from mar-antaya's innovative F1 prediction concept. We've extended the work with professional software engineering practices and additional data dimensions while maintaining the core predictive philosophy."

