# Enhanced ML Models Directory

This directory contains the enhanced ML models trained in Google Colab for the FinCoach ML Enhancement project.

## Expected Model Files

After training in Colab, this directory should contain:

- `user_segmentation_model.pkl` - K-means clustering model for user behavioral segmentation
- `segmentation_scaler.pkl` - StandardScaler for segmentation features
- `enhanced_cql_model.pth` - Enhanced Q-Network for recommendations (35+ dimensional)
- `spending_predictor.pth` - LSTM model for spending prediction
- `spending_scaler.pkl` - StandardScaler for spending prediction features
- `goal_achievement_model.pkl` - Random Forest for goal achievement probability
- `goal_scaler.pkl` - StandardScaler for goal prediction features
- `feature_config.json` - Feature engineering configuration
- `segment_profiles.json` - User segment characteristics and profiles
- `bandit_state.json` - Multi-armed bandit strategy performance state
- `model_metadata.json` - Model metadata and performance metrics
- `INTEGRATION_GUIDE.md` - Integration instructions for backend

## Usage

These models will be loaded by the enhanced `ml_service.py` to provide:

1. **User Behavioral Segmentation** - Classify users into 5 behavioral segments
2. **Enhanced Recommendations** - 35+ dimensional state vectors with segment-aware strategies
3. **Spending Prediction** - LSTM-based weekly spending forecasts
4. **Goal Achievement Prediction** - Probability of reaching financial goals
5. **Behavioral Analytics** - Advanced user insights and patterns

## Training

Models are trained using the implementation guide in `notebooks/FinCoach_ML_Enhancement_Implementation.md` with the anonymized transaction dataset.

## Integration

Follow the `BACKEND_ML_INTEGRATION_PLAN.md` for step-by-step backend integration instructions.
