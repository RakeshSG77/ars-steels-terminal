import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestRegressor, 
    VotingRegressor, 
    HistGradientBoostingRegressor
)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. ADVANCED FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    d = df.copy()
    d['Price_Momentum_1W'] = d.get('ARS_Price_Lag_1W', 0) - d.get('ARS_Price_Lag_2W', 0)
    d['Price_Velocity'] = d.get('ARS_Price_Lag_1W', 0) - d.get('ARS_Price_Rolling_Mean_4W', 0)
    
    # Upgraded Cost Pressure Index utilizing the new global datasets
    d['Cost_Pressure_Index'] = (
        0.20 * d.get('Scrap_Metal_Price_per_Ton', 0) +
        0.20 * d.get('Turkey_Scrap_Metal_Price_INR_per_Ton', 0) +
        0.25 * d.get('Iron_Ore_Price_per_Ton', 0) +
        0.10 * d.get('CDRI_Raipur_Price_INR_per_Ton', 0) +
        0.10 * d.get('Coking_Coal_Price_per_Ton', 0) +
        0.05 * d.get('RB2_Coal_Gangavaram_Price_INR_per_Ton', 0) +
        0.10 * d.get('Diesel_Price_Chennai', 0) * 10
    )
    
    d['Competitor_Premium'] = d.get('ARS_Price_Lag_1W', 0) - d.get('Competitor_Avg_Price_per_Ton', 0)
    d['Volatility_Proxy'] = abs(d.get('ARS_Price_Lag_1W', 0) - d.get('ARS_Price_Lag_2W', 0))
    return d

def main():
    print("=" * 70)
    print(" ⚡ ARS STEELS — PRODUCTION VOTING ENSEMBLE")
    print("=" * 70)

    print("[1/4] Loading and engineering features...")
    try:
        # Loaded the new perfectly accurate dataset
        df = pd.read_csv('ARS_Steels_Accurate_Extracted_Dataset.csv')
    except FileNotFoundError:
        print("FATAL ERROR: 'ARS_Steels_Accurate_Extracted_Dataset.csv' not found.")
        return

    df = engineer_features(df)

    # Core high-signal features including new additions
    FEATURES = [
        'Diesel_Price_Chennai', 'RBI_Repo_Rate', 'USD_to_INR',
        'Iron_Ore_Price_per_Ton', 'Coking_Coal_Price_per_Ton', 'Scrap_Metal_Price_per_Ton',
        'Turkey_Scrap_Metal_Price_INR_per_Ton', 'CDRI_Raipur_Price_INR_per_Ton', 
        'PDRI_Bellary_Price_INR_per_Ton', 'PDRI_Hyderabad_Price_INR_per_Ton', 
        'PDRI_Raipur_Price_INR_per_Ton', 'RB2_Coal_Gangavaram_Price_INR_per_Ton',
        'Competitor_Avg_Price_per_Ton', 'ARS_Price_Lag_1W', 'ARS_Price_Lag_2W',
        'ARS_Price_Rolling_Mean_4W', 'Price_Momentum_1W', 'Price_Velocity', 
        'Cost_Pressure_Index', 'Competitor_Premium', 'Volatility_Proxy'
    ]

    X = df[FEATURES].fillna(0)
    y = df['ARS_TMT_Price_per_Ton']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print(f"      Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # ─────────────────────────────────────────────
    # 2. THE VOTING ARCHITECTURE
    # ─────────────────────────────────────────────
    print("[2/4] Initializing Stable Stack...")

    estimators = [
        ('xgb', xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=6, 
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )),
        ('hgb', HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.03, max_depth=6, 
            l2_regularization=1.0, random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_split=4, 
            random_state=42, n_jobs=-1
        ))
    ]

    ensemble = VotingRegressor(estimators=estimators, n_jobs=-1)

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', ensemble)
    ])

    # ─────────────────────────────────────────────
    # 3. TRAINING & EVALUATION
    # ─────────────────────────────────────────────
    print("[3/4] Commencing Training...")
    pipeline.fit(X_train, y_train)

    print("[Evaluating out-of-sample accuracy...]")
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100

    print("\n" + "=" * 70)
    print("  🏆 ARS STEELS — ENTERPRISE PERFORMANCE METRICS")
    print("=" * 70)
    print(f"  Mean Absolute Error (MAE):   ₹{mae:.2f}")
    print(f"  Root Mean Squared Error:     ₹{rmse:.2f}")
    print(f"  Mean Absolute % Error:       {mape:.3f}%")
    print(f"  Accuracy (R² Score):         {r2*100:.4f}%")
    print("=" * 70)

    # ─────────────────────────────────────────────
    # 4. EXPORT ARTIFACTS
    # ─────────────────────────────────────────────
    print("\n[4/4] Serializing and exporting Pipeline...")
    joblib.dump(pipeline, 'ars_pricing_pipeline.pkl')
    joblib.dump(FEATURES, 'ars_features.pkl')

    print("✅ Maximum Accuracy Pipeline compiled and saved successfully.")
    print("   Launch the UI: `streamlit run app.py`")

if __name__ == "__main__":
    main()
