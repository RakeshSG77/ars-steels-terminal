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
    d['Cost_Pressure_Index'] = (
        0.40 * d.get('Scrap_Metal_Price_per_Ton', 0) +
        0.35 * d.get('Iron_Ore_Price_per_Ton', 0) +
        0.15 * d.get('Coking_Coal_Price_per_Ton', 0) +
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
        df = pd.read_csv('ARS_Steels_Dataset.csv')
    except FileNotFoundError:
        print("FATAL ERROR: 'ARS_Steels_Dataset.csv' not found.")
        return

    df = engineer_features(df)

    # Core high-signal features
    FEATURES = [
        'Diesel_Price_Chennai', 'RBI_Repo_Rate', 'USD_to_INR',
        'Iron_Ore_Price_per_Ton', 'Coking_Coal_Price_per_Ton', 'Scrap_Metal_Price_per_Ton',
        'Competitor_Avg_Price_per_Ton', 'ARS_Price_Lag_1W', 'ARS_Price_Lag_2W',
        'ARS_Price_Rolling_Mean_4W', 'Price_Momentum_1W', 'Price_Velocity', 
        'Cost_Pressure_Index', 'Competitor_Premium', 'Volatility_Proxy'
    ]

    X = df[FEATURES].fillna(0)
    y = df['ARS_TMT_Price_per_Ton']

    # CRITICAL FIX: Random split ensures train/test sets share the same price distributions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print(f"      Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # ─────────────────────────────────────────────
    # 2. THE VOTING ARCHITECTURE (BULLETPROOF)
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

    # CRITICAL FIX: VotingRegressor averages the models instead of trying to train on tiny data chunks
    ensemble = VotingRegressor(
        estimators=estimators,
        n_jobs=-1
    )

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
# ─────────────────────────────────────────────
    # 5. THE "COLD HARD TRUTH" TEST (Walk-Forward)
    # ─────────────────────────────────────────────
    print("\n[Running Strict Walk-Forward Validation...]")
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=5)
    true_maes = []
    
    # We test the model strictly moving forward in time, no random shuffling
    for train_index, test_index in tscv.split(X):
        cv_X_train, cv_X_test = X.iloc[train_index], X.iloc[test_index]
        cv_y_train, cv_y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Clone the pipeline to start fresh for each step
        from sklearn.base import clone
        cv_pipeline = clone(pipeline)
        
        cv_pipeline.fit(cv_X_train, cv_y_train)
        cv_preds = cv_pipeline.predict(cv_X_test)
        true_maes.append(mean_absolute_error(cv_y_test, cv_preds))
        
    real_world_mae = np.mean(true_maes)
    print("=" * 70)
    print(f" 📉 TRUE REAL-WORLD EXPECTED ERROR (Walk-Forward MAE): ₹{real_world_mae:.2f}")
    print("=" * 70)
if __name__ == "__main__":
    main()