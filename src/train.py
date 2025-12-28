import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime


def train_model():
    mlflow.set_experiment("House Price Prediction")
    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        df = pd.read_csv('data/house_data.csv')
        print(f"Loaded {len(df)} records")

        X = df.drop('price', axis=1)
        y = df['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_samples", len(df))

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        print("Training model...")
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)

        mlflow.sklearn.log_model(model, "model")

        # Save model locally
        joblib.dump(model, 'models/random_forest_model.joblib')

        # Save feature names
        joblib.dump(list(X.columns), 'models/feature_names.joblib')

        print("\n=== Model Performance ===")
        print(f"Train MAE: ${train_mae:,.2f}")
        print(f"Test MAE: ${test_mae:,.2f}")
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        
        return model


if __name__ == "__main__":
    train_model()