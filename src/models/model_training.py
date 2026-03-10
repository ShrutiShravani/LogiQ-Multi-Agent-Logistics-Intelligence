import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import yaml
from mlflow.tracking import MlflowClient
import platform

def model_train():
    # 1. Setup MLflow Experiment
    
    with open("params.yaml") as f:
        params_yaml=yaml.safe_load(f)


    mlflow.set_experiment("Logistics_Pricing")
    
    with mlflow.start_run():
        # Load the transformed data we just created
        dataset = pd.read_csv(r"data\transformed\train_final.csv")

        # 2. Senior Way to split: Target is the LAST column
        X = dataset.iloc[:, :-1]  # All columns except target_price
        y = dataset.iloc[:, -1]   # The target_price column
        
        weights = np.ones(len(y))
        if 'type_truck' in X.columns:
            weights[X['type_truck']==1]*=5.0
        if 'type_van' in X.columns:
            weights[X['type_van']==1]*=2.0

        # Drop non-numeric columns like datetime if they exist
        X = X.select_dtypes(include=[np.number])
        y_log = np.log1p(y)

        X_train, X_val, y_train, y_val,w_train, w_val = train_test_split(
            X, y_log,weights, test_size=0.2, random_state=42
        )

        print(f"Train set: {len(X_train)} rows")
        print(f"Val set: {len(X_val)} rows")
        
       
        # 3. Switching to REGRESSOR (Since we are predicting Price)
        xgb_params = {
            "n_estimators": params_yaml['train']['n_estimators'],
            "max_depth":params_yaml['train']['max_depth'],
            "learning_rate": params_yaml['train']['learning_rate'],
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42
        }
        
        model = xgb.XGBRegressor(**xgb_params)

        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False
        )

        # 4. Predictions & Regression Metrics
        y_pred = np.expm1(model.predict(X_val))
        y_actual = np.expm1(y_val)

        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)

        # 5. Corrected MLflow Logging (Fixed typos)
        mlflow.log_params(xgb_params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        print(f"\nMAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")

        os.makedirs("reports",exist_ok=True)
        with open("reports/metrics.json","w") as f:
            json.dump({"mae": mae, "rmse": rmse, "r2": r2}, f)


        # 6. Feature importance
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model, max_num_features=12)
        plt.title("Pricing Model - Feature Importance")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        
        # 7. Save Model correctly for MLflow Registry
        os.makedirs("trained_models", exist_ok=True)
        model.save_model("trained_models/pricing_xgb_model.json")
        mlflow.xgboost.log_model(model,artifact_path="pricing_model")

        #register_model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/pricing_model"
        model_details = mlflow.register_model(model_uri=model_uri, name="xgboost_logistics_pricing")

        # 3. Move it to "Production" (So the PricingAgent can find it)
        client = MlflowClient()
        client.set_registered_model_alias("xgboost_logistics_pricing", "Production", model_details.version)
        print(f"Model version {model_details.version} is now LIVE in Production.")
        print("\nModel saved and logged to MLflow.")

    return model, X_val, y_val, y_pred

if __name__ == "__main__":
    model_train()