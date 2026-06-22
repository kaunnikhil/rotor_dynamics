import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score

def train_surrogate_model():

    #training Gaussian Process model so as to have uncertainty quantification in final results

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), r"D:\cv_projects_2\rotor_dynamics\data\raw\rotor_dataset.csv"))
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    features = ['d_i', 'd_o', 'L_total', 'm_disk', 'k_bearing']
    targets = ['freq_1_hz', 'freq_2_hz']
    
    X = df[features].values
    y = df[targets].values
    
    # 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")
    
    # feature scaling
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # GPR kernel
    # ConstantKernel scales the amplitude, Matern kernel handles the physical smoothness
    #length scale answers how much change in variable is required to change o/p significantly.
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(len(features)), nu=1.5)
    # normalize_y=True automatically scales the targets to 0 mean, 1 variance internally
    # n_restarts_optimizer=10 ensures the model doesn't get stuck in a local minimum during training and try from 10 different starting points each time to keep the best result
    gpr_model = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=10, 
        normalize_y=True,
        random_state=42
    )
    
    gpr_model.fit(X_train_scaled, y_train)
    
    #evaluating
    print("\nEvaluating Model on Test Set:")
    y_pred, y_std = gpr_model.predict(X_test_scaled, return_std=True)
    
  
    for i, target_name in enumerate(targets):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"--- {target_name} ---")
        print(f"  R^2 Score: {r2:.4f} (Closer to 1 is better)")
        print(f"  RMSE:      {rmse:.2f} Hz")
        
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), r"D:\cv_projects_2\rotor_dynamics\src\ml"))
    
    joblib.dump(gpr_model, os.path.join(models_dir, 'gpr_model.pkl'))
    joblib.dump(scaler_X, os.path.join(models_dir, 'scaler_X.pkl'))

if __name__ == "__main__":
    train_surrogate_model()