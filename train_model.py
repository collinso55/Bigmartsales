import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from scipy.stats import mode

def train_pipeline():
    # 1. Load Data
    print("Loading datasets...")
    # Using relative paths for better portability
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Combine for processing (to ensure consistent encoding/imputation)
    train['source'] = 'train'
    test['source'] = 'test'
    data = pd.concat([train, test], ignore_index=True)

    # 2. Data Engineering & Cleaning (Phase 1)
    print("Phase 1: Data Engineering & Cleaning...")
    
    # Impute Item_Weight: mean weight of specific Item_Identifier
    item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
    def impute_weight(cols):
        weight = cols[0]
        identifier = cols[1]
        if pd.isnull(weight):
            if identifier in item_avg_weight.index:
                return item_avg_weight.loc[identifier].values[0]
            else:
                return data['Item_Weight'].mean() # Fallback if ID not found
        else:
            return weight
    data['Item_Weight'] = data[['Item_Weight', 'Item_Identifier']].apply(impute_weight, axis=1)

    # Impute Outlet_Size: mode of that Outlet_Type
    def get_mode(x):
        # Filtering out NaNs for mode calculation
        mode_vals = x.dropna().astype(str)
        if mode_vals.empty: return "Medium"
        vals, counts = np.unique(mode_vals, return_counts=True)
        return vals[np.argmax(counts)]

    outlet_size_mode = data.groupby('Outlet_Type')['Outlet_Size'].apply(get_mode).to_dict()
    
    def impute_size(cols):
        size = cols[0]
        type = cols[1]
        if pd.isnull(size) or size == 'nan':
            return outlet_size_mode.get(type, 'Medium')
        else:
            return size
            
    data['Outlet_Size'] = data[['Outlet_Size', 'Outlet_Type']].apply(impute_size, axis=1)

    # Handle Item_Visibility (replace 0 with mean visibility of that product)
    visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
    def impute_visibility(cols):
        visibility = cols[0]
        item = cols[1]
        if visibility == 0:
            if item in visibility_avg.index:
                return visibility_avg.loc[item].values[0]
            else:
                return data['Item_Visibility'].mean()
        else:
            return visibility
    data['Item_Visibility'] = data[['Item_Visibility', 'Item_Identifier']].apply(impute_visibility, axis=1)

    # Feature Engineering
    # Create Item_Type_Combined ('Food', 'Drinks', 'Non-Consumable')
    data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
    data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food', 'DR':'Drinks', 'NC':'Non-Consumable'})

    # Calculate Outlet_Years (Current Year - Outlet_Establishment_Year)
    # The dataset originated around 2013, so using 2013 as baseline is standard for this task
    data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

    # Standardize Item_Fat_Content
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
    # Mark NC items as Non-Edible fat content
    data.loc[data['Item_Type_Combined']=="Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"

    # Encoding (Phase 1)
    print("Encoding features...")
    # Label Encoding for ordinal variables (manual mapping for order control)
    size_map = {"High": 0, "Medium": 1, "Small": 2}
    loc_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
    data['Outlet_Size_Label'] = data['Outlet_Size'].map(size_map)
    data['Outlet_Location_Type_Label'] = data['Outlet_Location_Type'].map(loc_map)
    
    # One-Hot Encoding for nominal variables
    data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Type', 'Item_Type_Combined'])

    # Drop non-useful columns
    cols_to_drop = ['Item_Type', 'Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type']
    data.drop(cols_to_drop, axis=1, inplace=True)

    # Split back into train and test
    train = data.loc[data['source']=="train"].drop(['source'], axis=1)
    test_final = data.loc[data['source']=="test"].drop(['source', 'Item_Outlet_Sales'], axis=1)

    X = train.drop(['Item_Outlet_Sales'], axis=1)
    y = train['Item_Outlet_Sales']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Machine Learning Pipeline (Phase 2)
    print("Phase 2: Training Line-up...")
    
    # Linear Regression (Baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    lr_rmse = np.sqrt(mean_squared_error(y_val, lr_pred))
    lr_r2 = r2_score(y_val, lr_pred)
    print(f"Baseline Linear Regression -> RMSE: {lr_rmse:.2f}, R2: {lr_r2:.4f}")

    # Advanced Regressor (XGBoost or GradientBoosting fallback)
    try:
        import xgboost as xgb
        print("Model: Initializing XGBoost Regressor...")
        advanced_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        print("Model: XGBoost not found. Falling back to GradientBoostingRegressor...")
        advanced_model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }
    
    print("Optimization: Tuning advanced model...")
    grid_search = GridSearchCV(advanced_model, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_adv = grid_search.best_estimator_
    adv_pred = best_adv.predict(X_val)
    adv_rmse = np.sqrt(mean_squared_error(y_val, adv_pred))
    adv_r2 = r2_score(y_val, adv_pred)
    model_name = "XGBoost" if "XGB" in str(type(best_adv)) else "GradientBoosting"
    print(f"{model_name} Results -> RMSE: {adv_rmse:.2f}, R2: {adv_r2:.4f}")

    # 4. Finalize Model Selection
    if adv_rmse < lr_rmse:
        print(f"Model Selection: {model_name} selected as the production model.")
        best_model = best_adv
    else:
        print("Model Selection: Linear Regression selected as the production model.")
        best_model = lr

    joblib.dump(best_model, 'best_sales_model.pkl')
    
    # Save preprocessing metadata for UI integration
    preprocessing_data = {
        'columns': X.columns.tolist(),
        'feature_importance': None
    }
    
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = pd.Series(best_model.feature_importances_, X.columns).sort_values(ascending=False)
        preprocessing_data['feature_importance'] = feat_imp.to_dict()
    
    joblib.dump(preprocessing_data, 'preprocessing_info.pkl')
    print("Success: System ready for deployment.")

if __name__ == "__main__":
    train_pipeline()
