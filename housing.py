# ✅ Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from statsmodels.tsa.arima.model import ARIMA

print("🚀 Script Started...")

# ✅ Define file paths
file_paths = {
    "ForSale_Inventory": "/Users/ngoubimaximilliandiamgha/Desktop/Metro_invt_fs_uc_sfrcondo_sm_month.csv",
    "Market_Temp_Index": "/Users/ngoubimaximilliandiamgha/Desktop/Metro_market_temp_index_uc_sfrcondo_month.csv",
    "Days_On_Market": "/Users/ngoubimaximilliandiamgha/Desktop/Metro_mean_doz_pending_uc_sfrcondo_sm_month.csv",
    "New_Construction_Sales": "/Users/ngoubimaximilliandiamgha/Desktop/Metro_new_con_sales_count_raw_uc_sfrcondo_month.csv",
    "New_Homeowner_Income": "/Users/ngoubimaximilliandiamgha/Desktop/Metro_new_homeowner_income_needed_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "Sales_Count": "/Users/ngoubimaximilliandiamgha/Desktop/Metro_sales_count_now_uc_sfrcondo_month.csv",
    "Home_Value_Forecast": "/Users/ngoubimaximilliandiamgha/Desktop/Metro_zhvf_growth_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "Home_Value_Index": "/Users/ngoubimaximilliandiamgha/Desktop/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "Rent_Index": "/Users/ngoubimaximilliandiamgha/Desktop/Metro_zori_uc_sfrcondomfr_sm_month.csv",
    "Rent_Forecast": "/Users/ngoubimaximilliandiamgha/Desktop/National_zorf_growth_uc_sfr_sm_month.csv",
}


# ✅ Function to load and clean dataset
def load_clean_csv(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df = df.drop(columns=["RegionID", "SizeRank", "RegionType", "StateName"], errors="ignore")
        return df
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {e}")
        return None


# ✅ Load datasets
print("📂 Loading datasets...")
datasets = {name: load_clean_csv(path) for name, path in file_paths.items()}
datasets = {name: df for name, df in datasets.items() if df is not None}

if not datasets:
    print("❌ No datasets were loaded! Check file paths.")
    exit()

print(f"✅ Successfully loaded {len(datasets)} datasets.")


# ✅ Function to reshape dataset
def reshape_data(df, name):
    df_melted = df.melt(id_vars=['RegionName'], var_name='Date', value_name=name)
    df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%Y-%m', errors='coerce')
    return df_melted


# ✅ Reshape datasets
reshaped_datasets = {name: reshape_data(df, name) for name, df in datasets.items()}

# ✅ Debug: Print the shape of each dataset before merging
print("🔗 Merging datasets...")
for name, df in reshaped_datasets.items():
    print(f"📊 {name} dataset shape: {df.shape}")

# ✅ Merge datasets
merged_df = None
for name, df in reshaped_datasets.items():
    if merged_df is None:
        merged_df = df
    else:
        merged_df = merged_df.merge(df, on=['RegionName', 'Date'], how='left')

# ✅ Debug: Print merged dataset shape
if merged_df is not None:
    print(f"✅ Merged dataset shape: {merged_df.shape}")
else:
    print("❌ Merging failed! No data to merge.")
    exit()

# ✅ Check if the merged dataset is empty
if merged_df.empty:
    print("❌ Merged dataset is empty! Please check the individual dataset contents.")
    exit()

# ✅ Fix 'Home_Value_Index' Column Issue
correct_col_name = "ZHVI"  # Replace this if needed
if correct_col_name in merged_df.columns:
    merged_df.rename(columns={correct_col_name: "Home_Value_Index"}, inplace=True)
else:
    print("❌ 'Home_Value_Index' column is missing!")
    print("🔍 Available columns:", merged_df.columns)
    exit()

print(f"✅ Merged dataset has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")

# ✅ Feature Engineering
print("🧪 Performing feature engineering...")
merged_df["ZHVI_Growth"] = merged_df.groupby("RegionName")["Home_Value_Index"].pct_change() * 100
merged_df["ZORI_Growth"] = merged_df.groupby("RegionName")["Rent_Index"].pct_change() * 100
merged_df["Sales_Growth"] = merged_df.groupby("RegionName")["Sales_Count"].pct_change() * 100
merged_df["Affordability_Index"] = np.where(
    merged_df["Sales_Count"] > 0,
    merged_df["New_Homeowner_Income"] / merged_df["Sales_Count"],
    np.nan
)
merged_df["Supply_Demand_Ratio"] = merged_df["ForSale_Inventory"] / (merged_df["Sales_Count"] + 1)

# ✅ Drop NaN values from target variable
merged_df = merged_df.dropna(subset=["Home_Value_Index"])

# ✅ Reduce dataset size for faster training
merged_df = merged_df.sample(frac=0.02, random_state=42)

# ✅ Train-Test Split
print("📊 Splitting dataset for training...")
target = "Home_Value_Index"
features = ["ZHVI_Growth", "ZORI_Growth", "Sales_Growth", "Affordability_Index", "Supply_Demand_Ratio"]
X_train, X_test, y_train, y_test = train_test_split(merged_df[features], merged_df[target], test_size=0.2,
                                                    random_state=42)

# ✅ Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train XGBoost Model
print("🎯 Training XGBoost Model...")
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1, max_depth=5)

try:
    xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], early_stopping_rounds=5, verbose=False)
    print("✅ Model training complete!")

    # ✅ SAVE MODEL USING ABSOLUTE PATH
    model_path = "/Users/ngoubimaximilliandiamgha/Desktop/PythonProject6/xgboost_housing_model.pkl"
    scaler_path = "/Users/ngoubimaximilliandiamgha/Desktop/PythonProject6/scaler.pkl"

    print("💾 Saving model and scaler...")
    joblib.dump(xgb_model, model_path)
    joblib.dump(scaler, scaler_path)

    # ✅ VERIFY MODEL SAVE
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"✅ Model saved successfully at: {model_path}")
        print(f"✅ Scaler saved successfully at: {scaler_path}")
    else:
        print("❌ Model or Scaler was NOT saved!")

except Exception as e:
    print(f"❌ Model training failed: {e}")

# ✅ Visualize Predictions
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=xgb_model.predict(X_test_scaled), alpha=0.7)
plt.xlabel("Actual Home Price")
plt.ylabel("Predicted Home Price")
plt.title("Actual vs. Predicted House Prices")
plt.grid(True)
plt.show()

print("🚀 Model training completed successfully!")
