
# =====================================================
# AIR QUALITY ANALYSIS AND PREDICTION OF INDIA
# B.Tech Data Science Project
# =====================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import os
import joblib

sns.set_theme(style="whitegrid")

os.makedirs("output", exist_ok=True)
os.makedirs("graphs", exist_ok=True)


# 2. LOAD DATASET
df = pd.read_csv("air_quality.csv")

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())


# 3. DATA PREPROCESSING

print("\nDATA PREPROCESSING STARTED")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert numerical columns
num_cols = [
    "latitude",
    "longitude",
    "pollutant_min",
    "pollutant_max",
    "pollutant_avg"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert date
df["last_update"] = pd.to_datetime(
    df["last_update"],
    format="%d-%m-%Y %H:%M:%S",
    errors="coerce"
)

# Remove rows where target is missing
df.dropna(subset=["pollutant_avg"], inplace=True)

# Remove invalid coordinates
df = df[
    df["latitude"].between(-90, 90) &
    df["longitude"].between(-180, 180)
]

# Remove impossible pollutant readings
df = df[
    (df["pollutant_avg"] >= 0) &
    (
        df["pollutant_min"].isna() |
        (df["pollutant_min"] >= 0)
    ) &
    (
        df["pollutant_max"].isna() |
        (df["pollutant_max"] >= 0)
    )
]

# Remove logically inconsistent minimum and maximum
df = df[
    df["pollutant_min"].isna() |
    df["pollutant_max"].isna() |
    (df["pollutant_min"] <= df["pollutant_max"])
]

# Standardize text
cat_cols = [
    "country",
    "state",
    "city",
    "pollutant_id"
]

for col in cat_cols:
    df[col] = df[col].astype("string").str.strip().str.upper()

# Fill missing numerical values
for col in ["latitude", "longitude",
            "pollutant_min", "pollutant_max"]:

    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

print("\nCLEANED DATA SHAPE")
print(df.shape)

df.to_csv("output/cleaned_data.csv", index=False)

print("PREPROCESSING COMPLETED")


# 4. EXPLORATORY DATA ANALYSIS

print("\nEXPLORATORY DATA ANALYSIS")

print("\nPOLLUTANT COUNTS")
print(df["pollutant_id"].value_counts())

print("\nPOLLUTANT STATISTICS")
print(
    df.groupby("pollutant_id")["pollutant_avg"]
    .agg(["mean", "median", "min", "max", "std"])
)

print("\nTOP 10 CITIES BY PM2.5")

pm25 = df[df["pollutant_id"] == "PM2.5"]

print(
    pm25.groupby("city")["pollutant_avg"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# NumPy calculations

values = df["pollutant_avg"].to_numpy()

print("\nMEAN:", np.mean(values))
print("MEDIAN:", np.median(values))
print("STANDARD DEVIATION:", np.std(values))
print("VARIANCE:", np.var(values))
print("MINIMUM:", np.min(values))
print("MAXIMUM:", np.max(values))


# 5. DATA VISUALIZATION

# Graph 1: Histogram

plt.figure(figsize=(8, 5))

plt.hist(df["pollutant_avg"], bins=30, edgecolor="black")

plt.title("Pollutant Distribution")
plt.xlabel("Average Pollutant")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("graphs/01_histogram.png")
plt.show()


# Graph 2: Top 10 PM2.5 Cities

top_cities = (
    pm25.groupby("city")["pollutant_avg"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(9, 5))

top_cities.sort_values().plot(kind="barh")

plt.title("Top 10 Cities by PM2.5")
plt.xlabel("Average PM2.5")
plt.ylabel("City")

plt.tight_layout()
plt.savefig("graphs/02_top_cities.png")
plt.show()


# Graph 3: Box Plot

plt.figure(figsize=(8, 5))

sns.boxplot(y=df["pollutant_avg"])

plt.title("Outlier Detection")

plt.tight_layout()
plt.savefig("graphs/03_boxplot.png")
plt.show()


# Graph 4: Pie Chart

plt.figure(figsize=(7, 6))

df["pollutant_id"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Pollutant Distribution")
plt.ylabel("")

plt.tight_layout()
plt.savefig("graphs/04_piechart.png")
plt.show()


# Graph 5: Correlation Heatmap

numeric_data = df[
    ["latitude", "longitude",
     "pollutant_min", "pollutant_max", "pollutant_avg"]
]

plt.figure(figsize=(8, 6))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("graphs/05_heatmap.png")
plt.show()


# Graph 6: Scatter Plot

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="pollutant_min",
    y="pollutant_avg",
    hue="pollutant_id"
)

plt.title("Minimum vs Average Pollutant")

plt.tight_layout()
plt.savefig("graphs/06_scatter.png")
plt.show()


# Graph 7: State-wise PM2.5

state_data = (
    pm25.groupby("state")["pollutant_avg"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))

state_data.sort_values().plot(kind="barh")

plt.title("Top 10 States by PM2.5")
plt.xlabel("Average PM2.5")
plt.ylabel("State")

plt.tight_layout()
plt.savefig("graphs/07_state.png")
plt.show()


# 6. OUTLIER ANALYSIS USING IQR

Q1 = df["pollutant_avg"].quantile(0.25)
Q3 = df["pollutant_avg"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["pollutant_avg"] < lower) |
    (df["pollutant_avg"] > upper)
]

print("\nOUTLIER ANALYSIS")
print("Number of outliers:", len(outliers))


# 7. MACHINE LEARNING

print("\nMACHINE LEARNING STARTED")

# Select input features
X = df[
    [
        "state",
        "city",
        "pollutant_id",
        "latitude",
        "longitude",
        "pollutant_min",
        "pollutant_max"
    ]
]

# Select output
y = df["pollutant_avg"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["state", "city", "pollutant_id"],
    dtype=int
)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# 8. FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# 9. CREATE FIVE MODELS

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=5
    )
}


# 10. TRAIN AND TEST MODELS

results = []
trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 3))


# 11. MODEL COMPARISON

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\nMODEL COMPARISON")
print(result_df)

result_df.to_csv(
    "output/model_comparison.csv",
    index=False
)


# 12. BEST MODEL

best_name = result_df.iloc[0]["Model"]

best_model = trained_models[best_name]

print("\nBEST MODEL:", best_name)


# Save model and scaler
joblib.dump(
    {
        "model": best_model,
        "scaler": scaler,
        "columns": X.shape[1]
    },
    "output/best_model.pkl"
)


# 13. MODEL COMPARISON GRAPH

plt.figure(figsize=(9, 5))

sns.barplot(
    data=result_df,
    x="RMSE",
    y="Model"
)

plt.title("Comparison of Five ML Models")
plt.xlabel("RMSE")
plt.ylabel("Model")

plt.tight_layout()
plt.savefig("graphs/08_model_comparison.png")
plt.show()


# 14. ACTUAL VS PREDICTED

best_predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 5))

plt.scatter(y_test, best_predictions, alpha=0.6)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted Values")

plt.tight_layout()
plt.savefig("graphs/09_prediction.png")
plt.show()


# 15. FINAL RESULTS

print("\n================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("================================")

print("Total Records:", len(df))
print("Total ML Models:", len(models))
print("Best Model:", best_name)

print("\nAll files saved in output and graphs folders.")

