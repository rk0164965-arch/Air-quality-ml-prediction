# Air Quality Analysis and Prediction

A Python project that cleans, analyzes, and visualizes air quality data from Indian monitoring stations, then predicts the average pollutant level using five machine learning models.

## What the project does

- Cleans the data (duplicates, invalid values, missing values)
- Exploratory data analysis with 7 graphs
- Outlier detection using the IQR method
- Trains and compares 5 models:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - KNN
- Evaluates with MAE, RMSE and R2 score and saves the best model

## Technologies used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## How to run

1. Install the libraries:
   pip install -r requirements.txt
2. Put the dataset CSV in the same folder as the script.
3. Run the script:
   python "project 234 air qualtity project.py"

## Output

- cleaned_data.csv and model_comparison.csv in the `output` folder
- Charts in the `graphs` folder
