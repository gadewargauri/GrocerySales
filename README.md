# GrocerySales
# Grocery Product Sales Prediction Project Overview

This project uses machine learning regression algorithms to predict grocery product sales based on product and outlet information. It also includes an interactive Streamlit dashboard where users can enter details and view predicted sales.

## Objectives

- Predict grocery product sales using historical data.
- Compare different regression algorithms.
- Evaluate model performance using regression metrics.
- Build an interactive dashboard for predictions and visualizations.

## Dataset

**Dataset file:** `K_class_sales_clean.csv`

The dataset contains grocery product and outlet details.

### Input Features

- Item Weight
- Item Fat Content
- Item Visibility
- Item Type
- Item MRP
- Outlet Establishment Year
- Outlet Size
- Outlet Location Type
- Outlet Type

**Target Variable:** `Item_Outlet_Sales`

## Machine Learning Algorithms

The following regression models are trained and compared:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regression
5. Random Forest Regression

The best model is selected based on the highest R² score, with prediction errors also considered.

## Model Evaluation

The models are evaluated using:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Measures how well the model explains variation in sales.

In the current model comparison, Lasso Regression has the lowest MAE, MSE, and RMSE. Its R² score is 0.578 when rounded to three decimal places.

## Dashboard

The interactive dashboard is developed using Streamlit and Plotly.

### Dashboard Features

- Input fields for grocery product details.
- Input fields for outlet information.
- Predicted grocery sales.
- Interactive data visualizations.
- Pie chart, bar chart, donut chart, line chart, and scatter plot.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Project Workflow

1. Load the dataset.
2. Clean and preprocess the data.
3. Separate input features and target variable.
4. Split the data into training and testing sets.
5. Train multiple regression models.
6. Compare model performance.
7. Save the selected model and feature columns.
8. Use the trained model in the Streamlit dashboard.

## Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn streamlit plotly joblib
```

## Run the Dashboard

Navigate to the project folder and run:

```bash
streamlit run app.py
```

Make sure the trained model, feature-column file, dataset (if required by the dashboard), and any other required project files are in the correct locations.

## Expected Outcome

The project predicts grocery product sales from product and outlet details and presents the results through an interactive dashboard.

## Conclusion

This project demonstrates how machine learning and data visualization can be used to analyze grocery sales and support data-driven business decisions.

##

## Future Scope

- Improve prediction accuracy through model tuning.
- Add more product and outlet data.
- Deploy the dashboard online.
- Add inventory-planning and sales-trend analysis features.
