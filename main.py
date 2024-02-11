import pandas as pd
import numpy as np
from fastapi import FastAPI
from preprocessor import load_data, rename_columns, encode_features, create_train_test_split, scale_features
from interpreter import explain_prediction
from predictor import load_model, make_prediction
from employee import Employee
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AttritionGuard-API",
    description="AttritionGuard-API is a powerful tool designed to help organizations proactively manage employee "
                "attrition. With two intuitive endpoints, this API enables users to predict attrition and gain "
                "valuable insights into the contributing factors.",
)

origins = [
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict-attrition", tags=["Prediction"])
def predict_attrition(employee: Employee):
    # Loading the ANN
    model = load_model("BestModel.h5")

    # Get the scaler used for feature scaling after preprocessing
    scaler, X_train, X_test, y_train, y_test = preprocess()

    # Prediction
    input_data = prepare_input_data(employee)
    prediction_result = make_prediction(model, input_data, scaler)
    print(prediction_result)

    return {
        "employee_details": employee,
        "attrition_probability": f"{np.round(prediction_result.item(), 2)}",
        "attrition_result": "Yes" if prediction_result > 0.5 else "No"
    }


@app.post("/interpret-factors", tags=["Interpretation"])
def interpret_factors(employee: Employee):
    model = load_model("BestModel.h5")
    scaler, X_train, X_test, y_train, y_test = preprocess()
    input_data = prepare_input_data(employee)
    input_data_df = pd.DataFrame.from_dict(input_data)
    input_data_scaled_df = scaler.transform(input_data_df)
    shap_values = explain_prediction(model, X_train, input_data_scaled_df)
    shap_importance_dict = get_feature_importance(shap_values)
    print(shap_importance_dict)
    return {"feature_importance": shap_importance_dict}


def get_feature_importance(shap_values):
    feature_names = shap_values.feature_names
    pred_shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    vals = np.abs(pred_shap_df.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['feature_name', 'feature_importance_val'])
    shap_importance.sort_values(by=['feature_importance_val'], ascending=False, inplace=True)
    shap_importance_dict = shap_importance.to_dict(orient='records')
    return shap_importance_dict


def prepare_input_data(employee):
    input_data = {
        "satisfaction_level": [round(employee.job_satisfaction / 100, 2)],
        "last_evaluation": [round(employee.last_evaluation / 100, 2)],
        "number_project": [employee.no_of_projects],
        "average_montly_hours": [employee.avg_monthly_hours],
        "time_spend_company": [employee.time_spend_company],
        "Work_accident": [employee.work_accidents],
        "promotion_last_5years": [employee.promotions],
        "department": [employee.department],
        "salary_level": [employee.salary_level]
    }
    return input_data


def preprocess():
    # Dataset Loading and Preprocessing
    df = load_data("HR_comma_sep.csv")
    df = rename_columns(df)
    df = encode_features(df)
    # Creating dependent variable and independent variables
    X = df.drop("left", axis=1)
    y = df["left"]
    # Train-Test Split and Feature Scaling
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    return scaler, X_train, X_test, y_train, y_test
