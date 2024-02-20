import pandas as pd
import numpy as np
from fastapi import FastAPI
from preprocessor import preprocess, prepare_input_data
from interpreter import explain_prediction
from predictor import load_model, make_prediction
from employee import Employee
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

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

# Load .env file
load_dotenv()


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
        "attrition_probability": np.round(prediction_result.item(), 2),
        "attrition_result": "Yes" if prediction_result > 0.5 else "No"
    }


@app.post("/interpret-factors", tags=["Interpretation"])
def interpret_factors(employee: Employee):
    model = load_model("BestModel.h5")
    scaler, X_train, X_test, y_train, y_test = preprocess()
    input_data = prepare_input_data(employee)
    input_data_df = pd.DataFrame.from_dict(input_data)
    input_data_scaled_df = scaler.transform(input_data_df)
    interpret_factors_plot_url = explain_prediction(model, X_train, input_data_scaled_df)
    return {"interpret_factors_plot_url": interpret_factors_plot_url}