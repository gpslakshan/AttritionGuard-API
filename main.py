import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from employee import Employee
from interpreter import explain_prediction
from predictor import make_prediction
from preprocessor import preprocess, prepare_input_data

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


@app.post("/generate-insights")
def generate_insights(employee: Employee):
    # Data pre-processing
    scaler, X_train, X_test, y_train, y_test = preprocess()
    input_data = prepare_input_data(employee)
    input_data_df = pd.DataFrame.from_dict(input_data)
    input_data_scaled_df = scaler.transform(input_data_df)

    # Make Prediction
    prediction_result = make_prediction(input_data_scaled_df)

    # Factor Interpretation
    interpret_factors_plot_url = explain_prediction(X_train, input_data_scaled_df)

    return {
        "employee_details": employee,
        "attrition_probability": np.round(prediction_result.item(), 2),
        "attrition_result": "Yes" if prediction_result > 0.5 else "No",
        "factor_interpretation": interpret_factors_plot_url
    }
