from fastapi import FastAPI
from preprocessor import load_data, rename_columns, encode_features, create_train_test_split, scale_features
from predictor import load_model, make_prediction
from employee import Employee

app = FastAPI()


@app.post("/predict-attrition")
def predict_attrition(employee: Employee):
    # Loading the ANN
    model = load_model("BestModel.h5")

    # Get the scaler used for feature scaling
    scaler = get_scaler()

    # Prediction
    input_data = prepare_input_data(employee)
    prediction_result = make_prediction(model, input_data, scaler)
    print(prediction_result)

    return {
        "attrition_probability": f"{prediction_result}",
        "attrition": "Yes" if prediction_result > 0.5 else "No"
    }


def prepare_input_data(employee):
    input_data = {
        "satisfaction_level": [employee.satisfaction_level],
        "last_evaluation": [employee.last_evaluation],
        "number_project": [employee.number_project],
        "average_montly_hours": [employee.average_montly_hours],
        "time_spend_company": [employee.time_spend_company],
        "Work_accident": [employee.Work_accident],
        "promotion_last_5years": [employee.promotion_last_5years],
        "department": [employee.department],
        "salary_level": [employee.salary_level]
    }
    return input_data


def get_scaler():
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
    return scaler
