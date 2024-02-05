from fastapi import FastAPI
from preprocessor import load_data, rename_columns, encode_features, create_train_test_split, scale_features
from predictor import load_model, evaluate_model, make_prediction

app = FastAPI()


@app.get("/predict-attrition")
async def predict_attrition():
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

    # Loading and Evaluating Model
    model = load_model("BestModel.h5")
    evaluate_model(model, X_test, y_test)  # Only for the test purposes

    # Prediction
    input_data = {
        "satisfaction_level": [0.6],
        "last_evaluation": [0.52],
        "number_project": [3],
        "average_montly_hours": [240],
        "time_spend_company": [3],
        "Work_accident": 0,
        "promotion_last_5years": [0],
        "department": [3],
        "salary_level": [0]
    }

    prediction_result = make_prediction(model, input_data, scaler)
    print(prediction_result)

    return {"prediction": f"{prediction_result}"}
