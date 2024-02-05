import tensorflow as tf
import pandas as pd


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def evaluate_model(model, X_test, y_test):
    print(model.summary())
    print(model.evaluate(X_test, y_test))


def make_prediction(model, input_data, scaler):
    input_data = pd.DataFrame.from_dict(input_data)
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)

