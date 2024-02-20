import tensorflow as tf


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def evaluate_model(model, X_test, y_test):
    print(model.summary())
    print(model.evaluate(X_test, y_test))


def make_prediction(input_data):
    model = load_model("BestModel.h5")
    return model.predict(input_data)

