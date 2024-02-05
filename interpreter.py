import shap


def explain_prediction(model, X_train, input_data_scaled_df):
    f = lambda x: model.predict(x)
    med = X_train.median().values.reshape((1, X_train.shape[1]))
    explainer = shap.Explainer(f, med)
    pred_shap_values = explainer(input_data_scaled_df)
    return pred_shap_values
