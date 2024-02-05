import shap


def explain_prediction(model, X_train, pred_df_scaled):
    f = lambda x: model.predict(x)
    med = X_train.median().values.reshape((1, X_train.shape[1]))
    explainer = shap.Explainer(f, med)
    pred_shap_values = explainer(pred_df_scaled)
    return pred_shap_values
