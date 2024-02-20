import os

import cloudinary
import cloudinary.api
import cloudinary.uploader
import matplotlib
import matplotlib.pyplot as plt
import shap

from predictor import load_model

matplotlib.use('Agg')


def explain_prediction(X_train, input_data_scaled_df):
    model = load_model("BestModel.h5")
    f = lambda x: model.predict(x)
    med = X_train.median().values.reshape((1, X_train.shape[1]))
    explainer = shap.Explainer(f, med)
    pred_shap_values = explainer(input_data_scaled_df)
    shap.plots.bar(pred_shap_values[0], show=False)
    plt.savefig("shap_summary.png", dpi=700, bbox_inches='tight')
    plt.close()  # Close the current figure

    # Uploading the SHAP bar plot to Cloudinary
    cloudinary.config(
        cloud_name=os.getenv("CLOUD_NAME"),
        api_key=os.getenv("API_KEY"),
        api_secret=os.getenv("API_SECRET")
    )
    uploadResult = cloudinary.uploader.upload("shap_summary.png")
    uploadImgUrl = uploadResult.get("url")
    print(uploadImgUrl)
    os.remove("shap_summary.png")
    return uploadImgUrl
