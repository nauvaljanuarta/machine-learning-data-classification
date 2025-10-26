from flask import Flask, render_template, request, url_for
import joblib
import pandas as pd
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import warnings
from yellowbrick.exceptions import YellowbrickWarning

warnings.filterwarnings("ignore", category=YellowbrickWarning)
app = Flask(__name__, template_folder="pages")

models = {
    'linear': joblib.load("models/loan_approval_model_svm_linear.pkl"),
    'poly': joblib.load("models/loan_approval_model_svm_poly.pkl"),
    'rbf': joblib.load("models/loan_approval_model_svm_rbf.pkl")
}

le_target = joblib.load("encoders/label_encoder_target.pkl")
le_features = joblib.load("encoders/label_encoder_features.pkl")
scaler = joblib.load("encoders/scaler_features.pkl")

df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()
target_col = 'loan_status'

cat_features = ['no_of_dependents','education','self_employed']
num_features = ['income_annum','loan_amount','loan_term','cibil_score',
                'residential_assets_value','commercial_assets_value',
                'luxury_assets_value','bank_asset_value']

for col in cat_features:
    df[col] = le_features[col].transform(df[col])
df[num_features] = scaler.transform(df[num_features])
X = df.drop(['loan_id', target_col], axis=1)
y = le_target.transform(df[target_col])

cm_folder = "static/images"
os.makedirs(cm_folder, exist_ok=True)

eval_results = {}
images = {}
for kernel, model in models.items():
    y_pred = model.predict(X)
    acc = round(accuracy_score(y, y_pred)*100, 2)
    report = classification_report(y, y_pred, target_names=le_target.classes_, output_dict=True)
    
    # Confusion Matrix
    cm_viz = ConfusionMatrix(model, classes=le_target.classes_, title=f"Confusion Matrix - {kernel.capitalize()}")
    cm_viz.score(X, y)
    cm_path = os.path.join(cm_folder, f"cm_{kernel}.png")
    cm_viz.fig.savefig(cm_path)
    cm_viz.finalize()
    
    eval_results[kernel] = {"accuracy": acc, "report": report}
    
    images[kernel] = f"/static/images/cm_{kernel}.png"  # gunakan path relatif


@app.route("/", methods=['GET','POST'])
def index():
    user_prediction = None
    selected_kernel = None

    if request.method == 'POST':
        selected_kernel = request.form['kernel']
        model = models[selected_kernel]

        # --- Ambil input user ---
        user_input = {}
        for col in cat_features + num_features:
            val = request.form[col]
            if col in cat_features:
                user_input[col] = le_features[col].transform([val])[0]
            else:
                user_input[col] = float(val)

        user_input_array = np.array([user_input[col] for col in cat_features + num_features]).reshape(1,-1)
        user_input_array[:, len(cat_features):] = scaler.transform(user_input_array[:, len(cat_features):])

        # --- Prediksi ---
        pred = model.predict(user_input_array)[0]
        user_prediction = le_target.inverse_transform([pred])[0]

    return render_template("index.html",
                           kernels=models.keys(),
                           selected_kernel=selected_kernel,
                           user_prediction=user_prediction,
                           cat_features=cat_features,
                           num_features=num_features,
                           le_features=le_features,
                           images=images,
                           eval_results=eval_results)

if __name__ == "__main__":
    app.run(debug=True)
