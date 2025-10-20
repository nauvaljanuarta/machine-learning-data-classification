from flask import Flask, render_template, request
import joblib
import numpy as np

# Load model dan label encoder
model = joblib.load("badminton_model_cnb.pkl")
le_target = joblib.load("label_encoder_target.pkl")
le_features = joblib.load("label_encoder_features.pkl")  # dict semua fitur

app = Flask(__name__, template_folder="pages")

@app.route("/")
def home():
    # Ambil pilihan fitur dari label encoder untuk dropdown
    options = {col: le.classes_ for col, le in le_features.items()}
    return render_template("index.html", options=options)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari form dan encode sesuai label encoder
        sample = []
        for col in le_features.keys():
            val = request.form[col]
            encoded = le_features[col].transform([val])[0]
            sample.append(encoded)

        sample = np.array([sample])
        pred = model.predict(sample)
        result = le_target.inverse_transform(pred)[0]

        # Tampilkan hasil
        options = {col: le.classes_ for col, le in le_features.items()}
        return render_template("index.html", options=options, prediction_text=f"Prediksi: {result}")

    except Exception as e:
        options = {col: le.classes_ for col, le in le_features.items()}
        return render_template("index.html", options=options, prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
