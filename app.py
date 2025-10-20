from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load("iris_model_nb.pkl")
le = joblib.load("label_encoder.pkl")

app = Flask(__name__, template_folder="pages")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari form
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Buat array untuk model
        sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Prediksi
        pred = model.predict(sample)
        species = le.inverse_transform(pred)[0]

        return render_template("index.html", prediction_text=f"Prediksi: {species}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
