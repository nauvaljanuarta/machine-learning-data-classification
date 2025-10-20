import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from yellowbrick.classifier import ConfusionMatrix
import joblib

# Load data
df = pd.read_csv("IRIS.csv")

# Encode label
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Split fitur (X) dan target (y)
X = df.drop('species', axis=1)
y = df['species']

print("\nMapping kelas:", dict(zip(le.classes_, le.transform(le.classes_))))


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nUkuran data -> Train: {len(X_train)}, Test: {len(X_test)}")

print('ini adalah x test :', "\n", X_test)
print('ini adalah y test :',  "\n", y_test)

# pakai gaussian karena data iris sederhana dan kecil
model = GaussianNB()
model.fit(X_train, y_train)

# prediksi
y_pred = model.predict(X_test)

# hasil analisis
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
print("Akurasi:", acc, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
cm_viz = ConfusionMatrix(model, classes=y.unique())
cm_viz.score(X_test, y_test)
cm_viz.show()

joblib.dump(model, "iris_model_nb.pkl")
joblib.dump(le, "label_encoder.pkl")