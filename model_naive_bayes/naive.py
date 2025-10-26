import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import joblib

df = pd.read_csv("badminton_dataset.csv")

# Encode target
le_target = LabelEncoder()
df['Play_Badminton'] = le_target.fit_transform(df['Play_Badminton'])
print("\nMapping kelas target:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

le_features = {}
for col in df.columns.drop('Play_Badminton'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_features[col] = le
    print(f"Mapping fitur {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split fitur (X) dan target (y)
X = df.drop('Play_Badminton', axis=1)
y = df['Play_Badminton']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nUkuran data -> Train: {len(X_train)}, Test: {len(X_test)}")
print('X_test:\n', X_test)
print('y_test:\n', y_test)

model = CategoricalNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = round(accuracy_score(y_test, y_pred) * 100, 2)
print("\nAkurasi:", acc, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

cm_viz = ConfusionMatrix(model, classes=le_target.classes_)
cm_viz.score(X_test, y_test)
cm_viz.show()

joblib.dump(model, "badminton_model_cnb.pkl")
joblib.dump(le_target, "label_encoder_target.pkl")
joblib.dump(le_features, "label_encoder_features.pkl")
