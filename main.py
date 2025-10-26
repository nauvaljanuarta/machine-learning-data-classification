

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import joblib


df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()
print("Dataset Head:\n", df.head())

target_col = 'loan_status'  # ini target nya loan_staus yang di prediksi

print(target_col)

le_target = LabelEncoder()
df[target_col] = le_target.fit_transform(df[target_col])
print("\nMapping kelas target:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

# labeling categorical feature 
cat_features = ['no_of_dependents', 'education', 'self_employed']
le_features = {}

for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_features[col] = le
    print(f"Mapping fitur {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# scaling numerical feature 
num_features = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                'residential_assets_value', 'commercial_assets_value',
                'luxury_assets_value', 'bank_asset_value']

scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])


# # split data untuk testing dan training
X = df.drop(['loan_id', target_col], axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nUkuran data -> Train: {len(X_train)}, Test: {len(X_test)}")
print('X_test:\n', X_test.head())
print('y_test:\n', y_test.head())

# training models dengan 3 kernels 
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    print(f"\n====================\nTraining SVM with {kernel} kernel\n====================")
    
    model = SVC(kernel=kernel, probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # prediciton
    y_pred = model.predict(X_test)
    
    # akurasi
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(f"Akurasi: {acc} %")
    
    # report klasifikasi
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    
    # confussion matrix 
    cm_viz = ConfusionMatrix(model, classes=le_target.classes_, title=f"Confusion Matrix - Kernel {kernel.capitalize()}", size=(600, 400))
    
    cm_viz.score(X_test, y_test)
    cm_path = f"static/images/cm_{kernel}.png" 
    cm_viz.fig.tight_layout()  # pastikan layout rapih
    cm_viz.fig.savefig(cm_path, bbox_inches='tight')  # simpan dengan bbox_inches
    cm_viz.finalize()
    cm_viz.show()
    
    # simpan model
#     joblib.dump(model, f"loan_approval_model_svm_{kernel}.pkl")

# joblib.dump(le_target, "label_encoder_target.pkl")
# joblib.dump(le_features, "label_encoder_features.pkl")
# joblib.dump(scaler, "scaler_features.pkl")
