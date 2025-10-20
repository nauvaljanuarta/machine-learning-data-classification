import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("IRIS.csv")

print("Data Awal:")
print(df.head())
print("\nDistribusi kelas sebelum preprocessing:")
print(df['species'].value_counts())

# prreprocessing
# membuat class target menjadi angka
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
# Mapping kelas
print("\nMapping kelas:", dict(zip(le.classes_, le.transform(le.classes_))))

# pisahkan fitur (X) dan target (y)
X = df.drop('species', axis=1)
y = df['species']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# balancing data mengunakan smote
print("\nDistribusi awal y:")
print(y.value_counts())
X_res, y_res = (X, y)

print(pd.Series(y_res).value_counts())

# pambagian data untuk testing dan training
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

print(f"\nUkuran data -> Train: {len(X_train)}, Test: {len(X_test)}")

print('ini adalah x test :', X_test)
print('ini adalah y test :', y_test)
# clf merupakan model dari suatu data yang nnti akan di training
clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
clf.fit(X_train, y_train)

# menghasilkan y_pred hasil prediksi dari model clf pada data x_test 
y_pred = clf.predict(X_test)


acc = round(accuracy_score(y_test, y_pred)*100, 2)

print("\n akurasinya : ", acc, "%")
print("\n klasifikasi :")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ini decision treenya
plt.figure(figsize=(12,8))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=le.classes_,
    filled=True,
    rounded=True,
    proportion=True,       
    impurity=False,        
    label='root',         
)
plt.title("decision tree pada data bunga iris IRIS.csv")
plt.show()

# export model & scaler
# joblib.dump(clf, "iris_model.pkl")
# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(le, "label_encoder.pkl")
print("Model, scaler, dan encoder berhasil disimpan.")


