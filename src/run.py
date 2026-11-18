## This script runs previously saved MLP models on a test set and prints classification metrics.

import joblib
from sklearn.metrics import classification_report
from mlp import DATA_DIR, load_data, prepare_train_test

num = "3" ## WHICH MODEL VERSION TO RUN

data = load_data(DATA_DIR, max_rows_per_file=100_000, random_state=42)
_, X_test, _, y_test = prepare_train_test(data, test_size=0.2, random_state=42)

model = joblib.load("models/mlp_v" + num + ".joblib")
predictions = model.predict(X_test)

print(classification_report(y_test, predictions, zero_division=0))