from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    predict,
    save_model,
    load_model,
    create_submission,
)

# Chemins vers vos fichiers CSV
train_path = "data/train.csv"
test_path = "data/test.csv"

# ----------------------------
# 1. Tester prepare_data
# ----------------------------
data = prepare_data(train_path, test_path)

print("=== Test prepare_data ===")
print("X_train shape:", data["X_train"].shape)
print("X_test shape:", data["X_test"].shape)
print("y_train shape:", data["y_train"].shape)
print("y_test shape:", data["y_test"].shape)
print("X_final_test shape:", data["X_final_test"].shape)
print()

# ----------------------------
# 2. Tester train_model
# ----------------------------
model = train_model(data["X_train"], data["y_train"])
print("=== Test train_model ===")
print("Type de modèle:", type(model))
print()

# ----------------------------
# 3. Tester evaluate_model
# ----------------------------
eval_results = evaluate_model(model, data["X_test"], data["y_test"])
print("=== Test evaluate_model ===")
print("Accuracy:", eval_results["accuracy"])
print()

# ----------------------------
# 4. Tester predict
# ----------------------------
predictions = predict(model, data["X_final_test"].head())
print("=== Test predict ===")
print(predictions)
print()

# ----------------------------
# 5. Tester save_model et load_model
# ----------------------------
save_model(model, "test_model.pkl")
loaded_model = load_model("test_model.pkl")
loaded_predictions = predict(loaded_model, data["X_test"].head())
print("=== Test save/load model ===")
print(loaded_predictions)
print()

# ----------------------------
# 6. Tester create_submission
# ----------------------------
submission = create_submission(
    predictions, data["test_ids"].head(), "test_submission.csv"
)
print("=== Test create_submission ===")
print(submission.head())
