# main.py
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess
from train_models import train_models
from evaluate import evaluate_model, plot_roc_curves, explain_with_shap

def main():
    # === Step 1: Load & preprocess ===
    filepath = "diabetic_data.csv"
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(filepath)

    # Dataset summary (Deliverable 4 evidence)
    print("=== Dataset Summary After Preprocessing ===")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Number of features (pre-transformation): {X_train.shape[1]}")
    print("Class distribution in training set (0 = No Readmission, 1 = Readmission <30 days):")
    print(y_train.value_counts(normalize=True))
    print("=" * 60)

    # === Step 2: Train models ===
    results = train_models(X_train, X_test, y_train, y_test, preprocessor)

    # === Step 3: Evaluate models ===
    results_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])
    for name, grid in results.items():
        results_df = evaluate_model(name, grid.best_estimator_, X_test, y_test, results_df)

    # Display comparison table
    print("\n=== Final Model Performance Summary ===")
    print(results_df.round(3))

    # === Step 4: Compare ROC Curves ===
    plot_roc_curves({name: grid.best_estimator_ for name, grid in results.items()}, X_test, y_test)

    # === Step 5: SHAP analysis for XGBoost ===
    print("\n=== SHAP Feature Importance for XGBoost ===")
    explain_with_shap(results["XGBoost"].best_estimator_["clf"], X_test)

if __name__ == "__main__":
    main()
