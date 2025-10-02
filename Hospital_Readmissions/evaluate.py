# evaluate.py
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)

def evaluate_model(name, model, X_test, y_test, results_df=None):
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # === Metrics ===
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    # Print concise report
    print(f"\n{name} Performance:")
    print(f" Accuracy: {acc:.2f}")
    print(f" Precision: {prec:.2f}")
    print(f" Recall: {rec:.2f}")
    print(f" F1-score: {f1:.2f}")
    print(f" AUC: {auc_score:.3f}")

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # === ROC Curve ===
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    # Update results DataFrame
    row = pd.DataFrame({
        "Model": [name],
        "Accuracy": [acc],
        "Precision": [prec],
        "Recall": [rec],
        "F1": [f1],
        "AUC": [auc_score]
    })
    if results_df is not None:
        results_df = pd.concat([results_df, row], ignore_index=True)
        return results_df
    return row

def plot_roc_curves(models, X_test, y_test):
    
    plt.figure(figsize=(6, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.show()

def explain_with_shap(model, X_test, max_display=10):
   
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=max_display)
