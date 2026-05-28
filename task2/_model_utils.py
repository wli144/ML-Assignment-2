import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


OUTPUT_DIR = "submission_results"


def get_class_names(train_meta, unique_labels):
    if "class_name" not in train_meta.columns:
        return [str(l) for l in unique_labels]
    label_map = (train_meta.drop_duplicates("class_id")
                 .set_index("class_id")["class_name"].to_dict())
    return [label_map.get(l, str(l)) for l in unique_labels]


def evaluate(model_name, y_val, y_val_pred, train_meta, best_params=None):
    acc = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy : {acc * 100:.2f}%")
    print(classification_report(y_val, y_val_pred))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unique_labels = sorted(np.unique(y_val))
    class_names   = get_class_names(train_meta, unique_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_val, y_val_pred,
        display_labels=class_names,
        normalize="true",
        xticks_rotation="vertical",
        ax=ax, colorbar=False,
    )
    ax.set_title(f"{model_name} — Validation Confusion Matrix (normalised)")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}_confusion.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {plot_path}")
    return acc


def save_submission(image_ids, y_pred, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    pd.DataFrame({"image_id": image_ids, "class_id": y_pred}).to_csv(path, index=False)
    print(f"  Submission → {path}")
