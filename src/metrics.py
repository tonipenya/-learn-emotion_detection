from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calc_metrics(y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics = {
        "macro": {
            "accuracy": accuracy,
            "precision": precision_macro,
            "recall": recall_macro,
            "f1": f1_macro,
        }
    } | {
        class_name: {
            "precision": precision_class[i],
            "recall": recall_class[i],
            "f1": f1_class[i],
        }
        for i, class_name in enumerate(class_names)
    }

    return metrics
