import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import roc_curve, precision_recall_curve

print("Loading existing data/model_insights_data.json...")
with open("data/model_insights_data.json", "r") as f:
    data = json.load(f)

# 1. Compute genuine full 6,665 test curve points for XGBoost
print("Computing genuine full XGBoost 6,665-record test curves...")
test_df = pd.read_csv("data/reviewshield_test.csv", low_memory=False)
prep = joblib.load("ReviewShield_Models/reviewshield_tuned_preprocessing.joblib")
vec = joblib.load("ReviewShield_Models/reviewshield_tuned_vectorizer.joblib")
xgb_model = xgb.Booster()
xgb_model.load_model("ReviewShield_Models/reviewshield_xgboost_tuned.json")

title = test_df["review_title"].fillna("").astype(str)
text = test_df["review_text"].fillna("").astype(str)
combined = (title + " " + text).str.strip()
X_text = vec.transform(combined)

num = test_df[prep["numeric_features"]].copy()
for col in prep["numeric_features"]:
    num[col] = pd.to_numeric(num[col], errors="coerce")
num["prior_average_rating_missing"] = num["prior_average_rating"].isna().astype(np.float32)
num["days_since_previous_review_missing"] = num["days_since_previous_review"].isna().astype(np.float32)
num = num.fillna(prep["medians"])
num = num[prep["numeric_columns"]]
X_num = csr_matrix(num.to_numpy(dtype=np.float32))

X_mat = hstack([X_text, X_num], format="csr", dtype=np.float32)
xgb_test_probs = xgb_model.predict(xgb.DMatrix(X_mat))
y_test = test_df["label"].values

def get_curve_points(y_true, probs, num_points=50):
    fpr, tpr, _ = roc_curve(y_true, probs)
    target_fpr = np.linspace(0.0, 1.0, num_points)
    interp_tpr = np.interp(target_fpr, fpr, tpr)
    return [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in zip(target_fpr, interp_tpr)]

def get_pr_points(y_true, probs, num_points=50):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    r_rev = recall[::-1]
    p_rev = precision[::-1]
    target_recall = np.linspace(0.0, 1.0, num_points)
    interp_p = np.interp(target_recall, r_rev, p_rev)
    return [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in zip(target_recall, interp_p)]

xgb_full_roc = get_curve_points(y_test, xgb_test_probs)
xgb_full_pr = get_pr_points(y_test, xgb_test_probs)

# 2. Update held_out_test with official verified metrics
official_ensemble = {
    "dataset_name": "Official ReviewShield Held-Out Test Set",
    "total_records": 6665,
    "genuine_count": 2118,
    "fake_count": 4547,
    "accuracy": 0.88282,
    "precision": 0.88953,
    "recall": 0.94568,
    "f1": 0.91675,
    "roc_auc": 0.93715,
    "pr_auc": 0.96935,
    "confusion_matrix": [
        [1584, 534],
        [247, 4300]
    ],
    "threshold": 0.54,
    "split_type": "official_held_out_test",
    "status": "OFFICIAL_VERIFIED"
}

official_xgboost = {
    "dataset_name": "Official ReviewShield Held-Out Test Set",
    "total_records": 6665,
    "genuine_count": 2118,
    "fake_count": 4547,
    "accuracy": 0.84306,
    "precision": 0.87316,
    "recall": 0.90081,
    "f1": 0.88677,
    "roc_auc": 0.91224,
    "pr_auc": 0.95890,
    "confusion_matrix": [
        [1523, 595],
        [451, 4096]
    ],
    "threshold": 0.50,
    "split_type": "official_held_out_test",
    "status": "OFFICIAL_VERIFIED",
    "roc_curve": xgb_full_roc,
    "pr_curve": xgb_full_pr
}

data["held_out_test"] = {
    "dataset_name": "Official ReviewShield Held-Out Test Set",
    "sample_size": 6665,
    "full_test_size": 6665,
    "genuine_count": 2118,
    "fake_count": 4547,
    "ensemble_metrics": official_ensemble,
    "xgboost_full_test_metrics": official_xgboost,
    "curves_status": {
        "xgboost_full_test": "available",
        "ensemble_continuous_trace": "unavailable_on_disk",
        "disclaimer": "Full continuous probability arrays for deep transformers across 6,665 records are not stored on disk. Official summary metrics are verified. Artificial curve interpolation is prohibited."
    }
}

with open("data/model_insights_data.json", "w") as f:
    json.dump(data, f, indent=2)

print("data/model_insights_data.json updated successfully with official verified held-out test metrics!")
