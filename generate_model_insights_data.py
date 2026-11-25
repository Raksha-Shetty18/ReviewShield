import os
import json
import time
import torch
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

torch.set_num_threads(10)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading datasets...")
val = pd.read_csv('data/reviewshield_xgb_validation.csv', low_memory=False)
sample_val = pd.concat([
    val[val['label'] == 0].sample(500, random_state=42),
    val[val['label'] == 1].sample(500, random_state=42)
]).reset_index(drop=True)

test = pd.read_csv('data/reviewshield_test.csv', low_memory=False)
sample_test = pd.concat([
    test[test['label'] == 0].sample(500, random_state=42),
    test[test['label'] == 1].sample(500, random_state=42)
]).reset_index(drop=True)

# ----------------- 1. XGBoost -----------------
print("Evaluating XGBoost...")
prep = joblib.load('ReviewShield_Models/reviewshield_tuned_preprocessing.joblib')
vec = joblib.load('ReviewShield_Models/reviewshield_tuned_vectorizer.joblib')
xgb_model = xgb.Booster()
xgb_model.load_model('ReviewShield_Models/reviewshield_xgboost_tuned.json')

def prepare_xgb_matrix(df):
    title = df['review_title'].fillna('').astype(str)
    text = df['review_text'].fillna('').astype(str)
    combined = (title + ' ' + text).str.strip()
    X_text = vec.transform(combined)
    num = df[prep['numeric_features']].copy()
    for col in prep['numeric_features']:
        num[col] = pd.to_numeric(num[col], errors='coerce')
    num['prior_average_rating_missing'] = num['prior_average_rating'].isna().astype(np.float32)
    num['days_since_previous_review_missing'] = num['days_since_previous_review'].isna().astype(np.float32)
    num = num.fillna(prep['medians'])
    num = num[prep['numeric_columns']]
    X_num = csr_matrix(num.to_numpy(dtype=np.float32))
    return hstack([X_text, X_num], format='csr', dtype=np.float32)

xgb_val_probs = xgb_model.predict(xgb.DMatrix(prepare_xgb_matrix(sample_val)))
xgb_test_probs = xgb_model.predict(xgb.DMatrix(prepare_xgb_matrix(sample_test)))

# Also full XGBoost test
xgb_test_full_probs = xgb_model.predict(xgb.DMatrix(prepare_xgb_matrix(test)))

val_labels = sample_val['label'].values
test_labels = sample_test['label'].values
test_full_labels = test['label'].values

# Helper for transformers
def predict_transformer(model_path, texts, batch_size=64):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tok(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).numpy()[:, 1]
            all_probs.extend(probs)
    return np.array(all_probs)

val_texts = (sample_val['review_title'].fillna('') + ' ' + sample_val['review_text'].fillna('')).tolist()
test_texts = (sample_test['review_title'].fillna('') + ' ' + sample_test['review_text'].fillna('')).tolist()

print("Evaluating RoBERTa on validation...")
roberta_val_probs = predict_transformer('ReviewShield_Models/roberta_model', val_texts)

print("Evaluating RoBERTa on test sample...")
roberta_test_probs = predict_transformer('ReviewShield_Models/roberta_model', test_texts)

print("Evaluating BERT on validation...")
bert_val_probs = predict_transformer('ReviewShield_Models/reviewshield_bert_model-20261115T102956Z-1-001/reviewshield_bert_model', val_texts)

print("Evaluating DeBERTa on validation...")
deberta_val_probs = predict_transformer('ReviewShield_Models/deberta_model-20261115T102948Z-1-001/deberta_model', val_texts)

# Primary Ensemble on test sample
ens_test_probs = 0.55 * xgb_test_probs + 0.45 * roberta_test_probs
ens_test_preds = (ens_test_probs >= 0.54).astype(int)

# Primary Ensemble on validation sample
ens_val_probs = 0.55 * xgb_val_probs + 0.45 * roberta_val_probs
ens_val_preds = (ens_val_probs >= 0.54).astype(int)

# Helper to compute metrics dict
def get_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds).tolist()
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probs)), 4),
        "confusion_matrix": cm,
        "threshold": threshold
    }

# Curve points helpers
def get_roc_curve_points(y_true, probs, num_points=50):
    fpr, tpr, _ = roc_curve(y_true, probs)
    target_fpr = np.linspace(0.0, 1.0, num_points)
    # Monotonic interpolation for TPR
    interp_tpr = np.interp(target_fpr, fpr, tpr)
    return [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in zip(target_fpr, interp_tpr)]

def get_pr_curve_points(y_true, probs, num_points=50):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    # Reverse to make recall increasing
    r_rev = recall[::-1]
    p_rev = precision[::-1]
    target_recall = np.linspace(0.0, 1.0, num_points)
    interp_p = np.interp(target_recall, r_rev, p_rev)
    return [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in zip(target_recall, interp_p)]

def get_cumulative_gains(y_true, probs, num_points=50):
    order = np.argsort(probs)[::-1]
    sorted_y = y_true[order]
    total_positives = np.sum(sorted_y)
    cum_positives = np.cumsum(sorted_y)
    n = len(y_true)
    
    pop_pcts = np.linspace(0.0, 1.0, num_points)
    gains = []
    for p in pop_pcts:
        idx = int(round(p * (n - 1)))
        g = float(cum_positives[idx]) / total_positives if total_positives > 0 else 0.0
        gains.append({"x": round(float(p * 100), 2), "y": round(float(g * 100), 2)})
    return gains

def get_lift_chart(y_true, probs, deciles=10):
    order = np.argsort(probs)[::-1]
    sorted_y = y_true[order]
    total_positives = np.sum(sorted_y)
    total_samples = len(y_true)
    base_rate = total_positives / total_samples
    
    lift_points = []
    for d in range(1, deciles + 1):
        pct = d / deciles
        cutoff = int(round(pct * total_samples))
        pos_in_top = np.sum(sorted_y[:cutoff])
        precision_at_pct = pos_in_top / cutoff
        lift = precision_at_pct / base_rate if base_rate > 0 else 1.0
        lift_points.append({
            "decile": f"{int(pct * 100)}%",
            "percentile": int(pct * 100),
            "lift": round(float(lift), 3)
        })
    return lift_points

# Structure complete payload
data_payload = {
    "validation_benchmark": {
        "dataset_name": "ReviewShield Validation Split (1,000 stratified benchmark reviews)",
        "sample_size": len(sample_val),
        "genuine_count": int(np.sum(val_labels == 0)),
        "fake_count": int(np.sum(val_labels == 1)),
        "models": {
            "xgboost": {
                "name": "Tuned XGBoost",
                "metrics": get_metrics(val_labels, xgb_val_probs),
                "roc_curve": get_roc_curve_points(val_labels, xgb_val_probs),
                "pr_curve": get_pr_curve_points(val_labels, xgb_val_probs),
                "gains_curve": get_cumulative_gains(val_labels, xgb_val_probs),
                "lift_curve": get_lift_chart(val_labels, xgb_val_probs)
            },
            "roberta": {
                "name": "RoBERTa (Fine-Tuned)",
                "metrics": get_metrics(val_labels, roberta_val_probs),
                "roc_curve": get_roc_curve_points(val_labels, roberta_val_probs),
                "pr_curve": get_pr_curve_points(val_labels, roberta_val_probs),
                "gains_curve": get_cumulative_gains(val_labels, roberta_val_probs),
                "lift_curve": get_lift_chart(val_labels, roberta_val_probs)
            },
            "bert": {
                "name": "BERT (Fine-Tuned)",
                "metrics": get_metrics(val_labels, bert_val_probs),
                "roc_curve": get_roc_curve_points(val_labels, bert_val_probs),
                "pr_curve": get_pr_curve_points(val_labels, bert_val_probs),
                "gains_curve": get_cumulative_gains(val_labels, bert_val_probs),
                "lift_curve": get_lift_chart(val_labels, bert_val_probs)
            },
            "deberta": {
                "name": "DeBERTa (Fine-Tuned)",
                "metrics": get_metrics(val_labels, deberta_val_probs),
                "roc_curve": get_roc_curve_points(val_labels, deberta_val_probs),
                "pr_curve": get_pr_curve_points(val_labels, deberta_val_probs),
                "gains_curve": get_cumulative_gains(val_labels, deberta_val_probs),
                "lift_curve": get_lift_chart(val_labels, deberta_val_probs)
            },
            "ensemble": {
                "name": "Primary Ensemble (55% XGBoost + 45% RoBERTa)",
                "metrics": get_metrics(val_labels, ens_val_probs, threshold=0.54),
                "roc_curve": get_roc_curve_points(val_labels, ens_val_probs),
                "pr_curve": get_pr_curve_points(val_labels, ens_val_probs),
                "gains_curve": get_cumulative_gains(val_labels, ens_val_probs),
                "lift_curve": get_lift_chart(val_labels, ens_val_probs)
            }
        }
    },
    "held_out_test": {
        "dataset_name": "ReviewShield Held-Out Test Set",
        "sample_size": len(sample_test),
        "full_test_size": len(test),
        "ensemble_metrics": get_metrics(test_labels, ens_test_probs, threshold=0.54),
        "xgboost_full_test_metrics": get_metrics(test_full_labels, xgb_test_full_probs, threshold=0.50),
        "roberta_test_metrics": get_metrics(test_labels, roberta_test_probs, threshold=0.50),
        "roc_curve": get_roc_curve_points(test_labels, ens_test_probs),
        "pr_curve": get_pr_curve_points(test_labels, ens_test_probs),
        "gains_curve": get_cumulative_gains(test_labels, ens_test_probs),
        "lift_curve": get_lift_chart(test_labels, ens_test_probs)
    }
}

out_path = 'data/model_insights_data.json'
with open(out_path, 'w') as f:
    json.dump(data_payload, f, indent=2)

print(f"Model insights data saved to {out_path}!")

