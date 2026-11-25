import os
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ReviewShield_Models")

FEATURE_DESCRIPTIONS = {
    "prior_distinct_products": {
        "category": "Account History & Velocity",
        "description": "Number of distinct product ASINs previously reviewed by account. Single-product throwaway accounts exhibit elevated deceptive propensity."
    },
    "prior_review_count": {
        "category": "Account History & Velocity",
        "description": "Lifetime submission volume for reviewer profile. Isolated 1-review accounts frequently correlate with commercial review brokers."
    },
    "prior_30day_review_count": {
        "category": "Temporal Burst Pacing",
        "description": "Submission velocity within rolling 30-day window. Burst spikes indicate coordinated vendor astroturfing campaigns."
    },
    "prior_7day_review_count": {
        "category": "Temporal Burst Pacing",
        "description": "Rapid review burst submissions within a single week. High concentration flags automated or paid batch reviews."
    },
    "days_since_previous_review": {
        "category": "Temporal Burst Pacing",
        "description": "Temporal gap between consecutive reviews. Extremely tight intervals (<24 hours) signify scripted mass posting."
    },
    "review_rating": {
        "category": "Rating Polarization",
        "description": "Customer star rating (1-5). Deceptive submissions cluster at extreme poles: 5 stars (promotional puffery) or 1 star (smear campaigns)."
    },
    "prior_average_rating": {
        "category": "Rating Polarization",
        "description": "Historical star mean across all account reviews. Accounts with invariant 5.0 or 1.0 averages lack organic consumer variance."
    },
    "text_word_count": {
        "category": "Textual Stylometry",
        "description": "Total length in tokens. Very short generic praises or bloated keyword-stuffed text exhibit abnormal stylometric signals."
    },
    "text_character_count": {
        "category": "Textual Stylometry",
        "description": "Raw string length of review headline and body. Captures superficial brevity patterns common in syndicated fraud."
    },
    "text_uppercase_ratio": {
        "category": "Textual Stylometry",
        "description": "Ratio of capitalized characters. Excessive capitalization reflects artificial emphasis and urgency tactics."
    },
    "text_repeated_word_count": {
        "category": "Textual Stylometry",
        "description": "Frequency of redundant lexical tokens within review body. Common hallmark of paraphrased or spun AI templates."
    },
    "text_average_word_length": {
        "category": "Textual Stylometry",
        "description": "Mean characters per token. Indicates lexical complexity and vocabulary sophistication vs primitive repetitive syntax."
    },
    "text_exclamation_count": {
        "category": "Textual Stylometry",
        "description": "Number of exclamation marks. Pervasive exclamation density is significantly overrepresented in compensated promotional reviews."
    },
    "text_question_count": {
        "category": "Textual Stylometry",
        "description": "Rhetorical question frequency. Frequently utilized in marketing-style copy ('Need something durable? Look no further!')."
    }
}

_global_cache = None
_samples_cache = None

def get_global_shap_importance(limit: int = 30):
    global _global_cache
    if _global_cache is not None:
        return _global_cache[:limit]

    csv_path = os.path.join(MODELS_DIR, "reviewshield_shap_global_importance.csv")
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path)
    results = []
    for rank, row in df.head(50).iterrows():
        feat = str(row["feature"])
        score = float(row["mean_absolute_shap"])
        meta = FEATURE_DESCRIPTIONS.get(feat, {
            "category": "Lexical N-Gram Token" if not feat.startswith("prior_") and not feat.startswith("text_") else "Engineered Signal",
            "description": f"TF-IDF lexical presence of token '{feat}' across fraudulent vs genuine training corpus."
        })
        results.append({
            "rank": int(rank) + 1,
            "feature": feat,
            "mean_absolute_shap": round(score, 4),
            "category": meta["category"],
            "description": meta["description"]
        })

    _global_cache = results
    return results[:limit]

def get_shap_samples(limit: int = 20):
    global _samples_cache
    if _samples_cache is not None:
        return _samples_cache[:limit]

    csv_path = os.path.join(MODELS_DIR, "reviewshield_shap_individual_explanations.csv")
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path)
    grouped = {}
    for _, row in df.iterrows():
        sid = int(row["review_sample"])
        if sid not in grouped:
            if len(grouped) >= limit:
                break
            grouped[sid] = {
                "sample_id": sid,
                "predicted_class": str(row["predicted_class"]),
                "fake_probability": round(float(row["fake_probability"]), 4),
                "features": []
            }
        
        shap_val = float(row["shap_value"])
        feat_name = str(row["feature"])
        meta = FEATURE_DESCRIPTIONS.get(feat_name, {
            "category": "Lexical N-Gram",
            "description": f"Influence of lexical token '{feat_name}'."
        })

        grouped[sid]["features"].append({
            "feature": feat_name,
            "feature_value": float(row["feature_value"]) if pd.notnull(row["feature_value"]) else 0.0,
            "shap_value": round(shap_val, 4),
            "influence": str(row["influence"]),
            "category": meta["category"],
            "description": meta["description"]
        })

    samples = list(grouped.values())
    for s in samples:
        s["positive_contributions"] = [f for f in s["features"] if f["shap_value"] > 0]
        s["negative_contributions"] = [f for f in s["features"] if f["shap_value"] < 0]
        s["positive_contributions"].sort(key=lambda x: x["shap_value"], reverse=True)
        s["negative_contributions"].sort(key=lambda x: x["shap_value"])

    _samples_cache = samples
    return samples

def get_shap_sample_by_id(sample_id: int):
    samples = get_shap_samples(limit=100)
    for s in samples:
        if s["sample_id"] == sample_id:
            return s
    return None

if __name__ == "__main__":
    glob = get_global_shap_importance(5)
    print("Top 5 global:", glob)
    samp = get_shap_samples(2)
    print(f"Loaded {len(samp)} samples. Sample 0 features count: {len(samp[0]['features'])}")
