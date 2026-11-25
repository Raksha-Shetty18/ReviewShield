import os
import re
import sqlite3
import joblib
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ReviewShield_Models")
DATA_DIR = os.path.join(BASE_DIR, "data")
REVIEWERS_DB = os.path.join(DATA_DIR, "reviewers.db")

torch.set_num_threads(8)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class InferenceEngine:
    _instance = None

    def __init__(self):
        print("Initializing ReviewShield Inference Engine (Lazy Loading Mode)...")
        self.load_xgboost_pipeline()
        self.roberta_tok = None
        self.roberta_model = None
        self.bert_tok = None
        self.bert_model = None
        self.deberta_tok = None
        self.deberta_model = None
        print("Inference Engine initialized: XGBoost ready, transformers configured for safe lazy loading.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = InferenceEngine()
        return cls._instance

    def load_xgboost_pipeline(self):
        print("Loading XGBoost, Vectorizer, and Preprocessing...")
        xgb_path = os.path.join(MODELS_DIR, "reviewshield_xgboost_tuned.json")
        vec_path = os.path.join(MODELS_DIR, "reviewshield_tuned_vectorizer.joblib")
        prep_path = os.path.join(MODELS_DIR, "reviewshield_tuned_preprocessing.joblib")

        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(xgb_path)

        self.vectorizer = joblib.load(vec_path)
        self.preprocessing = joblib.load(prep_path)
        self.numeric_features = self.preprocessing["numeric_features"]
        self.numeric_columns = self.preprocessing["numeric_columns"]
        self.medians = self.preprocessing["medians"]

        # Invert feature names for TF-IDF to map SHAP indices back to words
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

    def ensure_roberta(self):
        if self.roberta_model is None or self.roberta_tok is None:
            print("Lazy loading RoBERTa model and tokenizer (Primary Ensemble Component)...")
            roberta_path = os.path.join(MODELS_DIR, "roberta_model")
            self.roberta_tok = AutoTokenizer.from_pretrained(roberta_path, local_files_only=True)
            self.roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path, local_files_only=True)
            self.roberta_model.eval()
            print("RoBERTa loaded successfully!")

    def ensure_bert(self):
        if self.bert_model is None or self.bert_tok is None:
            print("Lazy loading BERT model and tokenizer (Individual Model)...")
            bert_path = os.path.join(MODELS_DIR, "reviewshield_bert_model-20261115T102956Z-1-001", "reviewshield_bert_model")
            self.bert_tok = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path, local_files_only=True)
            self.bert_model.eval()
            print("BERT loaded successfully!")

    def ensure_deberta(self):
        if self.deberta_model is None or self.deberta_tok is None:
            print("Lazy loading DeBERTa model and tokenizer (Individual Model)...")
            deberta_path = os.path.join(MODELS_DIR, "deberta_model-20261115T102948Z-1-001", "deberta_model")
            self.deberta_tok = AutoTokenizer.from_pretrained(deberta_path, local_files_only=True)
            self.deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_path, local_files_only=True)
            self.deberta_model.eval()
            print("DeBERTa loaded successfully!")

    def lookup_reviewer_history(self, reviewer_id: str):
        if not reviewer_id or not os.path.exists(REVIEWERS_DB):
            return None
        try:
            conn = sqlite3.connect(REVIEWERS_DB)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reviewers WHERE reviewer_id = ? LIMIT 1", (reviewer_id.strip(),))
            row = cursor.fetchone()
            conn.close()
            if row:
                return dict(row)
        except Exception as e:
            print("Reviewer lookup error:", e)
        return None

    def extract_textual_features(self, text: str):
        text = str(text or "")
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        char_count = len(text)
        avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

        excl_count = text.count("!")
        ques_count = text.count("?")

        letters = [c for c in text if c.isalpha()]
        upper_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0

        word_freq = {}
        for w in words:
            wl = w.lower()
            word_freq[wl] = word_freq.get(wl, 0) + 1
        repeated_word_count = sum(c - 1 for c in word_freq.values() if c > 1)

        return {
            "text_word_count": word_count,
            "text_character_count": char_count,
            "text_average_word_length": round(avg_word_len, 2),
            "text_exclamation_count": excl_count,
            "text_question_count": ques_count,
            "text_uppercase_ratio": round(upper_ratio, 4),
            "text_repeated_word_count": repeated_word_count
        }

    def predict_single(self, review_title: str, review_text: str, rating: float = 5.0,
                       review_date: str = None, reviewer_id: str = None, product_id: str = None,
                       include_all_models: bool = False, include_bert: bool = False, include_deberta: bool = False):
        combined_text = (str(review_title or "") + " " + str(review_text or "")).strip()
        if not combined_text:
            combined_text = "No review content provided"

        # 1. Textual features
        text_feats = self.extract_textual_features(combined_text)

        # 2. Reviewer history lookup
        rev_history = self.lookup_reviewer_history(reviewer_id)
        has_history = rev_history is not None

        # Build numerical feature row
        num_dict = {col: np.nan for col in self.numeric_features}
        num_dict["review_rating"] = float(rating)
        num_dict["text_word_count"] = text_feats["text_word_count"]
        num_dict["text_character_count"] = text_feats["text_character_count"]
        num_dict["text_average_word_length"] = text_feats["text_average_word_length"]
        num_dict["text_exclamation_count"] = text_feats["text_exclamation_count"]
        num_dict["text_question_count"] = text_feats["text_question_count"]
        num_dict["text_uppercase_ratio"] = text_feats["text_uppercase_ratio"]
        num_dict["text_repeated_word_count"] = text_feats["text_repeated_word_count"]

        if has_history:
            num_dict["prior_review_count"] = float(rev_history.get("reviewer_total_reviews") or 1)
            num_dict["prior_distinct_products"] = float(rev_history.get("reviewer_distinct_products") or 1)
            num_dict["prior_average_rating"] = float(rev_history.get("reviewer_average_rating") or rating)
            raw_gap = rev_history.get("reviewer_average_days_between_reviews")
            num_dict["days_since_previous_review"] = float(raw_gap) if (raw_gap is not None and not pd.isna(raw_gap)) else np.nan
            # Estimated 7-day and 30-day velocity from average gap
            avg_gap = float(raw_gap) if (raw_gap is not None and not pd.isna(raw_gap)) else 999.0
            if avg_gap > 0 and avg_gap < 999.0:
                num_dict["prior_7day_review_count"] = min(5, int(7.0 / avg_gap))
                num_dict["prior_30day_review_count"] = min(15, int(30.0 / avg_gap))
            else:
                num_dict["prior_7day_review_count"] = 0
                num_dict["prior_30day_review_count"] = 0
        else:
            # Unobserved reviewer history: do NOT invent data
            num_dict["prior_review_count"] = 0.0
            num_dict["prior_distinct_products"] = 0.0
            num_dict["prior_average_rating"] = np.nan
            num_dict["days_since_previous_review"] = np.nan
            num_dict["prior_7day_review_count"] = 0.0
            num_dict["prior_30day_review_count"] = 0.0

        # Construct dataframe for XGBoost
        num_df = pd.DataFrame([num_dict])
        num_df["prior_average_rating_missing"] = num_df["prior_average_rating"].isna().astype(np.float32)
        num_df["days_since_previous_review_missing"] = num_df["days_since_previous_review"].isna().astype(np.float32)

        # Impute with training medians
        num_df = num_df.fillna(self.medians)
        num_df = num_df[self.numeric_columns]
        num_mat = csr_matrix(num_df.to_numpy(dtype=np.float32))

        # TF-IDF
        text_mat = self.vectorizer.transform([combined_text])
        X_mat = hstack([text_mat, num_mat], format="csr", dtype=np.float32)
        dmat = xgb.DMatrix(X_mat)

        # 3. XGBoost prediction & TreeSHAP contributions
        xgb_prob = float(self.xgb_model.predict(dmat)[0])
        tree_contribs = self.xgb_model.predict(dmat, pred_contribs=True)[0]  # shape: (10017,)
        bias = float(tree_contribs[-1])
        feature_contribs = tree_contribs[:-1]

        # 4. Transformer predictions
        # RoBERTa is loaded only when needed by the primary ensemble
        self.ensure_roberta()
        with torch.no_grad():
            r_inp = self.roberta_tok(combined_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
            roberta_prob = float(torch.softmax(self.roberta_model(**r_inp).logits, dim=1).numpy()[0, 1])

            # BERT (Loaded on-demand only when selected)
            if include_all_models or include_bert:
                self.ensure_bert()
                b_inp = self.bert_tok(combined_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
                bert_prob = float(torch.softmax(self.bert_model(**b_inp).logits, dim=1).numpy()[0, 1])
            else:
                bert_prob = None

            # DeBERTa (Loaded on-demand only when selected)
            if include_all_models or include_deberta:
                self.ensure_deberta()
                d_inp = self.deberta_tok(combined_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
                deberta_prob = float(torch.softmax(self.deberta_model(**d_inp).logits, dim=1).numpy()[0, 1])
            else:
                deberta_prob = None

        # 5. Primary Ensemble: 55% XGBoost + 45% RoBERTa, threshold 0.54
        ensemble_fake_prob = float(0.55 * xgb_prob + 0.45 * roberta_prob)
        # Probabilities must be strictly complementary
        ensemble_genuine_prob = float(1.0 - ensemble_fake_prob)

        final_prediction = "Potentially Fake" if ensemble_fake_prob >= 0.54 else "Potentially Genuine"

        # 6. Map TreeSHAP contributions into the 3 groups
        text_dim = len(self.feature_names)  # 10000
        tfidf_contribs = feature_contribs[:text_dim]
        num_contribs = feature_contribs[text_dim:]

        num_shap_map = {col: float(val) for col, val in zip(self.numeric_columns, num_contribs)}

        # Find top active TF-IDF words present in text
        active_indices = text_mat.indices
        tfidf_signals = []
        for idx in active_indices:
            term = self.feature_names[idx]
            val = float(tfidf_contribs[idx])
            if abs(val) > 0.001:
                tfidf_signals.append({
                    "term": term,
                    "shap": round(val, 4),
                    "influence": "Increases fake score" if val > 0 else "Decreases fake score"
                })
        tfidf_signals.sort(key=lambda x: abs(x["shap"]), reverse=True)

        # Helper for direction and explanation
        def format_factor(name, actual_val, shap_val, desc_pos, desc_neg, desc_neutral="Minimal impact on prediction"):
            direction = "Neutral"
            if shap_val > 0.01:
                direction = "Increases fake score"
                expl = desc_pos
            elif shap_val < -0.01:
                direction = "Decreases fake score"
                expl = desc_neg
            else:
                expl = desc_neutral
            return {
                "factor": name,
                "actual_value": actual_val,
                "shap_value": round(shap_val, 4),
                "influence": direction,
                "explanation": expl
            }

        # A. Textual Factors
        textual_factors = [
            format_factor(
                "Word Count",
                f"{text_feats['text_word_count']} words",
                num_shap_map.get("text_word_count", 0.0),
                "Brief or sparse review length characteristic of automated submissions.",
                "Substantial word length indicating natural consumer detail.",
                "Standard review length."
            ),
            format_factor(
                "Character Count",
                f"{text_feats['text_character_count']} chars",
                num_shap_map.get("text_character_count", 0.0),
                "Unusual character density or very short character footprint.",
                "Typical descriptive character count for genuine feedback.",
                "Character count within normal variance."
            ),
            format_factor(
                "Average Word Length",
                f"{text_feats['text_average_word_length']} chars/word",
                num_shap_map.get("text_average_word_length", 0.0),
                "Unusually low or irregular vocabulary depth.",
                "Natural vocabulary complexity and authentic variation.",
                "Word complexity aligns with organic reviews."
            ),
            format_factor(
                "Repeated Words",
                f"{text_feats['text_repeated_word_count']} duplicates",
                num_shap_map.get("text_repeated_word_count", 0.0),
                "Repetitive phrasing or keyword stuffing pattern.",
                "Low redundancy reflecting organic expression.",
                "Word repetition within expected bounds."
            ),
            format_factor(
                "Exclamation & Question Marks",
                f"!:{text_feats['text_exclamation_count']}, ?:{text_feats['text_question_count']}",
                num_shap_map.get("text_exclamation_count", 0.0) + num_shap_map.get("text_question_count", 0.0),
                "Elevated punctuation density often used in promotional exaggeration.",
                "Restrained, balanced punctuation typical of genuine opinions.",
                "Punctuation usage aligns with typical reviews."
            ),
            format_factor(
                "Uppercase Ratio",
                f"{round(text_feats['text_uppercase_ratio'] * 100, 1)}%",
                num_shap_map.get("text_uppercase_ratio", 0.0),
                "Disproportionate capitalization often associated with spam emphasis.",
                "Natural sentence-cased structure.",
                "Standard capitalization ratio."
            )
        ]

        # B. Behavioural Factors
        if has_history:
            beh_factors = [
                format_factor(
                    "Previous Review Count",
                    f"{int(num_dict['prior_review_count'])} prior reviews",
                    num_shap_map.get("prior_review_count", 0.0),
                    "High reviewer volume exhibiting mass-submission characteristics.",
                    "Consistent, moderate historical review volume.",
                    "Review volume within typical user activity."
                ),
                format_factor(
                    "Distinct Products Reviewed",
                    f"{int(num_dict['prior_distinct_products'])} products",
                    num_shap_map.get("prior_distinct_products", 0.0),
                    "Disproportionate or anomalous catalog cross-section.",
                    "Diverse, natural purchase and review distribution.",
                    "Product browsing scope within expectations."
                ),
                format_factor(
                    "Historical Average Rating",
                    f"{round(num_dict['prior_average_rating'], 2)} / 5.0",
                    num_shap_map.get("prior_average_rating", 0.0),
                    "Skewed historical rating bias (extreme positive or negative tendency).",
                    "Balanced rating distribution across diverse products.",
                    "Historical rating consistent with general population."
                ),
                format_factor(
                    "Review Rating Pattern",
                    f"Submitted: {int(rating)} Stars",
                    num_shap_map.get("review_rating", 0.0),
                    "Polarized 5-star or 1-star rating typical of promotional/smear bursts.",
                    "Measured rating reflecting critical assessment.",
                    "Rating alignment matches typical sentiment."
                )
            ]
        else:
            beh_factors = [
                {
                    "factor": "Reviewer Account History",
                    "actual_value": "Unavailable (No history found for ID)",
                    "shap_value": round(num_shap_map.get("prior_review_count", 0.0) + num_shap_map.get("prior_distinct_products", 0.0), 4),
                    "influence": "Neutral",
                    "explanation": "Reviewer ID was not provided or not present in historical database. Features safely imputed with population baselines; no synthetic history invented."
                },
                format_factor(
                    "Review Rating Pattern",
                    f"Submitted: {int(rating)} Stars",
                    num_shap_map.get("review_rating", 0.0),
                    "Polarized rating typical of promotional or smear campaigns.",
                    "Nuanced rating reflecting balanced feedback.",
                    "Rating conforms to organic product feedback."
                )
            ]

        # C. Temporal Factors
        if has_history:
            temp_factors = [
                format_factor(
                    "Days Since Previous Review",
                    f"{round(num_dict['days_since_previous_review'], 1)} days" if pd.notna(num_dict['days_since_previous_review']) else "Unobserved",
                    num_shap_map.get("days_since_previous_review", 0.0),
                    "Rapid submission cadence indicating clustered reviewing burst.",
                    "Spaced-out review intervals consistent with organic consumption.",
                    "Temporal spacing conforms to normal user behavior."
                ),
                format_factor(
                    "Reviews in Last 7 Days",
                    f"{int(num_dict['prior_7day_review_count'])} reviews",
                    num_shap_map.get("prior_7day_review_count", 0.0),
                    "High 7-day review cluster reflecting velocity spike.",
                    "Low short-term velocity consistent with sporadic buying.",
                    "Weekly review frequency is normal."
                ),
                format_factor(
                    "Reviews in Last 30 Days",
                    f"{int(num_dict['prior_30day_review_count'])} reviews",
                    num_shap_map.get("prior_30day_review_count", 0.0),
                    "Elevated 30-day velocity indicating sustained campaign activity.",
                    "Typical monthly engagement rate.",
                    "Monthly review volume is balanced."
                )
            ]
        else:
            temp_factors = [
                {
                    "factor": "Temporal Frequency & Velocity",
                    "actual_value": "Unavailable (No prior dates)",
                    "shap_value": round(num_shap_map.get("prior_7day_review_count", 0.0) + num_shap_map.get("prior_30day_review_count", 0.0), 4),
                    "influence": "Neutral",
                    "explanation": "Temporal interval signals cannot be calculated without multi-date reviewer history. Imputed using baseline training medians."
                }
            ]

        # 7. Generate Dynamic Natural-Language Explanation
        nl_text = self.generate_natural_language_explanation(
            final_pred=final_prediction,
            ens_prob=ensemble_fake_prob,
            has_history=has_history,
            text_feats=text_feats,
            rating=rating,
            textual_factors=textual_factors,
            beh_factors=beh_factors,
            temp_factors=temp_factors,
            tfidf_signals=tfidf_signals
        )

        return {
            "prediction": final_prediction,
            "ensemble_fake_prob": round(ensemble_fake_prob, 4),
            "ensemble_genuine_prob": round(ensemble_genuine_prob, 4),
            "ensemble_fake_pct": round(ensemble_fake_prob * 100, 2),
            "ensemble_genuine_pct": round(ensemble_genuine_prob * 100, 2),
            "has_reviewer_history": has_history,
            "models_breakdown": {
                "xgboost": {
                    "name": "Tuned XGBoost (TreeSHAP + Features)",
                    "fake_prob": round(xgb_prob, 4),
                    "genuine_prob": round(1.0 - xgb_prob, 4),
                    "fake_pct": round(xgb_prob * 100, 2),
                    "prediction": "Potentially Fake" if xgb_prob >= 0.50 else "Potentially Genuine",
                    "status": "active"
                },
                "roberta": {
                    "name": "RoBERTa (Primary Ensemble Component)",
                    "fake_prob": round(roberta_prob, 4),
                    "genuine_prob": round(1.0 - roberta_prob, 4),
                    "fake_pct": round(roberta_prob * 100, 2),
                    "prediction": "Potentially Fake" if roberta_prob >= 0.50 else "Potentially Genuine",
                    "status": "active"
                },
                "bert": {
                    "name": "BERT (Individual Model)",
                    "fake_prob": round(bert_prob, 4) if bert_prob is not None else None,
                    "genuine_prob": round(1.0 - bert_prob, 4) if bert_prob is not None else None,
                    "fake_pct": round(bert_prob * 100, 2) if bert_prob is not None else None,
                    "prediction": ("Potentially Fake" if bert_prob >= 0.50 else "Potentially Genuine") if bert_prob is not None else "Standby (On-Demand)",
                    "status": "active" if bert_prob is not None else "standby"
                },
                "deberta": {
                    "name": "DeBERTa (Individual Model)",
                    "fake_prob": round(deberta_prob, 4) if deberta_prob is not None else None,
                    "genuine_prob": round(1.0 - deberta_prob, 4) if deberta_prob is not None else None,
                    "fake_pct": round(deberta_prob * 100, 2) if deberta_prob is not None else None,
                    "prediction": ("Potentially Fake" if deberta_prob >= 0.50 else "Potentially Genuine") if deberta_prob is not None else "Standby (On-Demand)",
                    "status": "active" if deberta_prob is not None else "standby"
                }
            },
            "explainable_ai": {
                "natural_language_summary": nl_text,
                "textual_factors": textual_factors,
                "behavioural_factors": beh_factors,
                "temporal_factors": temp_factors,
                "top_tfidf_signals": tfidf_signals[:8],
                "base_log_odds": round(bias, 4),
                "shap_disclaimer": "SHAP feature contributions explain the XGBoost tabular & TF-IDF model component only. Deep transformers (RoBERTa, BERT, DeBERTa) evaluate semantic patterns through multi-head self-attention."
            }
        }

    def generate_natural_language_explanation(self, final_pred, ens_prob, has_history, text_feats,
                                              rating, textual_factors, beh_factors, temp_factors, tfidf_signals):
        reasons_pos = []
        reasons_neg = []

        # Check positive drivers (pushing toward fake)
        for f in textual_factors:
            if f["influence"] == "Increases fake score" and abs(f["shap_value"]) >= 0.05:
                reasons_pos.append(f["factor"].lower() + f" ({f['actual_value']})")
        for f in beh_factors:
            if f["influence"] == "Increases fake score" and abs(f["shap_value"]) >= 0.05:
                reasons_pos.append(f["factor"].lower() + f" ({f['actual_value']})")
        for f in temp_factors:
            if f["influence"] == "Increases fake score" and abs(f["shap_value"]) >= 0.05:
                reasons_pos.append(f["factor"].lower())

        # Check negative drivers (pushing toward genuine)
        for f in textual_factors:
            if f["influence"] == "Decreases fake score" and abs(f["shap_value"]) >= 0.05:
                reasons_neg.append(f["factor"].lower() + f" ({f['actual_value']})")
        for f in beh_factors:
            if f["influence"] == "Decreases fake score" and abs(f["shap_value"]) >= 0.05:
                reasons_neg.append(f["factor"].lower())
        for f in temp_factors:
            if f["influence"] == "Decreases fake score" and abs(f["shap_value"]) >= 0.05:
                reasons_neg.append(f["factor"].lower())

        prob_pct = f"{round(ens_prob * 100, 1)}%"

        if final_pred == "Potentially Fake":
            sentence1 = f"The review received a high fake probability of {prob_pct} based on the validated primary ensemble (55% XGBoost + 45% RoBERTa)."
            if has_history:
                history_note = "available reviewer history and velocity patterns contributed to the elevated risk score"
            else:
                history_note = "reviewer account history was unavailable, so the prediction is predominantly driven by textual and lexical markers"

            if reasons_pos:
                sentence2 = f"Key factors increasing the fake probability include {', '.join(reasons_pos[:3])}. Furthermore, {history_note}."
            else:
                sentence2 = f"Semantic and lexical structures identified by deep neural attention contributed to this assessment, alongside {history_note}."

            if tfidf_signals:
                top_terms = [t["term"] for t in tfidf_signals if t["influence"] == "Increases fake score"][:3]
                if top_terms:
                    sentence3 = f"Notable n-grams elevating the fake score include: '{', '.join(top_terms)}'."
                else:
                    sentence3 = "The overall rating pattern aligns with promotional sentiment spikes."
            else:
                sentence3 = "The rating pattern aligns with promotional sentiment spikes."

            return f"{sentence1} {sentence2} {sentence3}"

        else:
            sentence1 = f"The review received a low fake probability of {prob_pct} ({round((1.0 - ens_prob)*100, 1)}% genuine probability), classifying it as Potentially Genuine."
            if reasons_neg:
                sentence2 = f"Primary indicators decreasing the fake probability include {', '.join(reasons_neg[:3])}."
            else:
                sentence2 = f"Textual characteristics such as natural review length ({text_feats['text_word_count']} words) and balanced punctuation density reflect authentic consumer expression."

            if not has_history:
                sentence3 = "Note: No historical reviewer profile was linked; this score is based on content structure and semantic evaluation."
            else:
                sentence3 = "Available reviewer history reflects expected pacing and healthy product diversity."

            return f"{sentence1} {sentence2} {sentence3}"

    def predict_bulk(self, df: pd.DataFrame, mapping: dict):
        total_rows = len(df)
        if total_rows == 0:
            return {
                "total_uploaded": 0,
                "analyzed_count": 0,
                "fake_count": 0,
                "genuine_count": 0,
                "failure_count": 0,
                "fake_pct": 0,
                "genuine_pct": 0,
                "results": []
            }

        # Resolve mapped columns
        col_title = mapping.get("review_title", "review_title")
        col_text = mapping.get("review_text", "review_text")
        col_rating = mapping.get("rating", "rating")
        col_date = mapping.get("review_date", "review_date")
        col_rev_id = mapping.get("reviewer_id", "reviewer_id")
        col_prod_id = mapping.get("product_id", "product_id")

        results = []
        failures = 0

        # Vectorized prep for text and numbers
        titles = df[col_title].fillna("").astype(str).tolist() if col_title in df.columns else [""] * total_rows
        texts = df[col_text].fillna("").astype(str).tolist() if col_text in df.columns else [""] * total_rows
        combined_texts = [(t + " " + x).strip() if (t or x) else "No text" for t, x in zip(titles, texts)]

        ratings = pd.to_numeric(df[col_rating], errors="coerce").fillna(5.0).tolist() if col_rating in df.columns else [5.0] * total_rows
        rev_ids = df[col_rev_id].fillna("").astype(str).tolist() if col_rev_id in df.columns else [""] * total_rows
        prod_ids = df[col_prod_id].fillna("").astype(str).tolist() if col_prod_id in df.columns else [""] * total_rows
        dates = df[col_date].fillna("").astype(str).tolist() if col_date in df.columns else [""] * total_rows

        # 1. XGBoost batch inference
        text_mat = self.vectorizer.transform(combined_texts)
        num_rows = []
        for i in range(total_rows):
            t_feats = self.extract_textual_features(combined_texts[i])
            r_hist = self.lookup_reviewer_history(rev_ids[i])
            has_h = r_hist is not None

            nd = {col: np.nan for col in self.numeric_features}
            nd["review_rating"] = float(ratings[i])
            nd["text_word_count"] = t_feats["text_word_count"]
            nd["text_character_count"] = t_feats["text_character_count"]
            nd["text_average_word_length"] = t_feats["text_average_word_length"]
            nd["text_exclamation_count"] = t_feats["text_exclamation_count"]
            nd["text_question_count"] = t_feats["text_question_count"]
            nd["text_uppercase_ratio"] = t_feats["text_uppercase_ratio"]
            nd["text_repeated_word_count"] = t_feats["text_repeated_word_count"]

            if has_h:
                nd["prior_review_count"] = float(r_hist.get("reviewer_total_reviews") or 1)
                nd["prior_distinct_products"] = float(r_hist.get("reviewer_distinct_products") or 1)
                nd["prior_average_rating"] = float(r_hist.get("reviewer_average_rating") or ratings[i])
                raw_gap = r_hist.get("reviewer_average_days_between_reviews")
                nd["days_since_previous_review"] = float(raw_gap) if (raw_gap is not None and not pd.isna(raw_gap)) else np.nan
                avg_gap = float(raw_gap) if (raw_gap is not None and not pd.isna(raw_gap)) else 999.0
                if avg_gap > 0 and avg_gap < 999.0:
                    nd["prior_7day_review_count"] = min(5, int(7.0 / avg_gap))
                    nd["prior_30day_review_count"] = min(15, int(30.0 / avg_gap))
                else:
                    nd["prior_7day_review_count"] = 0
                    nd["prior_30day_review_count"] = 0
            else:
                nd["prior_review_count"] = 0.0
                nd["prior_distinct_products"] = 0.0
                nd["prior_average_rating"] = np.nan
                nd["days_since_previous_review"] = np.nan
                nd["prior_7day_review_count"] = 0.0
                nd["prior_30day_review_count"] = 0.0

            num_rows.append(nd)

        bulk_num_df = pd.DataFrame(num_rows)
        bulk_num_df["prior_average_rating_missing"] = bulk_num_df["prior_average_rating"].isna().astype(np.float32)
        bulk_num_df["days_since_previous_review_missing"] = bulk_num_df["days_since_previous_review"].isna().astype(np.float32)
        bulk_num_df = bulk_num_df.fillna(self.medians)
        bulk_num_df = bulk_num_df[self.numeric_columns]
        bulk_num_mat = csr_matrix(bulk_num_df.to_numpy(dtype=np.float32))

        X_bulk = hstack([text_mat, bulk_num_mat], format="csr", dtype=np.float32)
        xgb_bulk_probs = self.xgb_model.predict(xgb.DMatrix(X_bulk))

        # 2. RoBERTa batch inference (batches of 32 for memory efficiency)
        self.ensure_roberta()
        roberta_bulk_probs = []
        batch_size = 32
        with torch.no_grad():
            for b in range(0, total_rows, batch_size):
                chunk = combined_texts[b:b+batch_size]
                inp = self.roberta_tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
                logits = self.roberta_model(**inp).logits
                probs = torch.softmax(logits, dim=1).numpy()[:, 1]
                roberta_bulk_probs.extend(probs)

        # 3. Ensemble calculation
        fake_count = 0
        genuine_count = 0

        for i in range(total_rows):
            p_xgb = float(xgb_bulk_probs[i])
            p_rob = float(roberta_bulk_probs[i])
            p_ens = float(0.55 * p_xgb + 0.45 * p_rob)
            p_gen = float(1.0 - p_ens)

            pred = "Potentially Fake" if p_ens >= 0.54 else "Potentially Genuine"
            if pred == "Potentially Fake":
                fake_count += 1
            else:
                genuine_count += 1

            preview = (texts[i] if texts[i] else titles[i])
            if len(preview) > 100:
                preview = preview[:97] + "..."

            results.append({
                "row_id": i + 1,
                "review_title": titles[i],
                "review_text": texts[i],
                "preview": preview,
                "rating": float(ratings[i]),
                "review_date": dates[i],
                "reviewer_id": rev_ids[i],
                "product_id": prod_ids[i],
                "fake_prob": round(p_ens, 4),
                "genuine_prob": round(p_gen, 4),
                "fake_pct": round(p_ens * 100, 2),
                "genuine_pct": round(p_gen * 100, 2),
                "xgb_fake_prob": round(p_xgb, 4),
                "roberta_fake_prob": round(p_rob, 4),
                "prediction": pred
            })

        analyzed_count = len(results)
        fake_pct = round((fake_count / analyzed_count * 100), 1) if analyzed_count > 0 else 0
        genuine_pct = round((genuine_count / analyzed_count * 100), 1) if analyzed_count > 0 else 0

        return {
            "total_uploaded": total_rows,
            "analyzed_count": analyzed_count,
            "fake_count": fake_count,
            "genuine_count": genuine_count,
            "failure_count": failures,
            "fake_pct": fake_pct,
            "genuine_pct": genuine_pct,
            "results": results
        }

