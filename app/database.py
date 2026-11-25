import sqlite3
import os
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reviewshield_app.db")

def get_db_path() -> str:
    return os.path.abspath(DB_PATH)

def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def generate_prediction_id() -> str:
    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rand_suffix = uuid.uuid4().hex[:6].upper()
    return f"PRED_{timestamp_str}_{rand_suffix}"

def generate_batch_id() -> str:
    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rand_suffix = uuid.uuid4().hex[:6].upper()
    return f"BATCH_{timestamp_str}_{rand_suffix}"

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Create main table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyzed_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id TEXT UNIQUE,
            batch_id TEXT,
            source TEXT DEFAULT 'single',
            review_title TEXT,
            review_text TEXT,
            rating REAL,
            review_date TEXT,
            reviewer_id TEXT,
            product_id TEXT,
            ensemble_prediction TEXT,
            ensemble_fake_prob REAL,
            ensemble_genuine_prob REAL,
            xgb_fake_prob REAL,
            roberta_fake_prob REAL,
            bert_fake_prob REAL,
            deberta_fake_prob REAL,
            model_version TEXT DEFAULT 'v2.0-ensemble-55-45',
            explanation_json TEXT,
            has_reviewer_history INTEGER DEFAULT 0,
            analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Check and migrate columns if upgrading from earlier schema
    cursor.execute("PRAGMA table_info(analyzed_reviews)")
    existing_cols = {row["name"] for row in cursor.fetchall()}
    
    if "prediction_id" not in existing_cols:
        cursor.execute("ALTER TABLE analyzed_reviews ADD COLUMN prediction_id TEXT")
    if "batch_id" not in existing_cols:
        cursor.execute("ALTER TABLE analyzed_reviews ADD COLUMN batch_id TEXT")
    if "model_version" not in existing_cols:
        cursor.execute("ALTER TABLE analyzed_reviews ADD COLUMN model_version TEXT DEFAULT 'v2.0-ensemble-55-45'")
    if "explanation_json" not in existing_cols:
        cursor.execute("ALTER TABLE analyzed_reviews ADD COLUMN explanation_json TEXT")

    # Create indexes for fast lookup and dashboard aggregations
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_id ON analyzed_reviews(prediction_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_id ON analyzed_reviews(batch_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON analyzed_reviews(analysis_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction ON analyzed_reviews(ensemble_prediction)")

    # Ensure any existing records without prediction_id get one assigned
    cursor.execute("SELECT id FROM analyzed_reviews WHERE prediction_id IS NULL OR prediction_id = ''")
    rows_to_update = cursor.fetchall()
    for row in rows_to_update:
        r_id = row["id"]
        pred_id = f"PRED_LEGACY_{r_id:06d}"
        cursor.execute("UPDATE analyzed_reviews SET prediction_id = ? WHERE id = ?", (pred_id, r_id))
    
    conn.commit()
    conn.close()

def log_analysis(data: dict) -> dict:
    conn = get_db()
    cursor = conn.cursor()
    
    prediction_id = data.get("prediction_id") or generate_prediction_id()
    model_version = data.get("model_version") or "v2.0-ensemble-55-45"
    
    explanation_raw = data.get("explanation_json")
    if isinstance(explanation_raw, (dict, list)):
        explanation_json = json.dumps(explanation_raw)
    elif isinstance(explanation_raw, str):
        explanation_json = explanation_raw
    else:
        explanation_json = None

    cursor.execute("""
        INSERT INTO analyzed_reviews (
            prediction_id, batch_id, source, review_title, review_text, rating, review_date,
            reviewer_id, product_id, ensemble_prediction, ensemble_fake_prob,
            ensemble_genuine_prob, xgb_fake_prob, roberta_fake_prob,
            bert_fake_prob, deberta_fake_prob, model_version, explanation_json,
            has_reviewer_history
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction_id,
        data.get("batch_id"),
        data.get("source", "single"),
        data.get("review_title", ""),
        data.get("review_text", ""),
        float(data.get("rating") if data.get("rating") is not None else 5.0),
        data.get("review_date") or datetime.utcnow().strftime("%Y-%m-%d"),
        data.get("reviewer_id") or "",
        data.get("product_id") or "",
        data.get("ensemble_prediction", "Potentially Fake"),
        float(data.get("ensemble_fake_prob", 0.5)),
        float(data.get("ensemble_genuine_prob", 0.5)),
        float(data.get("xgb_fake_prob")) if data.get("xgb_fake_prob") is not None else None,
        float(data.get("roberta_fake_prob")) if data.get("roberta_fake_prob") is not None else None,
        float(data.get("bert_fake_prob")) if data.get("bert_fake_prob") is not None else None,
        float(data.get("deberta_fake_prob")) if data.get("deberta_fake_prob") is not None else None,
        model_version,
        explanation_json,
        1 if data.get("has_reviewer_history") else 0
    ))
    
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "db_id": new_id,
        "prediction_id": prediction_id
    }

def log_bulk_analyses(records: list, batch_id: Optional[str] = None) -> str:
    if not batch_id:
        batch_id = generate_batch_id()
        
    conn = get_db()
    cursor = conn.cursor()
    
    rows = []
    for r in records:
        pred_id = r.get("prediction_id") or generate_prediction_id()
        r["prediction_id"] = pred_id
        r["batch_id"] = batch_id
        
        explanation_raw = r.get("explanation_json")
        if isinstance(explanation_raw, (dict, list)):
            expl_str = json.dumps(explanation_raw)
        elif isinstance(explanation_raw, str):
            expl_str = explanation_raw
        else:
            expl_str = None
            
        rows.append((
            pred_id,
            batch_id,
            r.get("source", "bulk"),
            r.get("review_title", ""),
            r.get("review_text", ""),
            float(r.get("rating") if r.get("rating") is not None else 5.0),
            r.get("review_date") or datetime.utcnow().strftime("%Y-%m-%d"),
            r.get("reviewer_id") or "",
            r.get("product_id") or "",
            r.get("ensemble_prediction", "Potentially Fake"),
            float(r.get("ensemble_fake_prob", 0.5)),
            float(r.get("ensemble_genuine_prob", 0.5)),
            float(r.get("xgb_fake_prob")) if r.get("xgb_fake_prob") is not None else None,
            float(r.get("roberta_fake_prob")) if r.get("roberta_fake_prob") is not None else None,
            float(r.get("bert_fake_prob")) if r.get("bert_fake_prob") is not None else None,
            float(r.get("deberta_fake_prob")) if r.get("deberta_fake_prob") is not None else None,
            r.get("model_version") or "v2.0-ensemble-55-45",
            expl_str,
            1 if r.get("has_reviewer_history") else 0
        ))
        
    cursor.executemany("""
        INSERT INTO analyzed_reviews (
            prediction_id, batch_id, source, review_title, review_text, rating, review_date,
            reviewer_id, product_id, ensemble_prediction, ensemble_fake_prob,
            ensemble_genuine_prob, xgb_fake_prob, roberta_fake_prob,
            bert_fake_prob, deberta_fake_prob, model_version, explanation_json,
            has_reviewer_history
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    
    conn.commit()
    conn.close()
    return batch_id

def get_prediction_by_id(pred_identifier: str) -> Optional[dict]:
    conn = get_db()
    cursor = conn.cursor()
    
    # Try by prediction_id string first, then integer id
    cursor.execute("""
        SELECT * FROM analyzed_reviews
        WHERE prediction_id = ? OR id = ?
        LIMIT 1
    """, (pred_identifier, pred_identifier if pred_identifier.isdigit() else -1))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    res = dict(row)
    if res.get("explanation_json"):
        try:
            res["explanation_data"] = json.loads(res["explanation_json"])
        except Exception:
            res["explanation_data"] = None
    else:
        res["explanation_data"] = None
        
    return res

def get_prediction_history(limit: int = 50, offset: int = 0, category: Optional[str] = None, search: Optional[str] = None) -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.cursor()
    
    query = "SELECT * FROM analyzed_reviews WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM analyzed_reviews WHERE 1=1"
    params = []
    
    if category and isinstance(category, str) and category.upper() not in ["ALL", ""]:
        query += " AND ensemble_prediction = ?"
        count_query += " AND ensemble_prediction = ?"
        params.append(category)
        
    if search and isinstance(search, str) and search.strip():
        search_param = f"%{search.strip()}%"
        query += " AND (review_title LIKE ? OR review_text LIKE ? OR prediction_id LIKE ? OR batch_id LIKE ? OR reviewer_id LIKE ? OR product_id LIKE ? OR source LIKE ?)"
        count_query += " AND (review_title LIKE ? OR review_text LIKE ? OR prediction_id LIKE ? OR batch_id LIKE ? OR reviewer_id LIKE ? OR product_id LIKE ? OR source LIKE ?)"
        params.extend([search_param, search_param, search_param, search_param, search_param, search_param, search_param])
        
    # Get total count
    cursor.execute(count_query, params)
    total_count = cursor.fetchone()[0]
    
    # Get paginated records
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    paginated_params = list(params)
    paginated_params.extend([limit, offset])
    cursor.execute(query, paginated_params)
    
    records = []
    for row in cursor.fetchall():
        r_dict = dict(row)
        # Add preview fields for table
        text_preview = (r_dict.get("review_text") or "")[:120]
        if len(r_dict.get("review_text") or "") > 120:
            text_preview += "..."
        r_dict["review_text_preview"] = text_preview
        records.append(r_dict)
        
    conn.close()
    
    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "records": records
    }

def get_dashboard_stats() -> dict:
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM analyzed_reviews")
    total_analyzed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM analyzed_reviews WHERE ensemble_prediction = 'Potentially Fake'")
    fake_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM analyzed_reviews WHERE ensemble_prediction = 'Potentially Genuine'")
    genuine_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT batch_id) FROM analyzed_reviews WHERE batch_id IS NOT NULL AND batch_id != ''")
    batch_count = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT id, prediction_id, batch_id, source, review_title, review_text, rating,
               review_date, reviewer_id, product_id, ensemble_prediction,
               ensemble_fake_prob, ensemble_genuine_prob, xgb_fake_prob,
               roberta_fake_prob, analysis_timestamp
        FROM analyzed_reviews
        ORDER BY id DESC
        LIMIT 10
    """)
    recent_activity = [dict(row) for row in cursor.fetchall()]
    
    # Also fetch recent distribution data points for the live chart
    cursor.execute("""
        SELECT ensemble_fake_prob, ensemble_prediction, analysis_timestamp
        FROM analyzed_reviews
        ORDER BY id DESC
        LIMIT 30
    """)
    distribution_points = [dict(row) for row in cursor.fetchall()]
    distribution_points.reverse()  # chronological order
    
    conn.close()
    
    fake_pct = round((fake_count / total_analyzed * 100), 1) if total_analyzed > 0 else 0
    genuine_pct = round((genuine_count / total_analyzed * 100), 1) if total_analyzed > 0 else 0
    
    return {
        "total_analyzed": total_analyzed,
        "fake_count": fake_count,
        "genuine_count": genuine_count,
        "fake_percentage": fake_pct,
        "genuine_percentage": genuine_pct,
        "batch_count": batch_count,
        "recent_activity": recent_activity,
        "distribution_points": distribution_points
    }
