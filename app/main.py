from app.shap_loader import get_global_shap_importance, get_shap_samples, get_shap_sample_by_id
import os
import json
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from app.database import (
    init_db, log_analysis, log_bulk_analyses, get_dashboard_stats,
    get_prediction_by_id, get_prediction_history, generate_prediction_id, generate_batch_id
)
from app.report_generator import build_pdf_report
from app.inference import InferenceEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="ReviewShield", description="AI-Powered Fake Review Intelligence System", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Lazy inference engine loading
inference_engine = None

def get_engine():
    global inference_engine
    if inference_engine is None:
        inference_engine = InferenceEngine.get_instance()
    return inference_engine

# Request schema for single review
class SingleReviewRequest(BaseModel):
    review_title: Optional[str] = ""
    review_text: str
    rating: Optional[float] = 5.0
    review_date: Optional[str] = None
    reviewer_id: Optional[str] = None
    product_id: Optional[str] = None
    include_all_models: Optional[bool] = False
    include_bert: Optional[bool] = False
    include_deberta: Optional[bool] = False

@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "ReviewShield", "version": "2.0"}

@app.get("/api/dashboard/stats")
def dashboard_stats():
    stats = get_dashboard_stats()
    return stats

@app.get("/api/history")
def prediction_history(
    limit: int = 25,
    offset: int = 0,
    category: Optional[str] = None,
    search: Optional[str] = None
):
    """Retrieves paginated audit history from the persistent SQLite database."""
    history_data = get_prediction_history(limit=limit, offset=offset, category=category, search=search)
    return history_data

@app.get("/api/prediction/{pred_id}")
def get_prediction_details(pred_id: str):
    """Retrieves full prediction details and explainability payload for a single review."""
    record = get_prediction_by_id(pred_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Prediction record '{pred_id}' not found.")
    return record

@app.get("/api/prediction/{pred_id}/pdf")
def download_prediction_pdf(pred_id: str):
    """Generates and downloads a formal 8-section audit report PDF for the given prediction ID."""
    record = get_prediction_by_id(pred_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Prediction record '{pred_id}' not found.")
        
    try:
        pdf_buffer = build_pdf_report(record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")

    clean_filename = f"ReviewShield_Report_{record.get('prediction_id') or pred_id}.pdf"
    
    response = Response(
        content=pdf_buffer.getvalue(),
        media_type="application/pdf"
    )
    response.headers["Content-Disposition"] = f'attachment; filename="{clean_filename}"'
    return response

@app.get("/api/sample-reviews")
def get_sample_reviews():
    samples = [
        {
            "id": "sample-1",
            "tag": "Suspicious Promotional Burst",
            "type": "fake",
            "title": "AMAZING PRODUCT MUST BUY NOW!!!",
            "text": "This is hands down the greatest product on the planet!! I cannot live without it! The quality is 1000% better than anything else. Buy it immediately, five stars all the way!!",
            "rating": 5.0,
            "date": "2026-11-15",
            "reviewer_id": "AE222NWVQMRJ5Z2NRWQS7VR2Z7RQ",
            "product_id": "B01XYZ1234"
        },
        {
            "id": "sample-2",
            "tag": "Authentic In-Depth Analysis",
            "type": "genuine",
            "title": "Solid performance with a few ergonomic caveats",
            "text": "I tested this unit for three weeks before writing this review. Build quality is predominantly matte plastic, which prevents fingerprints. Battery life averages roughly 7.5 hours under moderate continuous load. However, the hinge feels slightly stiff, and the included USB-C cable is barely three feet long. Decent value for eighty dollars.",
            "rating": 4.0,
            "date": "2026-11-10",
            "reviewer_id": "AE224FTQKEHGCNQA66HNIDEAZEQQ",
            "product_id": "B02ABC5678"
        },
        {
            "id": "sample-3",
            "tag": "Short Vague Praise",
            "type": "fake",
            "title": "Good",
            "text": "Very good product love it works well very nice five star quality highly recommend to all.",
            "rating": 5.0,
            "date": "2026-11-12",
            "reviewer_id": "AE224DEP6EXXCREHJKBTAWE26R2Q",
            "product_id": "B03DEF9012"
        },
        {
            "id": "sample-4",
            "tag": "Critical Negative Feedback",
            "type": "genuine",
            "title": "Did not meet expectations for noise cancellation",
            "text": "Purchased primarily for daily subway commute. The passive seal is acceptable, but the active cancellation struggles significantly with low-frequency rumble. Mic clarity for phone calls is surprisingly clear, but overall sound profile is heavy on bloated bass.",
            "rating": 2.0,
            "date": "2026-11-05",
            "reviewer_id": "AE223AXDVIGECK3QS3QEFY5DAPGQ",
            "product_id": "B04GHI3456"
        }
    ]
    return {"samples": samples}

@app.post("/api/predict/single")
def predict_single_review(req: SingleReviewRequest):
    if not req.review_text.strip() and not req.review_title.strip():
        raise HTTPException(status_code=400, detail="Review text or title is required.")

    engine = get_engine()
    result = engine.predict_single(
        review_title=req.review_title,
        review_text=req.review_text,
        rating=req.rating or 5.0,
        review_date=req.review_date,
        reviewer_id=req.reviewer_id,
        product_id=req.product_id,
        include_all_models=req.include_all_models or False,
        include_bert=req.include_bert or False,
        include_deberta=req.include_deberta or False
    )

    pred_id = generate_prediction_id()
    
    explanation_payload = {
        "textual_factors": result.get("textual_factors", []),
        "behavioural_factors": result.get("behavioural_factors", []),
        "temporal_factors": result.get("temporal_factors", []),
        "tfidf_signals": result.get("tfidf_signals", []),
        "natural_language_explanation": result.get("natural_language_explanation", "")
    }

    # Log to persistent SQLite DB
    log_record = {
        "prediction_id": pred_id,
        "source": "single",
        "review_title": req.review_title,
        "review_text": req.review_text,
        "rating": req.rating or 5.0,
        "review_date": req.review_date,
        "reviewer_id": req.reviewer_id,
        "product_id": req.product_id,
        "ensemble_prediction": result["prediction"],
        "ensemble_fake_prob": result["ensemble_fake_prob"],
        "ensemble_genuine_prob": result["ensemble_genuine_prob"],
        "xgb_fake_prob": result["models_breakdown"]["xgboost"].get("fake_prob"),
        "roberta_fake_prob": result["models_breakdown"]["roberta"].get("fake_prob"),
        "bert_fake_prob": result["models_breakdown"]["bert"].get("fake_prob"),
        "deberta_fake_prob": result["models_breakdown"]["deberta"].get("fake_prob"),
        "model_version": "v2.0-ensemble-55-45",
        "explanation_json": explanation_payload,
        "has_reviewer_history": result["has_reviewer_history"]
    }
    
    db_res = log_analysis(log_record)
    result["review_id"] = db_res["db_id"]
    result["prediction_id"] = pred_id

    return result

@app.post("/api/predict/bulk")
async def predict_bulk_reviews(file: UploadFile = File(...), mapping_json: Optional[str] = Form(None)):
    filename = file.filename.lower()
    content = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel (.xlsx/.xls) file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Limit to maximum 1000 reviews per batch to protect memory and runtime
    if len(df) > 1000:
        df = df.iloc[:1000].copy()

    # Parse column mapping
    mapping = {}
    if mapping_json:
        try:
            mapping = json.loads(mapping_json)
        except Exception:
            mapping = {}

    # Auto-detect missing mappings
    cols_lower = {col.lower(): col for col in df.columns}

    def auto_map(candidates, default_name):
        if default_name in mapping and mapping[default_name] in df.columns:
            return mapping[default_name]
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        return None

    resolved_mapping = {
        "review_title": auto_map(["review_title", "title", "summary", "subject", "heading"], "review_title"),
        "review_text": auto_map(["review_text", "text", "content", "body", "review", "comments"], "review_text"),
        "rating": auto_map(["review_rating", "rating", "score", "stars", "star_rating"], "rating"),
        "review_date": auto_map(["review_date", "date", "timestamp", "time"], "review_date"),
        "reviewer_id": auto_map(["reviewer_id", "user_id", "author_id", "reviewer"], "reviewer_id"),
        "product_id": auto_map(["asin", "product_id", "item_id", "product"], "product_id")
    }

    engine = get_engine()
    bulk_result = engine.predict_bulk(df, resolved_mapping)

    batch_id = generate_batch_id()

    # Log bulk reviews to DB with unique prediction IDs and shared batch_id
    db_records = []
    for r in bulk_result["results"]:
        p_id = generate_prediction_id()
        r["prediction_id"] = p_id
        r["batch_id"] = batch_id
        db_records.append({
            "prediction_id": p_id,
            "batch_id": batch_id,
            "source": "bulk",
            "review_title": r.get("review_title", ""),
            "review_text": r.get("review_text", ""),
            "rating": r.get("rating", 5.0),
            "review_date": r.get("review_date", ""),
            "reviewer_id": r.get("reviewer_id", ""),
            "product_id": r.get("product_id", ""),
            "ensemble_prediction": r.get("prediction", "Potentially Fake"),
            "ensemble_fake_prob": r.get("fake_prob", 0.5),
            "ensemble_genuine_prob": r.get("genuine_prob", 0.5),
            "xgb_fake_prob": r.get("xgb_fake_prob", 0.5),
            "roberta_fake_prob": r.get("roberta_fake_prob", 0.5),
            "bert_fake_prob": r.get("bert_fake_prob", 0.5),
            "deberta_fake_prob": r.get("deberta_fake_prob", 0.5),
            "model_version": "v2.0-ensemble-55-45",
            "has_reviewer_history": bool(r.get("reviewer_id"))
        })

    log_bulk_analyses(db_records, batch_id=batch_id)

    bulk_result["batch_id"] = batch_id
    bulk_result["column_mapping_used"] = resolved_mapping
    return bulk_result

@app.post("/api/export/bulk")
def export_bulk_results(payload: dict):
    results = payload.get("results", [])
    if not results:
        raise HTTPException(status_code=400, detail="No analysis results to export.")

    export_df = pd.DataFrame(results)
    # Reorder columns logically
    cols = [
        "row_id", "prediction_id", "prediction", "fake_pct", "genuine_pct", "rating",
        "review_title", "review_text", "reviewer_id", "product_id", "review_date",
        "fake_prob", "genuine_prob", "xgb_fake_prob", "roberta_fake_prob", "batch_id"
    ]
    available_cols = [c for c in cols if c in export_df.columns]
    export_df = export_df[available_cols]

    stream = io.StringIO()
    export_df.to_csv(stream, index=False)
    stream.seek(0)

    response = Response(content=stream.getvalue(), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=reviewshield_bulk_analysis.csv"
    return response

@app.get("/api/model-insights")
def get_model_insights():
    file_path = os.path.join(DATA_DIR, "model_insights_data.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model insights data file not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/shap/global")
def get_shap_global(limit: int = 30):
    return get_global_shap_importance(limit=limit)

@app.get("/api/shap/samples")
def get_shap_samples_list(limit: int = 20):
    return get_shap_samples(limit=limit)

@app.get("/api/shap/sample/{sample_id}")
def get_shap_sample(sample_id: int):
    res = get_shap_sample_by_id(sample_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found in SHAP benchmarks.")
    return res

@app.get("/api/patterns")
def get_patterns():
    file_path = os.path.join(DATA_DIR, "pattern_analysis_data.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Pattern analysis data file not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Mount static files
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    html_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ReviewShield Backend Online</h1><p>Static UI not yet found.</p>"
