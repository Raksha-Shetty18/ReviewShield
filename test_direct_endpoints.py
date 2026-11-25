import os
import io
import json
import sqlite3
from app.main import (
    health_check, dashboard_stats, prediction_history,
    get_prediction_details, download_prediction_pdf,
    predict_single_review, export_bulk_results, SingleReviewRequest
)
from app.database import DB_PATH, get_prediction_by_id
from app.report_generator import build_pdf_report

print("=============================================================")
print("REVIEWSHIELD DIRECT E2E VERIFICATION TEST")
print("=============================================================")

# 1. Health check
h = health_check()
print("\n[Step 1] Testing /api/health...")
print("Health:", h)
assert h["status"] == "ok"
assert h["app"] == "ReviewShield"
print("✓ Health check passed.")

# 2. Initial Dashboard Stats
print("\n[Step 2] Testing initial /api/dashboard/stats...")
init_stats = dashboard_stats()
print(f"Total: {init_stats['total_analyzed']}, Fake: {init_stats['fake_count']}, Genuine: {init_stats['genuine_count']}, Batches: {init_stats['batch_count']}")
total_start = init_stats["total_analyzed"]
fake_start = init_stats["fake_count"]

# 3. Single Review Prediction
print("\n[Step 3] Testing /api/predict/single...")
req = SingleReviewRequest(
    review_title="AMAZING MIRACLE PRODUCT FIVE STARS!!",
    review_text="This is by far the greatest product in the entire universe!! Must buy right now, 1000% perfection!!",
    rating=5.0,
    review_date="2026-11-20",
    reviewer_id="AE222NWVQMRJ5Z2NRWQS7VR2Z7RQ",
    product_id="B01XYZ1234"
)
pred_res = predict_single_review(req)
pred_id = pred_res.get("prediction_id")
print(f"Prediction result: Verdict='{pred_res['prediction']}', FakeProb={pred_res['ensemble_fake_pct']}%, ID={pred_id}")
assert pred_id is not None and pred_id.startswith("PRED_")
assert pred_res["prediction"] in ["Potentially Fake", "Potentially Genuine"]
assert "models_breakdown" in pred_res
print("✓ Single review prediction returned valid ensemble verdict and prediction_id.")

# 4. Confirm Database Storage
print("\n[Step 4] Confirming prediction persisted in SQLite database...")
stored = get_prediction_by_id(pred_id)
assert stored is not None
assert stored["prediction_id"] == pred_id
assert stored["review_title"] == req.review_title
assert stored["model_version"] == "v2.0-ensemble-55-45"
assert stored["explanation_data"] is not None
print("✓ Confirmed record stored in SQLite table 'analyzed_reviews' with explanation JSON.")

# 5. Dashboard Counters Update
print("\n[Step 5] Confirming dashboard counters updated from database...")
after_stats = dashboard_stats()
print(f"Updated Total: {after_stats['total_analyzed']}, Fake: {after_stats['fake_count']}, Genuine: {after_stats['genuine_count']}")
assert after_stats["total_analyzed"] == total_start + 1
print("✓ Dashboard counter accurately incremented by exactly 1.")

# 6. PDF Report Generation
print("\n[Step 6] Testing /api/prediction/{pred_id}/pdf...")
pdf_resp = download_prediction_pdf(pred_id)
assert "ReviewShield_Report_" in pdf_resp.headers["Content-Disposition"]
assert pdf_resp.media_type == "application/pdf"
pdf_bytes = pdf_resp.body
print(f"PDF bytes generated via Response: {len(pdf_bytes)} bytes")
assert len(pdf_bytes) > 2000
assert pdf_bytes.startswith(b"%PDF")
with open("data/sample_generated_report.pdf", "wb") as f:
    f.write(pdf_bytes)
print("✓ PDF generated successfully with valid %PDF header and saved to data/sample_generated_report.pdf.")

# 7. Prediction History Retrieval
print("\n[Step 7] Testing /api/history query and filtering...")
hist = prediction_history(limit=10, search=pred_id)
print(f"History search result: total={hist['total']}, returned={len(hist['records'])}")
assert hist["total"] >= 1
assert any(r["prediction_id"] == pred_id for r in hist["records"])
print("✓ Retrieved newly predicted review from History API.")

# 8. Bulk CSV Export
print("\n[Step 8] Testing /api/export/bulk...")
bulk_sample_results = [
    {
        "row_id": 1,
        "prediction_id": pred_id,
        "prediction": pred_res["prediction"],
        "fake_pct": pred_res["ensemble_fake_pct"],
        "genuine_pct": pred_res["ensemble_genuine_pct"],
        "rating": 5.0,
        "review_title": req.review_title,
        "review_text": req.review_text,
        "reviewer_id": req.reviewer_id,
        "product_id": req.product_id,
        "review_date": "2026-11-20",
        "fake_prob": pred_res["ensemble_fake_prob"],
        "genuine_prob": pred_res["ensemble_genuine_prob"],
        "xgb_fake_prob": pred_res["models_breakdown"]["xgboost"]["fake_prob"],
        "roberta_fake_prob": pred_res["models_breakdown"]["roberta"]["fake_prob"]
    }
]
export_resp = export_bulk_results({"results": bulk_sample_results})
csv_output = export_resp.body.decode("utf-8")
print("CSV Export preview:\n" + csv_output[:250])
assert "prediction_id" in csv_output
assert pred_id in csv_output
print("✓ Bulk CSV export generated successfully with prediction_id.")

# 9. Verify Frontend Assets
print("\n[Step 9] Verifying Frontend Static HTML & Navigation...")
with open("static/index.html", "r", encoding="utf-8") as f:
    index_html = f.read()

with open("static/js/app.js", "r", encoding="utf-8") as f:
    app_js = f.read()

assert "ReviewShield" in index_html
assert "Trust the Reviews. Understand the Patterns." in index_html
assert "How ReviewShield Works in Three Steps" in index_html
assert "pane-history" in index_html
assert "data-tab=\"history\"" in index_html
assert "kpiTotalBatches" in index_html
assert "dashboardLiveTable" in index_html

# Check JS functions
assert "downloadSinglePdf" in app_js
assert "loadHistoryPage" in app_js
assert "renderHistoryTable" in app_js
assert "inspectHistoryRecord" in app_js
print("✓ Verified Frontend HTML and JS contain minimal Home, 3-step guide, Prediction History tab, Live Database Table, and PDF actions.")

print("\n=============================================================")
print("ALL 9 VERIFICATION MODULES PASSED COMPLETELY AND FLAWLESSLY!")
print("=============================================================")
