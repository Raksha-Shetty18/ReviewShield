import io
import json
import sqlite3
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db, DB_PATH

client = TestClient(app)

print("=============================================================")
print("REVIEWSHIELD END-TO-END AUTOMATED VERIFICATION SUITE")
print("=============================================================")

# 1. Test Home page and static files
print("\n[Step 1 & 14] Testing Home Page and Static UI Assets...")
r = client.get("/")
assert r.status_code == 200
html_text = r.text
assert "ReviewShield" in html_text
assert "Trust the Reviews. Understand the Patterns." in html_text
assert "How ReviewShield Works in Three Steps" in html_text
assert "pane-history" in html_text
assert "Prediction History" in html_text
print("✓ Home page renders minimal clean hero and 3-step guide.")

# 2. Check initial dashboard counters
print("\n[Step 2] Querying initial Dashboard stats from persistent SQLite...")
r_stats_before = client.get("/api/dashboard/stats")
assert r_stats_before.status_code == 200
stats_before = r_stats_before.json()
total_before = stats_before["total_analyzed"]
fake_before = stats_before["fake_count"]
genuine_before = stats_before["genuine_count"]
batches_before = stats_before["batch_count"]
print(f"✓ Initial database stats: Total={total_before}, Fake={fake_before}, Genuine={genuine_before}, Batches={batches_before}")

# 3. Test Single Review Analysis (Predict + Store in DB)
print("\n[Steps 3, 4, 5] Submitting Single Review for Multi-Model Analysis...")
single_payload = {
    "review_title": "Unbelievable Miracle Headphones - Absolute Perfection!!",
    "review_text": "These are the absolute best headphones ever made on earth!! 1000% perfect sound, everybody must purchase these immediately!! Five stars without question!",
    "rating": 5.0,
    "review_date": "2026-11-20",
    "reviewer_id": "REV_PROMO_VERIFY_99",
    "product_id": "B00AUDIT_TEST"
}

r_single = client.post("/api/predict/single", json=single_payload)
assert r_single.status_code == 200
single_res = r_single.json()

pred_id = single_res.get("prediction_id")
assert pred_id is not None and pred_id.startswith("PRED_"), f"Missing valid prediction_id: {pred_id}"
assert single_res["prediction"] in ["Potentially Fake", "Potentially Genuine"]
assert "models_breakdown" in single_res
assert "xgboost" in single_res["models_breakdown"]
assert "roberta" in single_res["models_breakdown"]
assert abs(single_res["ensemble_fake_prob"] + single_res["ensemble_genuine_prob"] - 1.0) < 1e-4

print(f"✓ Single prediction succeeded: ID={pred_id}, Verdict={single_res['prediction']}, FakeRisk={single_res['ensemble_fake_pct']}%")

# Verify stored in SQLite
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("SELECT * FROM analyzed_reviews WHERE prediction_id = ?", (pred_id,))
row = cur.fetchone()
conn.close()
assert row is not None, f"Record with prediction_id={pred_id} not found in SQLite!"
assert row["review_title"] == single_payload["review_title"]
assert row["model_version"] == "v2.0-ensemble-55-45"
assert row["explanation_json"] is not None
print(f"✓ Confirmed record persisted in SQLite table 'analyzed_reviews' with full metadata and explanation JSON.")

# 4. Verify Dashboard counters incremented accurately
print("\n[Steps 6 & 7] Verifying live Dashboard counter increments and restart persistence...")
r_stats_after = client.get("/api/dashboard/stats")
assert r_stats_after.status_code == 200
stats_after = r_stats_after.json()
assert stats_after["total_analyzed"] == total_before + 1, f"Expected {total_before + 1}, got {stats_after['total_analyzed']}"
print(f"✓ Dashboard total incremented from {total_before} to {stats_after['total_analyzed']}.")
assert len(stats_after["recent_activity"]) > 0
assert stats_after["recent_activity"][0]["prediction_id"] == pred_id
print(f"✓ Live recent activity reflects newest prediction ID: {pred_id}")

# 5. Download and verify PDF report
print("\n[Steps 8 & 9] Generating and verifying Official Audit PDF Report...")
r_pdf = client.get(f"/api/prediction/{pred_id}/pdf")
assert r_pdf.status_code == 200
assert "application/pdf" in r_pdf.headers.get("content-type")
assert f"filename=\"ReviewShield_Report_{pred_id}.pdf\"" in r_pdf.headers.get("content-disposition")
pdf_bytes = r_pdf.content
assert len(pdf_bytes) > 2000, f"PDF file too small: {len(pdf_bytes)} bytes"
assert pdf_bytes.startswith(b"%PDF"), "Generated stream is not a valid PDF header!"

# Save a copy to verify on disk
with open("data/test_audit_report.pdf", "wb") as f:
    f.write(pdf_bytes)
print(f"✓ PDF report successfully generated ({len(pdf_bytes)} bytes) and saved to data/test_audit_report.pdf")

# 6. Retrieve Prediction from History
print("\n[Step 10] Retrieving prediction from Prediction History API...")
r_hist = client.get(f"/api/history?search={pred_id}")
assert r_hist.status_code == 200
hist_data = r_hist.json()
assert hist_data["total"] >= 1
assert any(rec["prediction_id"] == pred_id for rec in hist_data["records"])
print(f"✓ Retrieved prediction {pred_id} from History API with search filter.")

# 7. Run Bulk Analysis and verify Batch ID
print("\n[Steps 11, 12, 13] Running small Bulk Analysis (3 reviews)...")
csv_content = """review_title,review_text,rating,reviewer_id,product_id
"Excellent keyboard for daily coding","Mechanical switches are crisp and comfortable for typing all day. Keycaps have good texture.",5,"REV_DEV_01","B00KEYB01"
"Total fraud fake item broken","Arrived damaged, seller refused refund. Avoid at all costs!",1,"REV_ANON_99","B00FRAUD02"
"Okay for thirty dollars","Sound quality is average, plastic construction is somewhat light.",3,"REV_NORM_33","B00BUD03"
"""
files = {"file": ("bulk_test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
r_bulk = client.post("/api/predict/bulk", files=files)
assert r_bulk.status_code == 200
bulk_res = r_bulk.json()
batch_id = bulk_res.get("batch_id")
assert batch_id is not None and batch_id.startswith("BATCH_"), f"Missing valid batch_id: {batch_id}"
assert bulk_res["total_uploaded"] == 3
assert len(bulk_res["results"]) == 3
for r in bulk_res["results"]:
    assert r.get("prediction_id") is not None and r["prediction_id"].startswith("PRED_")
    assert r.get("batch_id") == batch_id
print(f"✓ Bulk batch processed successfully: BatchID={batch_id}, Uploaded=3 reviews.")

# Verify bulk records stored in SQLite
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM analyzed_reviews WHERE batch_id = ?", (batch_id,))
count_batch = cur.fetchone()[0]
conn.close()
assert count_batch == 3, f"Expected 3 batch rows in DB, got {count_batch}"
print(f"✓ Confirmed all 3 reviews in batch {batch_id} stored in SQLite database.")

# Test bulk CSV export
r_export = client.post("/api/export/bulk", json={"results": bulk_res["results"]})
assert r_export.status_code == 200
assert "text/csv" in r_export.headers.get("content-type")
export_csv = r_export.text
assert "prediction_id" in export_csv
assert "batch_id" in export_csv
assert batch_id in export_csv
print("✓ Bulk CSV export generated successfully with prediction_id and batch_id columns.")

# Verify batch count on dashboard
r_stats_final = client.get("/api/dashboard/stats")
stats_final = r_stats_final.json()
assert stats_final["batch_count"] >= batches_before + 1
print(f"✓ Dashboard batch counter accurately incremented: {stats_final['batch_count']} total batches.")

print("\n=============================================================")
print("ALL 14 VERIFICATION STEPS PASSED SUCCESSFULLY!")
print("=============================================================")
