import urllib.request
import json
import io
import mimetypes

print("=== 1. Testing GET /api/health ===")
req = urllib.request.urlopen("http://localhost:8000/api/health")
health_data = json.loads(req.read().decode())
print("Health response:", health_data)
assert health_data["status"] == "ok"

print("\n=== 2. Testing GET /api/dashboard/stats ===")
req = urllib.request.urlopen("http://localhost:8000/api/dashboard/stats")
dash_data = json.loads(req.read().decode())
print("Total analyzed in DB:", dash_data["total_analyzed"], "| Fake count:", dash_data["fake_count"])

print("\n=== 3. Testing POST /api/predict/single ===")
single_payload = {
    "review_title": "Great headphones",
    "review_text": "I bought these headphones last week and they work really well. Battery lasts about 7 hours and sound quality is clear for the price.",
    "rating": 4.0,
    "reviewer_id": ""
}
req = urllib.request.Request(
    "http://localhost:8000/api/predict/single",
    data=json.dumps(single_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req)
single_res = json.loads(resp.read().decode())
print("Prediction:", single_res["prediction"])
print("Ensemble Fake %:", single_res["ensemble_fake_pct"], "| Genuine %:", single_res["ensemble_genuine_pct"])
print("Complementary Check:", single_res["ensemble_fake_pct"] + single_res["ensemble_genuine_pct"])
print("XGBoost Fake %:", single_res["models_breakdown"]["xgboost"]["fake_pct"])
print("RoBERTa Fake %:", single_res["models_breakdown"]["roberta"]["fake_pct"])
print("BERT Status:", single_res["models_breakdown"]["bert"]["status"])
print("DeBERTa Status:", single_res["models_breakdown"]["deberta"]["status"])
print("TreeSHAP Factors Count:", len(single_res["explainable_ai"]["textual_factors"]))
print("XAI Summary:", single_res["explainable_ai"]["natural_language_summary"][:120] + "...")

print("\n=== 4. Testing POST /api/predict/bulk (5 reviews sample) ===")
# Create a 5-review test CSV
sample_csv = """review_title,review_text,rating,reviewer_id
AMAZING MUST BUY!!,"Best purchase of my entire life, incredible quality!!",5.0,AE222NWVQMRJ5Z2NRWQS7VR2Z7RQ
Disappointed with battery,"The battery dies after only 2 hours. Not recommended.",1.0,
Solid daily driver,"Good value for $30. Fits nicely in ear with decent bass.",4.0,AE224FTQKEHGCNQA66HNIDEAZEQQ
Scam seller beware,"Never received package and seller blocked me. Terrible!",1.0,
Average product,"It does the job, nothing fancy. A bit plasticky.",3.0,
"""

boundary = "----ReviewShieldBoundary123456"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="sample_5_reviews.csv"\r\n'
    f"Content-Type: text/csv\r\n\r\n"
    f"{sample_csv}\r\n"
    f"--{boundary}--\r\n"
).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:8000/api/predict/bulk",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)
resp = urllib.request.urlopen(req)
bulk_res = json.loads(resp.read().decode())
print("Total Uploaded:", bulk_res["total_uploaded"])
print("Analyzed Count:", bulk_res["analyzed_count"])
print("Fake Count:", bulk_res["fake_count"], f"({bulk_res['fake_pct']}%)")
print("Genuine Count:", bulk_res["genuine_count"], f"({bulk_res['genuine_pct']}%)")
for r in bulk_res["results"]:
    print(f"  Row {r['row_id']}: [{r['prediction']}] Fake%={r['fake_pct']} | {r['preview'][:45]}...")

print("\n=== 5. Testing Navigation / Data Endpoints ===")
req = urllib.request.urlopen("http://localhost:8000/api/patterns")
patterns = json.loads(req.read().decode())
print("Patterns Keys:", list(patterns.keys()))

req = urllib.request.urlopen("http://localhost:8000/api/model-insights")
insights = json.loads(req.read().decode())
print("Insights Models:", list(insights["validation_benchmark"]["models"].keys()))

ens_test = insights["held_out_test"]["ensemble_metrics"]
xgb_test = insights["held_out_test"]["xgboost_full_test_metrics"]
print("Held-Out Test Ensemble Accuracy:", ens_test["accuracy"])
print("Held-Out Test Ensemble Recall:", ens_test["recall"])
print("Held-Out Test Ensemble Precision:", ens_test["precision"])
print("Held-Out Test Ensemble F1:", ens_test["f1"])
print("Held-Out Test Ensemble ROC-AUC:", ens_test["roc_auc"])
print("Held-Out Test Ensemble PR-AUC:", ens_test["pr_auc"])
print("Held-Out Test Confusion Matrix:", ens_test["confusion_matrix"])
print("Full XGBoost Test Accuracy:", xgb_test["accuracy"])
print("Full XGBoost Confusion Matrix:", xgb_test["confusion_matrix"])

assert ens_test["accuracy"] == 0.88282
assert ens_test["precision"] == 0.88953
assert ens_test["recall"] == 0.94568
assert ens_test["f1"] == 0.91675
assert ens_test["roc_auc"] == 0.93715
assert ens_test["pr_auc"] == 0.96935
assert ens_test["confusion_matrix"] == [[1584, 534], [247, 4300]]
assert xgb_test["accuracy"] == 0.84306
assert xgb_test["confusion_matrix"] == [[1523, 595], [451, 4096]]
print("--> ALL OFFICIAL HELD-OUT TEST METRICS & CONFUSION MATRICES CONFIRMED ON LIVE SERVER!")

print("\n=======================================================")
print("ALL LIVE TESTS ON PORT 8000 PASSED SUCCESSFULLY!")
print("=======================================================")
