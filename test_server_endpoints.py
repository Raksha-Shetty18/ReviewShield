import threading
import time
import requests
import uvicorn
from app.main import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)

base_url = "http://127.0.0.1:8001"

print("--- Testing /api/health ---")
r = requests.get(f"{base_url}/api/health")
print("Status:", r.status_code, "Response:", r.json())
assert r.status_code == 200

print("\n--- Testing /api/dashboard/stats ---")
r = requests.get(f"{base_url}/api/dashboard/stats")
print("Status:", r.status_code, "Total analyzed:", r.json().get("total_analyzed"))
assert r.status_code == 200

print("\n--- Testing /api/sample-reviews ---")
r = requests.get(f"{base_url}/api/sample-reviews")
print("Status:", r.status_code, "Samples count:", len(r.json().get("samples", [])))
assert r.status_code == 200

print("\n--- Testing /api/patterns ---")
r = requests.get(f"{base_url}/api/patterns")
print("Status:", r.status_code, "Keys:", list(r.json().keys()))
assert r.status_code == 200

print("\n--- Testing /api/model-insights ---")
r = requests.get(f"{base_url}/api/model-insights")
print("Status:", r.status_code, "Validation models:", list(r.json()["validation_benchmark"]["models"].keys()))
assert r.status_code == 200

print("\n--- Testing /api/predict/single (Suspected Fake) ---")
payload_fake = {
    "review_title": "BEST PRODUCT EVER AMAZING!!!",
    "review_text": "I absolutely love this miracle product!! Five stars, everybody must buy right now!!",
    "rating": 5.0,
    "reviewer_id": "AE222NWVQMRJ5Z2NRWQS7VR2Z7RQ"
}
r = requests.post(f"{base_url}/api/predict/single", json=payload_fake)
print("Status:", r.status_code)
res = r.json()
print("Prediction:", res.get("prediction"))
print("Ensemble Fake %:", res.get("ensemble_fake_pct"), "Genuine %:", res.get("ensemble_genuine_pct"))
print("Models:", {k: v.get("fake_pct") for k, v in res.get("models_breakdown", {}).items()})
print("Explanation excerpt:", res.get("explainable_ai", {}).get("natural_language_summary"))
assert r.status_code == 200
assert res.get("prediction") == "Potentially Fake"
assert abs(res.get("ensemble_fake_pct") + res.get("ensemble_genuine_pct") - 100.0) < 0.1

print("\n--- Testing /api/predict/single (Suspected Genuine) ---")
payload_gen = {
    "review_title": "Detailed feedback after 3 weeks",
    "review_text": "I purchased this item three weeks ago. The battery lasts around 8 hours, plastic feels somewhat cheap but sound quality is decent for commuting. A bit overpriced at $60.",
    "rating": 3.0,
    "reviewer_id": ""  # No reviewer ID -> unobserved history
}
r = requests.post(f"{base_url}/api/predict/single", json=payload_gen)
print("Status:", r.status_code)
res_gen = r.json()
print("Prediction:", res_gen.get("prediction"))
print("Ensemble Fake %:", res_gen.get("ensemble_fake_pct"), "Genuine %:", res_gen.get("ensemble_genuine_pct"))
print("Reviewer history unobserved handled correctly:", not res_gen.get("has_reviewer_history"))
assert r.status_code == 200
assert abs(res_gen.get("ensemble_fake_pct") + res_gen.get("ensemble_genuine_pct") - 100.0) < 0.1

print("\n--- Testing /api/predict/bulk (CSV upload) ---")
csv_data = "review_title,review_text,rating,reviewer_id\n" \
           "\"Awesome purchase\",\"Works exactly as advertised, very pleased.\",5,\"AE222NWVQMRJ5Z2NRWQS7VR2Z7RQ\"\n" \
           "\"Total scam fake\",\"Broke after one hour, refund refused!\",1,\"\"\n"
files = {"file": ("test.csv", csv_data, "text/csv")}
r = requests.post(f"{base_url}/api/predict/bulk", files=files)
print("Status:", r.status_code)
bulk_res = r.json()
print("Total uploaded:", bulk_res.get("total_uploaded"))
print("Fake count:", bulk_res.get("fake_count"), "Genuine count:", bulk_res.get("genuine_count"))
assert r.status_code == 200
assert bulk_res.get("total_uploaded") == 2

print("\nALL SERVER ENDPOINTS TESTED AND VERIFIED SUCCESSFULLY!")

