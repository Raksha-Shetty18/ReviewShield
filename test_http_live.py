import urllib.request
import json
import os
import io

print("--- 1. Testing GET /api/health ---")
req = urllib.request.Request("http://localhost:8000/api/health")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    print("Health response:", data)
    assert data["status"] == "ok"

print("\n--- 2. Testing GET /api/model-insights ---")
req = urllib.request.Request("http://localhost:8000/api/model-insights")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    ens = data["held_out_test"]["ensemble_metrics"]
    print(f"Verified Accuracy: {ens['accuracy']*100:.3f}% (Expected: 88.282%)")
    print(f"Verified Recall: {ens['recall']*100:.3f}% (Expected: 94.568%)")
    print(f"Verified Precision: {ens['precision']*100:.3f}% (Expected: 88.953%)")
    print(f"Verified F1: {ens['f1']*100:.3f}% (Expected: 91.675%)")
    print(f"Verified ROC-AUC: {ens['roc_auc']*100:.3f}% (Expected: 93.715%)")
    print(f"Verified PR-AUC: {ens['pr_auc']*100:.3f}% (Expected: 96.935%)")
    cm = ens["confusion_matrix"]
    print(f"Verified Confusion Matrix: {cm}")
    assert cm == [[1584, 534], [247, 4300]]

print("\n--- 3. Testing POST /api/predict/single ---")
payload = {
    "review_title": "AMAZING PRODUCT MUST BUY NOW!!!",
    "review_text": "This is hands down the greatest product on the planet!! I cannot live without it! The quality is 1000% better than anything else. Buy it immediately, five stars all the way!!",
    "rating": 5.0,
    "review_date": "2026-11-19",
    "reviewer_id": "AE222NWVQMRJ5Z2NRWQS7VR2Z7RQ",
    "product_id": "B01XYZ1234",
    "include_all_models": False
}
req = urllib.request.Request(
    "http://localhost:8000/api/predict/single",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    print("Single Prediction Verdict:", res["prediction"])
    print("Ensemble Fake Probability:", f"{res['ensemble_fake_pct']}%")
    print("Ensemble Genuine Probability:", f"{res['ensemble_genuine_pct']}%")
    assert "prediction" in res

print("\n--- 4. Testing POST /api/predict/bulk (5 reviews) ---")
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = io.BytesIO()

csv_content = """review_title,review_text,rating,review_date,reviewer_id,product_id
Super fantastic buy!,Totally loved it best purchase ever made in my entire life!!,5.0,2026-01-01,REV_01,P01
Broke immediately,Stopped working after two days. Very cheap material do not recommend.,1.0,2026-01-02,REV_02,P01
Decent for price,The item is reasonable for the money. Not amazing but gets the job done adequately.,3.0,2026-01-03,REV_03,P02
Five stars best ever,Five stars!! Highly recommended to all friends and family members!!,5.0,2026-01-04,REV_04,P02
Average quality,It was delivered on time and works as described in the manual.,4.0,2026-01-05,REV_05,P03
"""

# File part
body.write(f"--{boundary}\r\n".encode("utf-8"))
body.write(b'Content-Disposition: form-data; name="file"; filename="sample_5.csv"\r\n')
body.write(b'Content-Type: text/csv\r\n\r\n')
body.write(csv_content.encode("utf-8"))
body.write(b"\r\n")

# Mapping part
body.write(f"--{boundary}\r\n".encode("utf-8"))
body.write(b'Content-Disposition: form-data; name="mapping_json"\r\n\r\n')
body.write(b"{}\r\n")

body.write(f"--{boundary}--\r\n".encode("utf-8"))

content_bytes = body.getvalue()

req = urllib.request.Request(
    "http://localhost:8000/api/predict/bulk",
    data=content_bytes,
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(content_bytes))
    }
)
with urllib.request.urlopen(req) as resp:
    bdata = json.loads(resp.read().decode("utf-8"))
    print(f"Bulk Analysis Total Uploaded: {bdata['total_uploaded']}")
    print(f"Bulk Analysis Fake Count: {bdata['fake_count']} ({bdata['fake_pct']}%)")
    print(f"Bulk Analysis Genuine Count: {bdata['genuine_count']} ({bdata['genuine_pct']}%)")
    assert bdata["total_uploaded"] == 5
    assert len(bdata["results"]) == 5
    print("5 reviews classified successfully in batch!")

print("\nALL HTTP TESTS PASSED AGAINST LIVE SERVER ON PORT 8000!")
