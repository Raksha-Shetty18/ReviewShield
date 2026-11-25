import io
import pandas as pd
from app.database import log_bulk_analyses, generate_batch_id, generate_prediction_id, get_db

batch_id = generate_batch_id()
records = [
    {
        "source": "bulk",
        "review_title": "Great wireless mouse",
        "review_text": "Good battery life, ergonomic shape, works seamlessly.",
        "rating": 5.0,
        "reviewer_id": "REV_BATCH_01",
        "product_id": "B00MOUSE",
        "ensemble_prediction": "Potentially Genuine",
        "ensemble_fake_prob": 0.15,
        "ensemble_genuine_prob": 0.85,
        "xgb_fake_prob": 0.18,
        "roberta_fake_prob": 0.11,
        "has_reviewer_history": True
    },
    {
        "source": "bulk",
        "review_title": "SCAM FAKE BROKE IMMEDIATELY",
        "review_text": "Arrived broken in half, customer support never replied, refund refused!!",
        "rating": 1.0,
        "reviewer_id": "REV_BATCH_02",
        "product_id": "B00BROKEN",
        "ensemble_prediction": "Potentially Fake",
        "ensemble_fake_prob": 0.88,
        "ensemble_genuine_prob": 0.12,
        "xgb_fake_prob": 0.90,
        "roberta_fake_prob": 0.85,
        "has_reviewer_history": False
    }
]

res_batch_id = log_bulk_analyses(records, batch_id=batch_id)
print("Logged batch:", res_batch_id)

conn = get_db()
cursor = conn.cursor()
cursor.execute("SELECT prediction_id, batch_id, review_title, ensemble_prediction FROM analyzed_reviews WHERE batch_id = ?", (res_batch_id,))
rows = cursor.fetchall()
conn.close()

print(f"Retrieved {len(rows)} rows for batch {res_batch_id}:")
for r in rows:
    print(f" - ID: {r['prediction_id']}, Verdict: {r['ensemble_prediction']}, Title: {r['review_title']}")
assert len(rows) == 2
print("✓ Bulk batch logging verified successfully!")
