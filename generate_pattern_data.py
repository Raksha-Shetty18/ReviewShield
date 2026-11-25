import json
import pandas as pd
import numpy as np

print("Generating Behaviour & Pattern Analysis dataset-level data...")

test = pd.read_csv('data/reviewshield_test.csv', low_memory=False)

# 1. Rating Distribution (Fake vs Genuine)
rating_fake = test[test['label'] == 1]['review_rating'].value_counts().sort_index().to_dict()
rating_genuine = test[test['label'] == 0]['review_rating'].value_counts().sort_index().to_dict()

# Ensure 1 to 5 stars are present
all_stars = [1.0, 2.0, 3.0, 4.0, 5.0]
rating_dist = {
    "labels": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
    "fake": [int(rating_fake.get(s, 0)) for s in all_stars],
    "genuine": [int(rating_genuine.get(s, 0)) for s in all_stars]
}

# 2. Timing Distribution (Days between consecutive reviews)
valid_days = test.dropna(subset=['days_since_previous_review'])
fake_days = valid_days[valid_days['label'] == 1]['days_since_previous_review']
gen_days = valid_days[valid_days['label'] == 0]['days_since_previous_review']

def bin_days(series):
    return [
        int((series <= 1).sum()),
        int(((series > 1) & (series <= 7)).sum()),
        int(((series > 7) & (series <= 30)).sum()),
        int(((series > 30) & (series <= 90)).sum()),
        int((series > 90).sum())
    ]

time_dist = {
    "labels": ["Within 24 Hours", "2 - 7 Days", "8 - 30 Days", "31 - 90 Days", "> 90 Days"],
    "fake": bin_days(fake_days),
    "genuine": bin_days(gen_days)
}

# 3. Burst Patterns (7-day and 30-day review clusters)
burst_7day = {
    "labels": ["0 Prior (Isolated)", "1 - 2 Reviews", "3 - 5 Reviews", "6+ Rapid Reviews"],
    "fake": [
        int((test[test['label'] == 1]['prior_7day_review_count'] == 0).sum()),
        int(((test[test['label'] == 1]['prior_7day_review_count'] >= 1) & (test[test['label'] == 1]['prior_7day_review_count'] <= 2)).sum()),
        int(((test[test['label'] == 1]['prior_7day_review_count'] >= 3) & (test[test['label'] == 1]['prior_7day_review_count'] <= 5)).sum()),
        int((test[test['label'] == 1]['prior_7day_review_count'] >= 6).sum())
    ],
    "genuine": [
        int((test[test['label'] == 0]['prior_7day_review_count'] == 0).sum()),
        int(((test[test['label'] == 0]['prior_7day_review_count'] >= 1) & (test[test['label'] == 0]['prior_7day_review_count'] <= 2)).sum()),
        int(((test[test['label'] == 0]['prior_7day_review_count'] >= 3) & (test[test['label'] == 0]['prior_7day_review_count'] <= 5)).sum()),
        int((test[test['label'] == 0]['prior_7day_review_count'] >= 6).sum())
    ]
}

# 4. Reviewer-Product Activity (Distinct Products vs Total Reviews)
prod_dist = {
    "labels": ["Single Product Only", "2 - 4 Products", "5 - 10 Products", "11+ Products"],
    "fake": [
        int((test[test['label'] == 1]['prior_distinct_products'] <= 1).sum()),
        int(((test[test['label'] == 1]['prior_distinct_products'] >= 2) & (test[test['label'] == 1]['prior_distinct_products'] <= 4)).sum()),
        int(((test[test['label'] == 1]['prior_distinct_products'] >= 5) & (test[test['label'] == 1]['prior_distinct_products'] <= 10)).sum()),
        int((test[test['label'] == 1]['prior_distinct_products'] >= 11).sum())
    ],
    "genuine": [
        int((test[test['label'] == 0]['prior_distinct_products'] <= 1).sum()),
        int(((test[test['label'] == 0]['prior_distinct_products'] >= 2) & (test[test['label'] == 0]['prior_distinct_products'] <= 4)).sum()),
        int(((test[test['label'] == 0]['prior_distinct_products'] >= 5) & (test[test['label'] == 0]['prior_distinct_products'] <= 10)).sum()),
        int((test[test['label'] == 0]['prior_distinct_products'] >= 11).sum())
    ]
}

# 5. Textual Pattern Differences
text_stats = {
    "avg_word_count": {
        "fake": round(float(test[test['label'] == 1]['text_word_count'].mean()), 1),
        "genuine": round(float(test[test['label'] == 0]['text_word_count'].mean()), 1)
    },
    "avg_char_count": {
        "fake": round(float(test[test['label'] == 1]['text_character_count'].mean()), 1),
        "genuine": round(float(test[test['label'] == 0]['text_character_count'].mean()), 1)
    },
    "uppercase_ratio": {
        "fake": round(float(test[test['label'] == 1]['text_uppercase_ratio'].mean() * 100), 2),
        "genuine": round(float(test[test['label'] == 0]['text_uppercase_ratio'].mean() * 100), 2)
    },
    "exclamation_per_review": {
        "fake": round(float(test[test['label'] == 1]['text_exclamation_count'].mean()), 2),
        "genuine": round(float(test[test['label'] == 0]['text_exclamation_count'].mean()), 2)
    },
    "repeated_words_avg": {
        "fake": round(float(test[test['label'] == 1]['text_repeated_word_count'].mean()), 1),
        "genuine": round(float(test[test['label'] == 0]['text_repeated_word_count'].mean()), 1)
    }
}

# Summary dataset metrics
dataset_summary = {
    "total_reviews": len(test),
    "fake_count": int((test['label'] == 1).sum()),
    "genuine_count": int((test['label'] == 0).sum()),
    "unique_reviewers": int(test['reviewer_id'].nunique()),
    "unique_products": int(test['asin'].nunique())
}

payload = {
    "dataset_summary": dataset_summary,
    "rating_distribution": rating_dist,
    "temporal_gaps": time_dist,
    "burst_patterns_7day": burst_7day,
    "product_diversity": prod_dist,
    "textual_metrics": text_stats
}

with open('data/pattern_analysis_data.json', 'w') as f:
    json.dump(payload, f, indent=2)

print("Pattern analysis data successfully saved to data/pattern_analysis_data.json!")
