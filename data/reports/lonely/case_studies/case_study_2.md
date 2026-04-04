# Case Study: High-Distress Signal Week 2020-W03
**Week starting:** 2020-01-13
**Distress score:** 0.369

## What Happened
The community distress score spiked to 0.369, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2019-W52 (FLAGGED, probability: 0.43)
- **topic_shift_jsd_roll4w**: 0.0000 (100.0% below average)
- **first_person_singular_ratio_delta**: 0.0000 (100.0% below average)
- **avg_type_token_ratio_roll2w**: 0.0000 (100.0% below average)
- **avg_negative_delta**: 0.0000 (100.0% above average)
- **new_poster_ratio_delta**: 0.0000 (100.0% above average)

### 2020-W01 (FLAGGED, probability: 0.30)
- **topic_shift_jsd_roll4w**: 0.0208 (80.3% below average)
- **first_person_singular_ratio_delta**: 0.0823 (184781.7% above average)
- **avg_type_token_ratio_roll2w**: 0.3729 (45.9% below average)
- **avg_negative_delta**: 0.1275 (181561.7% above average)
- **new_poster_ratio_delta**: 0.8311 (31416.3% above average)

### 2020-W02 (FLAGGED, probability: 0.17)
- **topic_shift_jsd_roll4w**: 0.0424 (59.9% below average)
- **first_person_singular_ratio_delta**: 0.0034 (7573.6% above average)
- **avg_type_token_ratio_roll2w**: 0.7409 (7.6% above average)
- **avg_negative_delta**: 0.0045 (6440.8% above average)
- **new_poster_ratio_delta**: 0.0333 (1354.6% above average)

## Top Contributing Features (SHAP)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | topic_shift_jsd_roll4w | 1.3655 |
| 2 | first_person_singular_ratio_delta | 1.1921 |
| 3 | avg_type_token_ratio_roll2w | 0.7126 |
| 4 | avg_negative_delta | 0.6652 |
| 5 | new_poster_ratio_delta | 0.3385 |
| 6 | pct_very_negative_delta | 0.2952 |
| 7 | avg_flesch_kincaid_roll2w | 0.2937 |
| 8 | std_word_count | 0.2326 |
| 9 | first_person_plural_ratio_delta | 0.2102 |
| 10 | pct_very_negative | 0.2013 |

## Summary

The early warning system detected precursor signals 3 weeks before this high-distress event. Key indicators included changes in topic_shift_jsd_roll4w, first_person_singular_ratio_delta, avg_type_token_ratio_roll2w.