# Case Study: High-Distress Signal Week 2019-W37
**Week starting:** 2019-09-09
**Distress score:** 0.004

## What Happened
The community distress score spiked to 0.004, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2019-W34 (not flagged, probability: 0.18)
- **topic_shift_jsd_roll4w**: 0.0457 (56.9% below average)
- **first_person_singular_ratio_delta**: 0.0025 (5483.8% above average)
- **avg_type_token_ratio_roll2w**: 0.7275 (5.6% above average)
- **avg_negative_delta**: -0.0074 (10441.7% below average)
- **new_poster_ratio_delta**: 0.0194 (831.6% above average)

### 2019-W35 (FLAGGED, probability: 0.17)
- **topic_shift_jsd_roll4w**: 0.0541 (48.9% below average)
- **first_person_singular_ratio_delta**: -0.0012 (2840.4% below average)
- **avg_type_token_ratio_roll2w**: 0.7256 (5.4% above average)
- **avg_negative_delta**: 0.0011 (1668.5% above average)
- **new_poster_ratio_delta**: 0.0058 (318.3% above average)

### 2019-W36 (FLAGGED, probability: 0.17)
- **topic_shift_jsd_roll4w**: 0.0672 (36.5% below average)
- **first_person_singular_ratio_delta**: 0.0012 (2632.3% above average)
- **avg_type_token_ratio_roll2w**: 0.7180 (4.3% above average)
- **avg_negative_delta**: -0.0017 (2293.3% below average)
- **new_poster_ratio_delta**: -0.0602 (2167.1% below average)

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