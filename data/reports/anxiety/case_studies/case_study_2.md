# Case Study: High-Distress Signal Week 2020-W15
**Week starting:** 2020-04-06
**Distress score:** 0.197

## What Happened
The community distress score spiked to 0.197, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2020-W12 (FLAGGED, probability: 0.39)
- **isolation_total_roll2w**: 158.0000 (0.1% below average)
- **post_volume**: 949.0000 (10.1% below average)
- **suicidality_total**: 40.0000 (5.8% above average)
- **new_poster_ratio_delta**: 0.0077 (750.8% above average)
- **avg_negative_delta**: -0.0034 (47189.6% below average)

### 2020-W13 (FLAGGED, probability: 0.39)
- **isolation_total_roll2w**: 173.0000 (9.4% above average)
- **post_volume**: 942.0000 (10.8% below average)
- **suicidality_total**: 40.0000 (5.8% above average)
- **new_poster_ratio_delta**: 0.0193 (1744.4% above average)
- **avg_negative_delta**: -0.0048 (66635.1% below average)

### 2020-W14 (FLAGGED, probability: 0.43)
- **isolation_total_roll2w**: 171.0000 (8.1% above average)
- **post_volume**: 930.0000 (11.9% below average)
- **suicidality_total**: 54.0000 (42.9% above average)
- **new_poster_ratio_delta**: 0.0128 (1186.5% above average)
- **avg_negative_delta**: 0.0043 (58649.0% above average)

## Top Contributing Features (SHAP)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | isolation_total_roll2w | 0.6558 |
| 2 | post_volume | 0.5988 |
| 3 | suicidality_total | 0.4467 |
| 4 | new_poster_ratio_delta | 0.3521 |
| 5 | avg_negative_delta | 0.3135 |
| 6 | isolation_total_roll4w | 0.2956 |
| 7 | suicidality_total_roll2w | 0.2764 |
| 8 | pct_neutral_roll4w | 0.2686 |
| 9 | hopelessness_density | 0.2620 |
| 10 | std_word_count_roll4w | 0.2539 |

## Summary

The early warning system detected precursor signals 3 weeks before this high-distress event. Key indicators included changes in isolation_total_roll2w, post_volume, suicidality_total.