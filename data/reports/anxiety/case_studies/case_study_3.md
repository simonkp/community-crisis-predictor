# Case Study: High-Distress Signal Week 2023-W01
**Week starting:** 2023-01-02
**Distress score:** 0.099

## What Happened
The community distress score spiked to 0.099, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2022-W50 (not flagged, probability: 0.29)
- **isolation_total_roll2w**: 120.0000 (24.1% below average)
- **post_volume**: 936.0000 (11.4% below average)
- **suicidality_total**: 25.0000 (33.9% below average)
- **new_poster_ratio_delta**: -0.0395 (3258.0% below average)
- **avg_negative_delta**: 0.0049 (67186.6% above average)

### 2022-W51 (not flagged, probability: 0.29)
- **isolation_total_roll2w**: 129.5000 (18.1% below average)
- **post_volume**: 944.0000 (10.6% below average)
- **suicidality_total**: 23.0000 (39.1% below average)
- **new_poster_ratio_delta**: 0.0337 (2962.8% above average)
- **avg_negative_delta**: 0.0022 (29959.6% above average)

### 2022-W52 (not flagged, probability: 0.29)
- **isolation_total_roll2w**: 152.0000 (3.9% below average)
- **post_volume**: 1032.0000 (2.3% below average)
- **suicidality_total**: 31.0000 (18.0% below average)
- **new_poster_ratio_delta**: -0.0088 (650.9% below average)
- **avg_negative_delta**: 0.0056 (77427.1% above average)

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