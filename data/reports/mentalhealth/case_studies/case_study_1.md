# Case Study: High-Distress Signal Week 2018-W44
**Week starting:** 2018-10-29
**Distress score:** 0.339

## What Happened
The community distress score spiked to 0.339, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2018-W41 (FLAGGED, probability: 0.29)
- **first_person_plural_ratio**: 0.0029 (30.5% above average)
- **topic_shift_jsd_4w_roll2w**: 0.1364 (16.5% above average)
- **distress_density_roll2w**: 0.0094 (3.3% below average)
- **pct_very_negative**: 0.4836 (1.1% below average)
- **first_person_plural_ratio_roll4w**: 0.0027 (21.7% above average)

### 2018-W42 (FLAGGED, probability: 0.29)
- **first_person_plural_ratio**: 0.0032 (45.2% above average)
- **topic_shift_jsd_4w_roll2w**: 0.1441 (23.1% above average)
- **distress_density_roll2w**: 0.0097 (0.8% below average)
- **pct_very_negative**: 0.4965 (1.6% above average)
- **first_person_plural_ratio_roll4w**: 0.0028 (28.9% above average)

### 2018-W43 (FLAGGED, probability: 0.29)
- **first_person_plural_ratio**: 0.0022 (1.0% below average)
- **topic_shift_jsd_4w_roll2w**: 0.1427 (21.9% above average)
- **distress_density_roll2w**: 0.0101 (3.7% above average)
- **pct_very_negative**: 0.4949 (1.2% above average)
- **first_person_plural_ratio_roll4w**: 0.0027 (25.2% above average)

## Top Contributing Features (SHAP)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | first_person_plural_ratio | 1.2667 |
| 2 | topic_shift_jsd_4w_roll2w | 1.1160 |
| 3 | distress_density_roll2w | 1.0782 |
| 4 | pct_very_negative | 0.9450 |
| 5 | first_person_plural_ratio_roll4w | 0.4199 |
| 6 | topic_shift_jsd_4w | 0.2789 |
| 7 | first_person_plural_ratio_delta | 0.2655 |
| 8 | avg_type_token_ratio_delta | 0.2574 |
| 9 | pct_very_negative_delta | 0.2314 |
| 10 | pct_negative_roll2w | 0.2039 |

## Summary

The early warning system detected precursor signals 3 weeks before this high-distress event. Key indicators included changes in first_person_plural_ratio, topic_shift_jsd_4w_roll2w, distress_density_roll2w.