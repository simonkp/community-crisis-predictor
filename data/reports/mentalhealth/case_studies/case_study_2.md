# Case Study: High-Distress Signal Week 2018-W47
**Week starting:** 2018-11-19
**Distress score:** 0.983

## What Happened
The community distress score spiked to 0.983, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2018-W44 (FLAGGED, probability: 0.29)
- **first_person_plural_ratio**: 0.0029 (32.9% above average)
- **topic_shift_jsd_4w_roll2w**: 0.1441 (23.1% above average)
- **distress_density_roll2w**: 0.0105 (7.0% above average)
- **pct_very_negative**: 0.5477 (12.0% above average)
- **first_person_plural_ratio_roll4w**: 0.0028 (27.2% above average)

### 2018-W45 (FLAGGED, probability: 0.29)
- **first_person_plural_ratio**: 0.0021 (2.0% below average)
- **topic_shift_jsd_4w_roll2w**: 0.1182 (1.0% above average)
- **distress_density_roll2w**: 0.0111 (13.5% above average)
- **pct_very_negative**: 0.5869 (20.1% above average)
- **first_person_plural_ratio_roll4w**: 0.0026 (19.0% above average)

### 2018-W46 (not flagged, probability: 0.20)
- **first_person_plural_ratio**: 0.0013 (39.2% below average)
- **topic_shift_jsd_4w_roll2w**: 0.0744 (36.4% below average)
- **distress_density_roll2w**: 0.0109 (12.0% above average)
- **pct_very_negative**: 0.5422 (10.9% above average)
- **first_person_plural_ratio_roll4w**: 0.0021 (2.1% below average)

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