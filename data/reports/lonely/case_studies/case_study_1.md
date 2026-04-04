# Case Study: High-Distress Signal Week 2023-W19
**Week starting:** 2023-05-08
**Distress score:** 0.377

## What Happened
The community distress score spiked to 0.377, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2023-W16 (FLAGGED, probability: 0.50)
- **post_volume_roll2w**: 697.5000 (0.5% below average)
- **domestic_stress_total_roll2w**: 2.0000 (37.9% below average)
- **unique_posters_roll4w**: 516.0000 (6.5% below average)
- **topic_shift_jsd_4w_delta**: -0.0373 (54781.6% below average)
- **first_person_singular_ratio_roll4w**: 0.0816 (0.8% above average)

### 2023-W17 (not flagged, probability: 0.68)
- **post_volume_roll2w**: 668.5000 (4.6% below average)
- **domestic_stress_total_roll2w**: 2.5000 (22.3% below average)
- **unique_posters_roll4w**: 494.0000 (10.5% below average)
- **topic_shift_jsd_4w_delta**: 0.0076 (11336.9% above average)
- **first_person_singular_ratio_roll4w**: 0.0809 (0.0% below average)

### 2023-W18 (FLAGGED, probability: 0.48)
- **post_volume_roll2w**: 638.0000 (9.0% below average)
- **domestic_stress_total_roll2w**: 4.0000 (24.3% above average)
- **unique_posters_roll4w**: 512.0000 (7.2% below average)
- **topic_shift_jsd_4w_delta**: 0.0210 (31075.7% above average)
- **first_person_singular_ratio_roll4w**: 0.0813 (0.4% above average)

## Top Contributing Features (SHAP)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | post_volume_roll2w | 1.0466 |
| 2 | domestic_stress_total_roll2w | 0.8454 |
| 3 | unique_posters_roll4w | 0.4101 |
| 4 | topic_shift_jsd_4w_delta | 0.3611 |
| 5 | first_person_singular_ratio_roll4w | 0.3527 |
| 6 | isolation_total | 0.3433 |
| 7 | avg_compound_roll4w | 0.3345 |
| 8 | posting_time_entropy_roll4w | 0.3082 |
| 9 | suicidality_total | 0.2963 |
| 10 | avg_positive_delta | 0.2452 |

## Summary

The early warning system detected precursor signals 3 weeks before this high-distress event. Key indicators included changes in post_volume_roll2w, domestic_stress_total_roll2w, unique_posters_roll4w.