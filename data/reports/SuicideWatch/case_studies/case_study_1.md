# Case Study: High-Distress Signal Week 2020-W18
**Week starting:** 2020-04-27
**Distress score:** -0.726

## What Happened
The community distress score spiked to -0.726, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2020-W15 (not flagged, probability: 0.61)
- **economic_stress_total**: 191.0000 (20.6% above average)
- **suicidality_total**: 1148.0000 (11.6% above average)
- **suicidality_total_roll4w**: 1132.7500 (10.2% above average)
- **std_word_count_roll2w**: 183.8049 (7.2% below average)
- **economic_stress_total_roll4w**: 151.5000 (4.2% below average)

### 2020-W16 (not flagged, probability: 0.63)
- **economic_stress_total**: 137.0000 (13.5% below average)
- **suicidality_total**: 1038.0000 (0.9% above average)
- **suicidality_total_roll4w**: 1135.2500 (10.5% above average)
- **std_word_count_roll2w**: 195.5412 (1.2% below average)
- **economic_stress_total_roll4w**: 150.0000 (5.2% below average)

### 2020-W17 (not flagged, probability: 0.63)
- **economic_stress_total**: 27.0000 (82.9% below average)
- **suicidality_total**: 174.0000 (83.1% below average)
- **suicidality_total_roll4w**: 877.5000 (14.6% below average)
- **std_word_count_roll2w**: 180.7267 (8.7% below average)
- **economic_stress_total_roll4w**: 122.7500 (22.4% below average)

## Top Contributing Features (SHAP)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | economic_stress_total | 1.6149 |
| 2 | suicidality_total | 1.3776 |
| 3 | suicidality_total_roll4w | 0.8979 |
| 4 | std_word_count_roll2w | 0.3609 |
| 5 | economic_stress_total_roll4w | 0.3394 |
| 6 | hopelessness_density_delta | 0.2331 |
| 7 | suicidality_total_roll2w | 0.2208 |
| 8 | pct_negative | 0.1808 |
| 9 | new_poster_ratio | 0.1748 |
| 10 | avg_compound_delta | 0.1649 |

## Summary

The early warning system detected precursor signals 3 weeks before this high-distress event. Key indicators included changes in economic_stress_total, suicidality_total, suicidality_total_roll4w.