# Case Study: High-Distress Signal Week 2022-W34
**Week starting:** 2022-08-22
**Distress score:** 0.017

## What Happened
The community distress score spiked to 0.017, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2022-W31 (FLAGGED, probability: 0.33)
- **economic_stress_total**: 194.0000 (22.5% above average)
- **suicidality_total**: 1260.0000 (22.5% above average)
- **suicidality_total_roll4w**: 1148.7500 (11.8% above average)
- **std_word_count_roll2w**: 200.5873 (1.3% above average)
- **economic_stress_total_roll4w**: 190.0000 (20.1% above average)

### 2022-W32 (not flagged, probability: 0.33)
- **economic_stress_total**: 145.0000 (8.4% below average)
- **suicidality_total**: 1028.0000 (0.0% below average)
- **suicidality_total_roll4w**: 1154.0000 (12.3% above average)
- **std_word_count_roll2w**: 201.9067 (2.0% above average)
- **economic_stress_total_roll4w**: 185.2500 (17.1% above average)

### 2022-W33 (not flagged, probability: 0.35)
- **economic_stress_total**: 175.0000 (10.5% above average)
- **suicidality_total**: 1027.0000 (0.1% below average)
- **suicidality_total_roll4w**: 1126.7500 (9.6% above average)
- **std_word_count_roll2w**: 199.4699 (0.8% above average)
- **economic_stress_total_roll4w**: 179.7500 (13.6% above average)

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