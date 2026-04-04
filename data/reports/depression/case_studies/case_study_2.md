# Case Study: High-Distress Signal Week 2020-W11
**Week starting:** 2020-03-09
**Distress score:** 0.402

## What Happened
The community distress score spiked to 0.402, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2020-W08 (FLAGGED, probability: 0.36)
- **unique_posters**: 2590.0000 (70.7% above average)
- **help_seeking_density**: 0.0041 (17.1% above average)
- **suicidality_total**: 892.0000 (71.6% above average)
- **domestic_stress_total**: 24.0000 (37.2% above average)
- **topic_shift_jsd_4w_roll4w**: 0.0402 (49.8% below average)

### 2020-W09 (FLAGGED, probability: 0.37)
- **unique_posters**: 2402.0000 (58.3% above average)
- **help_seeking_density**: 0.0041 (15.7% above average)
- **suicidality_total**: 789.0000 (51.8% above average)
- **domestic_stress_total**: 20.0000 (14.4% above average)
- **topic_shift_jsd_4w_roll4w**: 0.0402 (49.8% below average)

### 2020-W10 (FLAGGED, probability: 0.41)
- **unique_posters**: 2381.0000 (56.9% above average)
- **help_seeking_density**: 0.0042 (19.7% above average)
- **suicidality_total**: 795.0000 (53.0% above average)
- **domestic_stress_total**: 20.0000 (14.4% above average)
- **topic_shift_jsd_4w_roll4w**: 0.0402 (49.8% below average)

## Top Contributing Features (SHAP)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | unique_posters | 1.4831 |
| 2 | help_seeking_density | 0.5494 |
| 3 | suicidality_total | 0.4454 |
| 4 | domestic_stress_total | 0.4171 |
| 5 | topic_shift_jsd_4w_roll4w | 0.3767 |
| 6 | pct_positive_delta | 0.3700 |
| 7 | help_seeking_density_roll4w | 0.3677 |
| 8 | isolation_total_roll2w | 0.3581 |
| 9 | pct_negative_roll4w | 0.3465 |
| 10 | week_sin | 0.3244 |

## Summary

The early warning system detected precursor signals 3 weeks before this high-distress event. Key indicators included changes in unique_posters, help_seeking_density, suicidality_total.