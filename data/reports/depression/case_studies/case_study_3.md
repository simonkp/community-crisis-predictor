# Case Study: High-Distress Signal Week 2020-W14
**Week starting:** 2020-03-30
**Distress score:** 0.445

## What Happened
The community distress score spiked to 0.445, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2020-W11 (FLAGGED, probability: 0.39)
- **unique_posters**: 2093.0000 (38.0% above average)
- **help_seeking_density**: 0.0041 (16.0% above average)
- **suicidality_total**: 665.0000 (27.9% above average)
- **pct_negative_roll4w**: 0.0897 (4.5% above average)
- **domestic_stress_total**: 25.0000 (42.9% above average)

### 2020-W12 (not flagged, probability: 0.43)
- **unique_posters**: 1824.0000 (20.2% above average)
- **help_seeking_density**: 0.0038 (8.8% above average)
- **suicidality_total**: 615.0000 (18.3% above average)
- **pct_negative_roll4w**: 0.0879 (2.5% above average)
- **domestic_stress_total**: 21.0000 (20.1% above average)

### 2020-W13 (FLAGGED, probability: 0.48)
- **unique_posters**: 1869.0000 (23.2% above average)
- **help_seeking_density**: 0.0038 (7.9% above average)
- **suicidality_total**: 674.0000 (29.7% above average)
- **pct_negative_roll4w**: 0.0877 (2.3% above average)
- **domestic_stress_total**: 19.0000 (8.6% above average)

## Top Contributing Features (SHAP)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | unique_posters | 1.4664 |
| 2 | help_seeking_density | 0.6045 |
| 3 | suicidality_total | 0.4389 |
| 4 | pct_negative_roll4w | 0.3948 |
| 5 | domestic_stress_total | 0.3894 |
| 6 | isolation_total_roll2w | 0.3611 |
| 7 | week_sin | 0.3519 |
| 8 | help_seeking_density_roll4w | 0.3366 |
| 9 | pct_positive_delta | 0.3030 |
| 10 | pct_neutral_roll4w | 0.2512 |

## Summary

The early warning system detected precursor signals 3 weeks before this high-distress event. Key indicators included changes in unique_posters, help_seeking_density, suicidality_total.