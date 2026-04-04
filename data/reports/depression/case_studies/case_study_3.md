# Case Study: High-Distress Signal Week 2020-W22
**Week starting:** 2020-05-25
**Distress score:** 0.444

## What Happened
The community distress score spiked to 0.444, 
exceeding the severe community distress threshold.

## Early Warning Signals

### 2020-W19 (not flagged, probability: 0.54)
- **unique_posters**: 1891.0000 (24.6% above average)
- **help_seeking_density**: 0.0034 (3.7% below average)
- **suicidality_total**: 678.0000 (30.4% above average)
- **domestic_stress_total**: 31.0000 (77.3% above average)
- **topic_shift_jsd_4w_roll4w**: 0.0277 (65.4% below average)

### 2020-W20 (not flagged, probability: 0.39)
- **unique_posters**: 1746.0000 (15.1% above average)
- **help_seeking_density**: 0.0031 (10.9% below average)
- **suicidality_total**: 646.0000 (24.3% above average)
- **domestic_stress_total**: 24.0000 (37.2% above average)
- **topic_shift_jsd_4w_roll4w**: 0.0277 (65.4% below average)

### 2020-W21 (not flagged, probability: 0.42)
- **unique_posters**: 1725.0000 (13.7% above average)
- **help_seeking_density**: 0.0031 (10.2% below average)
- **suicidality_total**: 639.0000 (22.9% above average)
- **domestic_stress_total**: 19.0000 (8.6% above average)
- **topic_shift_jsd_4w_roll4w**: 0.0488 (39.0% below average)

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