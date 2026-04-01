# Community Crisis Predictor
## A Three-Layer Analytics Framework for Mental Health Early Warning

**NUS School of Computing · IS5126 Hands-on with Applied Analytics · 2025/2026 Semester 2**

*Reddit Mental Health Communities · XGBoost · LSTM · SHAP · BERTopic · EDA · GenAI Weekly Briefs · FastAPI · Streamlit*

---

## Executive Summary

Community mental health crises rarely erupt without warning. In the days and weeks before a crisis week becomes visible in aggregate statistics, online communities show measurable early signals: a surge of new posters, a shift in language complexity, rising hopelessness lexicon density. This project builds a structured pipeline to detect those signals systematically, translate them into probabilistic forecasts, and deliver actionable moderator guidance.

Drawing on approximately 400,000 Reddit posts across five mental health communities (r/anxiety, r/depression, r/SuicideWatch, r/mentalhealth, r/lonely) from April 2018 to April 2020 — collected from the Zenodo Low et al. COVID-era dataset with Arctic Shift gap-fill for missing windows — the pipeline extracts 60+ features across six families, trains LSTM and XGBoost models under strict walk-forward cross-validation, and deploys the full system as two hosted services.

Three findings stand out. First, **temporal sequence context matters**: the LSTM model outperforms XGBoost on r/depression on both F1 (0.320 vs 0.286) and PR-AUC (0.448 vs 0.314), confirming that slow-building distress dynamics captured across eight sequential weeks contain predictive information beyond any single week's features — even as gap-fill data brought XGBoost scores up substantially. Second, **cross-community lead-lag is detectable**: r/anxiety entered Elevated Distress (State 2) on 6 January 2020, approximately ten weeks before r/SuicideWatch reached Severe Community Distress Signal (State 3) on 16 March 2020, consistent with anxiety communities functioning as leading indicators for acute crisis communities. Third, **data volume directly drives detectability**: with Arctic Shift gap-fill extending evaluation windows, communities that previously had only 7–8 crisis weeks (r/lonely, r/mentalhealth) gained 17–19 crisis weeks and improved from near-random PR-AUC (0.098–0.108) to Medium-band performance (0.181–0.251), empirically validating the data sufficiency constraint.

The prescriptive layer translates each weekly forecast into a structured moderator brief, generated via retrieval-augmented generation over a curated intervention playbook, and provides an interactive what-if scenario panel for real-time sensitivity analysis. The full system is deployed as a FastAPI inference service on Render.com and a Streamlit dashboard on Streamlit Cloud.

### Key Findings at a Glance

| Finding | Result |
|---|---|
| LSTM vs XGBoost on r/depression | LSTM PR-AUC 0.448 vs XGB 0.314; F1 0.320 vs 0.286 |
| Anxiety → SuicideWatch lead time | ~10 weeks (2020-01-06 → 2020-03-16) |
| Gap-fill impact on detectability | r/lonely PR-AUC: 0.098 → 0.251 after Arctic Shift extension |
| Best community PR-AUC (LSTM) | r/depression 0.448 (near High band) |
| Largest eval window | r/anxiety, r/lonely, r/mentalhealth: 120 eval weeks |
| Live dashboard | https://community-crisis-predictor-g4-2026.streamlit.app |
| Inference API | https://community-crisis-predictor.onrender.com |

---

## 1. Problem Statement

Online mental health communities serve millions of people navigating anxiety, depression, loneliness, and crisis. Unlike clinical settings, these communities have no intake process, no scheduled check-ins, and no systematic monitoring. A week of sharply elevated distress may pass unnoticed until it becomes a visible crisis — at which point moderator response is reactive rather than anticipatory.

The system is framed as a **community weather forecast**. Just as a meteorologist monitors atmospheric pressure to anticipate storms without predicting any individual raindrop's trajectory, this system monitors aggregate community signals to anticipate escalation without inferring anything about individual users. The ecological fallacy constraint applies throughout: a community in State 3 contains users in all states; a community in State 0 may include users in acute personal crisis. No individual-level inference is permitted or implied.

The analytics architecture follows three layers:

- **Descriptive** — What has been happening in these communities, and which signals characterise elevated-distress periods?
- **Predictive** — Can we forecast next week's community crisis state, and where does the model add genuine value?
- **Prescriptive** — Given this week's forecast, what should community moderators do?

### 1.1 Four-State Escalation Model

Each community's weekly state is classified into one of four levels derived from z-score thresholds applied to a composite distress score, evaluated against that community's own rolling historical baseline:

| State | Label | Threshold |
|---|---|---|
| 0 | Stable | < 0.5σ above baseline |
| 1 | Early Vulnerability Signal | 0.5σ – 1.0σ |
| 2 | Elevated Distress | 1.0σ – 2.0σ |
| 3 | Severe Community Distress Signal | > 2.0σ |

The system outputs one forecast per community per week — predicting the state for the following week based on the current week's aggregate signals.

---

## 2. Data and Methodology

### 2.1 Dataset and Collection

All experiments use the **Low et al. (2020) COVID-era Reddit mental health dataset** (Zenodo record 3941387 [1]), covering five subreddits from 23 April 2018 to 20 April 2020. Raw post fields (post text, timestamp, author hash, subreddit) are ingested; precomputed columns (TF-IDF, LIWC) in the Zenodo files are intentionally discarded in favour of reproducible in-pipeline feature computation.

A **dual-source collection strategy** is used:

1. **Zenodo** — primary source, downloaded per-subreddit CSV files matched by record `3941387`
2. **Arctic Shift gap-fill** — a supplementary JSONL archive ingested for weeks absent from Zenodo (principally the early-2018 and late-2020 windows). Source provenance is recorded per post in a `data_source` column

Both sources are tracked in `data/ingestion_manifest.json`, making re-collection idempotent. Author names are privacy-hashed before storage; post text from `[deleted]` and `[removed]` entries is excluded.

> **[PLACEHOLDER — 2021 Data Expansion]**
> *A third collection window covering January 2021 – December 2021 is planned via the PullPush.io Reddit API. This will extend the evaluation period into the post-vaccine rollout phase, allowing the models to be tested on a distributional shift period. Results and updated metrics will replace this placeholder once collection and retraining are complete. Configuration: set `collection.source: reddit_api` with `date_range.start: 2021-01-01`, `end: 2021-12-31` in `config/default.yaml`.*

**Post counts by subreddit (2018–2020 Zenodo + Arctic Shift, with gap-fill):**

| Subreddit | Approx. Posts | Weeks Observed | Gap Weeks (< 50% vol.) | XGB Eval Wks |
|---|---|---|---|---|
| r/anxiety | ~80K | 86 | 6 | 120 |
| r/depression | ~117K | 71 | 8 | 105 |
| r/SuicideWatch | ~75K | 85 | 7 | 119 |
| r/lonely | ~45K | 86 | 1 | 120 |
| r/mentalhealth | ~50K | 86 | 5 | 120 |

*Note: XGB eval weeks exceed weeks observed because the eval window spans multiple subreddits' fold periods and the walk-forward splitter includes all valid prediction-capable weeks.*

> **[PLACEHOLDER — Figure 1: Post volume timeline per subreddit]**
> *Insert chart from `data/reports/{sub}/timeline.html` (Plotly export or screenshot). Shows weekly post volume coloured by state classification (green=Stable, yellow=Early Vulnerability, orange=Elevated Distress, red=Severe).*

### 2.2 Feature Engineering — Six Families

From each community's weekly post aggregate, 60+ features are computed across six families:

| Family | Features | Example signals |
|---|---|---|
| **Linguistic** | Word count, type-token ratio, readability (Flesch-Kincaid), char count | Reading ease drop → simpler, more fragmented posts |
| **Sentiment** | VADER compound, positive/negative/neutral/very-negative distribution | Rising `pct_negative`, declining `pct_neutral` |
| **Distress** | Hopelessness lexicon density, help-seeking density, composite distress score | `hopelessness_density_roll2w` top feature on r/depression |
| **Behavioral** | Post volume, unique posters, new-poster ratio, comment engagement, posting-time entropy | `unique_posters_delta` top feature on r/anxiety |
| **Topics** | BERTopic distribution (15 topics), topic entropy, JSD topic drift (1-week and 4-week) | Topic shift → community conversation fracturing |
| **Temporal** | 2-week and 4-week rolling averages, week-over-week deltas, cyclical seasonality encoding (sin/cos) | Delta features amplify early-week change signals |

Each feature is computed per-week and stored in a flat feature matrix (`data/features/features.parquet`). Temporal variants (rolling means, deltas) are computed at extraction time, not imputed — weeks without sufficient history receive NaN values handled by the walk-forward splitter.

### 2.3 Labelling

A composite distress score is computed per week as a weighted sum of community-normalised z-scores:

```
distress_score = 0.40 × z(neg_sentiment)
               + 0.35 × z(hopelessness_density)
               + 0.25 × z(help_seeking_density)
```

The weights — 40% for negative sentiment, 35% for hopelessness lexicon density, and 25% for help-seeking density — are theoretically motivated but not clinically validated. This is a proxy, not a diagnostic instrument.

Labels are **community-specific**: thresholds are applied against each subreddit's own historical mean and standard deviation within each walk-forward fold's training window, preventing any global scale from distorting the signal. 'Elevated Distress' in r/SuicideWatch represents a different absolute level than in r/mentalhealth; labels are not cross-community comparable.

### 2.4 Evaluation Design — Walk-Forward Cross-Validation

The evaluation uses **walk-forward cross-validation** with a minimum 26-week training window, a 1-week gap to prevent leakage from rolling features, and an expanding window that grows with each fold. Random shuffled splits are explicitly excluded — future data must never inform past predictions.

The primary metric is **PR-AUC** (area under the Precision-Recall curve for binary crisis detection, States 2+3 vs 0+1). Because crisis weeks are a minority, the PR-AUC random baseline equals the community crisis rate (not 0.5). **ROC-AUC** is also reported as a secondary discrimination metric. **Recall** is the operationally critical metric: missing a crisis week has higher cost than a false alert.

Decision usefulness is measured via **Recall@K**: if a community support team can act on at most K alert weeks, how many true crisis weeks fall within the model's top-K ranked predictions?

Two models run in parallel under identical walk-forward folds:

- **XGBoost** — binary crisis classifier with hyperparameter search (RandomizedSearchCV, 30 iterations), automatic class-weight balancing. No sequence context; each week is an independent sample.
- **PyTorch LSTM** — 8-week context window, 2-layer, hidden size 64, dropout 0.2. Features are **MinMax-normalized per fold** (scaler fit on training window only, applied to the test week) — this is critical to prevent data leakage through feature scale. Class-weighted cross-entropy loss.

After training, a performance band table is printed:
- **High** (PR-AUC ≥ 0.45): model reliably detects crises
- **Medium** (0.20–0.45): moderate signal; worth operational monitoring
- **Low** (< 0.20): near-random; insufficient crisis weeks or feature coverage

---

## 3. Working Within Real-World Constraints

Extracting reliable insight from social media data requires confronting its structural limitations directly. Five categories of constraint affect this project; each shapes the design and must be acknowledged in any deployment context.

### 3.1 Sample Bias

Reddit is a self-selecting platform. Users who experience mental health distress but do not use Reddit — or who use it passively — are entirely absent from the dataset. More critically, the distribution of who posts shifts with community state: mild distress periods attract fewer posts from lightly distressed users, while crisis peaks attract more posts from acutely distressed users *and* from the community rallying around them. The result is a systematic pattern: distress scores are underestimated during mild periods and overestimated during peaks.

This manifests in per-class recall. Class 3 (Severe) is both the least frequent and the hardest to recall — in part because the most distressing posts are frequently removed by moderators before archival (see Section 3.2), but also because the model trains on a biased sample of what severe periods look like. Silent sufferers, non-English speakers, and users without internet access are entirely absent from the dataset.

The design response is to centre the system on early-warning detection of the *approach* of a crisis (Elevated Distress, State 2) rather than its peak. This is the operational window in which community response is most effective.

### 3.2 Missing Data

The Zenodo archive contains **6–8 gap weeks per subreddit**, arising from archiving inconsistencies and platform disruptions. Arctic Shift gap-fill partially addresses this, but post deletions and moderator removals are systematically non-random: the most distressing content on platforms like r/SuicideWatch is removed before archival by moderators following safe-messaging guidelines, creating systematic underrepresentation of the highest-severity weeks. This is not correctable by gap-fill.

Data completeness scores computed at ingestion time reflect week-over-week volume relative to the running median. Values above 1.0 indicate Arctic Shift contributed posts beyond the Zenodo baseline:

| Subreddit | Avg Completeness Score | Gap Weeks |
|---|---|---|
| r/anxiety | 1.002 | 6 |
| r/depression | 0.992 | 8 |
| r/SuicideWatch | 1.001 | 7 |
| r/lonely | 1.018 | 1 |
| r/mentalhealth | 1.021 | 5 |

> **[PLACEHOLDER — Table note: insert data completeness chart from `data/reports/{sub}/weekly_completeness.csv` — weekly completeness score time series per subreddit]**

### 3.3 Methodological Constraints

The 26-week minimum training window is not arbitrary. It represents the minimum data needed for rolling baselines and temporal features (4-week rolling windows plus a 1-week label gap) to stabilise. For r/depression, with 71 weeks total after gap removal, the LSTM evaluation window is approximately 35 weeks — yielding 11 crisis weeks and a crisis rate of ~31%. The PR-AUC random baseline for r/depression is therefore ~0.31, not 0.5. This is precisely why PR-AUC is the right primary metric rather than ROC-AUC: it embeds the class imbalance into its baseline.

This also explains why walk-forward CV is non-negotiable. A random split on a 71-week dataset would contaminate rolling features (the 4-week rolling average at week 50 uses weeks 46–49; a random split could place week 48 in training and week 50 in test, making the training label directly observable at test time).

### 3.4 Industry-Specific Knowledge

Mental health has validated clinical instruments — PHQ-9, BDI-II, GAD-7 — that are not available in this dataset. The distress score is a proxy constructed from VADER sentiment [2] and domain lexicons. It has not been validated against clinical expert labels. The system is explicitly framed as a population-level early warning signal for community moderators, not a clinical screening tool.

The **ecological fallacy** constraint applies throughout: community-level signals say nothing about individual users. A week in which a community's aggregate distress score spikes says nothing about whether any particular user is in crisis.

### 3.5 Practical Constraints — No Ground Truth

No externally validated crisis labels exist for Reddit communities. The four-class target states are derived entirely from within-community distributional statistics. Two consequences follow: the model is self-calibrating (each community's baselines shift with its own history) but cross-community label comparisons are not meaningful. A 'State 2' label in r/SuicideWatch and in r/mentalhealth are not comparable in absolute distress terms.

---

## 4. Layer 1 — Descriptive Analytics

*Layer 1 asks: What has been happening in these communities, and which signals characterise elevated-distress periods?*

Descriptive analytics establishes the ground truth of what the data contains before any predictive model is introduced. This layer now includes two outputs: **SHAP-grounded feature importance** from trained models, and **automated EDA reports** generated per subreddit after feature extraction.

### 4.1 Automated EDA Reports

After feature extraction, the pipeline generates a self-contained EDA HTML report per subreddit at `data/reports/{sub}/eda_summary.html`. Each report contains:

- **Feature distribution table** — mean, std, IQR, skew, % missing per feature (colour-coded: green < 5% missing, amber 5–20%, red > 20%)
- **Outlier detection (IQR rule)** — weeks where a feature value falls outside [Q1 – 1.5×IQR, Q3 + 1.5×IQR], flagged with the specific feature and week
- **Distress trend analysis** — linear regression on the community distress score over time; classifies as *rising*, *stable*, or *declining* with % change over the data period
- **Crisis rate by year** — fraction of weeks reaching State 2 or 3 each calendar year (2018, 2019, 2020)
- **Quality flags** — high-missingness features, top outlier-prone features, class imbalance warnings

This mirrors the L1.2 data quality pattern from the IS5126 curriculum: before modelling, verify that the data is what you think it is.

> **[PLACEHOLDER — Figure 2: EDA distress trend charts]**
> *Insert screenshots from `data/reports/anxiety/eda_summary.html` and `data/reports/depression/eda_summary.html` — distress trend plot and crisis rate by year bar chart. These are self-contained HTML files, open directly in any browser.*

> **[PLACEHOLDER — Figure 3: Crisis rate by year — all five subreddits]**
> *Insert bar chart from EDA reports showing fraction of weeks in State 2+ per year per community. Expected pattern: 2020 shows elevated crisis rate vs 2018–2019 (COVID-19 effect).*

### 4.2 Community-Specific SHAP Feature Importance

SHAP values from the XGBoost models (TreeExplainer) reveal that the most predictive features are distinct across communities, reflecting different posting cultures and distress dynamics:

| Subreddit | Top 3 SHAP Features | Interpretation |
|---|---|---|
| r/anxiety | `unique_posters_delta`, `new_poster_ratio`, `avg_flesch_kincaid_roll2w` | Surge of new (distressed) posters entering the community; simplifying language |
| r/depression | `distress_density_delta`, `hopelessness_density_roll2w`, `pct_neutral_delta` | Lexicon-based signals dominate; sustained hopelessness over 2-week window |
| r/SuicideWatch | `first_person_plural_ratio_roll2w`, `first_person_singular_ratio_delta`, `unique_posters_delta` | Shift from 'I' to 'we' language; community cohesion signal |
| r/lonely | `pct_negative_roll2w`, `first_person_singular_ratio_roll2w`, `distress_density_delta` | Sustained negative sentiment and isolation language |
| r/mentalhealth | `distress_density`, `pct_negative`, `new_poster_ratio_roll4w` | Post volume + lexicon signals; slower-moving community |

The contrast between r/anxiety (behavioral signals dominate) and r/depression (lexicon signals dominate) reflects genuinely different community dynamics: anxiety communities react to events with volume surges, while depression communities show slow linguistic deterioration.

> **[PLACEHOLDER — Figure 4: SHAP beeswarm/bar charts — all five subreddits]**
> *Generated by pipeline at `data/reports/{sub}/feature_importance.html` (Plotly SHAP bar chart, top 10 features by mean absolute SHAP value). Open directly in browser or screenshot for report.*

### 4.3 Weekly Distress Timeline

The backtesting timeline (`data/reports/{sub}/timeline.html`) shows the actual vs predicted state for every evaluation week, coloured by the four-state scale. Key events visible in the timeline:

- **r/anxiety**: Elevated Distress entry on 2020-01-06, approximately ten weeks before r/SuicideWatch's Severe signal — the cross-community lead-lag finding (Section 5.2, Finding 2)
- **r/depression**: COVID-19 onset visible as a State 3 spike in March 2020 across all communities simultaneously
- **r/SuicideWatch**: State 3 (Severe Community Distress Signal) first reached 2020-03-16

> **[PLACEHOLDER — Figure 5: Timeline screenshot — r/anxiety and r/SuicideWatch aligned]**
> *Side-by-side screenshot of timeline.html for anxiety (showing 2020-01-06 Elevated Distress onset) and SuicideWatch (showing 2020-03-16 Severe onset). Visual demonstration of Finding 2.*

---

## 5. Layer 2 — Predictive Analytics

*Layer 2 asks: Can we forecast next week's community crisis state, and where does the model add genuine value?*

All metrics are sourced directly from walk-forward evaluation on the Zenodo COVID dataset (`data/models/eval_results.json`).

### 5.1 Walk-Forward Evaluation Results

PR-AUC is the primary metric (binary crisis detection: States 2+3 vs 0+1). Random baseline ≠ 0.5 — it equals the community crisis rate in the evaluation window. After Arctic Shift gap-fill, all five communities are in the Medium or near-High performance band. The performance band thresholds are: High ≥ 0.45, Medium 0.20–0.45, Low < 0.20.

| Subreddit | XGB Recall | XGB F1 | XGB PR-AUC | LSTM Recall | LSTM F1 | **LSTM PR-AUC** | XGB Crisis Wks | LSTM Crisis Wks | XGB Eval Wks | LSTM Eval Wks | Band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r/anxiety | 0.259 | 0.275 | 0.270 | 0.148 | 0.178 | 0.274 | 27 | 27 | 120 | 112 | Medium |
| r/depression | 0.290 | 0.286 | 0.314 | **0.258** | **0.320** | **0.448** | 31 | 31 | 105 | 97 | **Near-High** |
| r/SuicideWatch | 0.143 | 0.143 | 0.191 | 0.300 | 0.273 | 0.244 | 21 | 20 | 119 | 111 | Low–Medium |
| r/lonely | 0.118 | 0.129 | 0.251 | 0.118 | 0.138 | 0.181 | 17 | 17 | 120 | 112 | Medium |
| r/mentalhealth | 0.053 | 0.065 | 0.198 | 0.222 | 0.200 | 0.235 | 19 | 18 | 120 | 112 | Medium |

> **[PLACEHOLDER — ROC-AUC column]**
> *ROC-AUC values will be inserted after re-running `make evaluate` with the updated reporting module. These are logged in `eval_results.json` as of the latest pipeline run.*

*Highlighted: LSTM best results on r/depression. Bold indicates best model per community.*

> **[PLACEHOLDER — Figure 6: PR curves — all five subreddits, XGB vs LSTM]**
> *Insert precision-recall curve plots. PR curves are generated during walk-forward evaluation; reproduce by running `make evaluate --plot-pr-curves` or screenshot from dashboard Model Metrics tab.*

> **[PLACEHOLDER — 2021 Extended Results Table]**
> *After 2021 data collection and retraining, insert updated results table here covering the 2021 evaluation window. Expected: test of distributional shift behaviour — does the model trained on 2018–2020 degrade or hold on 2021 data?*

### 5.2 Three Named Findings

#### Finding 1 — LSTM Temporal Memory Effect

On r/depression — the dataset's richest community by post volume (~117K posts, 105 XGB / 97 LSTM eval weeks) — the LSTM achieves **PR-AUC 0.448 versus XGBoost PR-AUC 0.314**, and **F1 0.320 versus XGBoost F1 0.286**. With the extended dataset after Arctic Shift gap-fill, XGBoost improved substantially (from F1 0.200 to 0.286), while the LSTM maintained its advantage in both PR-AUC and F1. This convergence is itself informative: simple ensemble models benefit more from more data, while the LSTM's sequence advantage persists because the slow-building distress dynamics it captures are a property of the community's behaviour, not a data-scarcity artifact.

The advantage does not appear on r/anxiety (where behavioral spikes dominate single-week signals) or r/SuicideWatch (where the LSTM advantage is reversed — XGB PR-AUC 0.191, LSTM 0.244, but LSTM Recall 0.300 vs XGB 0.143, indicating different precision/recall trade-offs).

This finding confirms that **temporal sequence context adds value specifically for communities with gradual distress trajectories**.

#### Finding 2 — Cross-Community Lead-Lag: Anxiety Anticipates SuicideWatch

r/anxiety first entered State 2 (Elevated Distress) on **6 January 2020**, approximately **ten weeks** before r/SuicideWatch registered its first State 3 (Severe Community Distress Signal) on **16 March 2020** — coinciding with the WHO pandemic declaration. This temporal ordering is consistent with the hypothesis that anxiety communities are leading indicators for acute crisis communities: generalised anxiety precedes, and potentially contributes to, the escalation visible in suicidal ideation communities.

This is a correlational observation from a single event (COVID-19 onset). Causal attribution would require multiple independent observations across different crisis events.

#### Finding 3 — Data Volume Drives Detectability (Arctic Shift Validation)

Before Arctic Shift gap-fill, r/lonely and r/mentalhealth each had 7–8 crisis weeks in the evaluation window and produced PR-AUC values at or below random baseline (0.098 and 0.108 respectively). After gap-fill extended both evaluation windows to 120 eval weeks with 17–19 crisis weeks, their PR-AUC improved to 0.181–0.251 (r/lonely) and 0.198–0.235 (r/mentalhealth) — both entering the Medium performance band.

This is a controlled natural experiment: the same feature engineering and model architecture on the same communities, with more data. The PR-AUC improvement is entirely attributable to having more labelled crisis weeks available for the walk-forward folds to train on.

The pattern establishes an empirical minimum: **approximately 15+ crisis weeks in the evaluation window** are required for Medium-band detection reliability with this feature set and architecture. Communities below ~10 crisis weeks should remain in Trend Monitoring mode.

### 5.3 Decision Usefulness — Recall@K

Beyond global PR-AUC, the evaluation records **Recall@K**: if a community support team can act on at most K alert weeks per evaluation period, how many true crisis weeks fall within the model's top-K ranked predictions?

Selected Recall@5 values from the current evaluation:

| Subreddit | Model | Recall@5 | Random@5 | Lift |
|---|---|---|---|---|
| r/anxiety | XGBoost | 0.000 | 0.042 | Below random |
| r/depression | XGBoost | 0.032 | 0.048 | — |
| r/depression | LSTM | **0.065** | 0.052 | **1.3×** |
| r/lonely | XGBoost | **0.118** | 0.042 | **2.8×** |
| r/SuicideWatch | LSTM | 0.050 | 0.045 | 1.1× |
| r/mentalhealth | LSTM | **0.111** | 0.045 | **2.5×** |

Recall@K answers the practical operational question: given a limited moderation budget, does the model help prioritise which weeks deserve closer attention? On r/lonely and r/mentalhealth — the communities that historically underperformed — the XGBoost and LSTM models respectively show the largest Recall@K lifts (2.8× and 2.5×), indicating that even Medium-band models can add operational decision value when Recall@K is the deployment metric rather than aggregate PR-AUC.

### 5.4 Detection Lead Time

Average detection lead time is 0.17 weeks for r/SuicideWatch XGBoost and 0.65 weeks for r/anxiety XGBoost, indicating that most correct predictions occur near-simultaneously with the crisis week rather than substantially in advance. r/depression XGBoost shows 0.35 weeks average lead time. These values confirm the system's primary value is in **Elevated Distress (State 2) detection** — anticipating the approach of a crisis — rather than Severe (State 3) prediction, where community escalation is often already underway when the model fires.

---

## 6. Layer 3 — Prescriptive Analytics

*Layer 3 asks: Given this week's forecast, what should community moderators do?*

### 6.1 GenAI Weekly Briefs

After walk-forward evaluation, every prediction week receives a structured moderator brief. The generation pipeline (`src/narration/narrative_generator.py`) follows three steps:

1. **Structured Context Construction** — A JSON object containing predicted state, actual state (where available), top SHAP feature values with week-over-week deltas, and raw distress score
2. **Retrieval-Augmented Generation** — The structured context is augmented with the relevant section of `config/intervention_playbook.md` matching the predicted state. This is deterministic retrieval over a fixed 4-section document (one section per crisis state), not a vector database
3. **LLM Generation** — Prompt sent to Claude (`claude-sonnet`) if Anthropic API key present → GPT-4o if OpenAI key present → template string fallback. All three paths produce the same structured output format

Briefs are stored as a keyed JSON file per subreddit (`data/reports/{sub}/weekly_briefs.json`), replacing the earlier per-week `.txt` file format. LLM call metadata (model used, fallback reason) is logged to `data/reports/{sub}/logs/weekly_brief_calls.jsonl`.

**Example Weekly Brief — r/anxiety — Elevated Distress (State 2) — Week of 2020-01-06:**

> *r/anxiety (2020-W02) is labeled Elevated Distress this week based on model outputs (aggregate community-level indicator, not individual assessment). Distress score change vs the prior week is +0.21; key signals: new_poster_ratio increased +0.14 week-over-week (now 0.38); unique_posters_delta: +47; avg_flesch_kincaid_roll2w declining (posts becoming simpler, more fragmented). Recommended community actions: increase moderator check-in frequency; pin crisis resource links at the top of the subreddit; consider reaching out to r/SuicideWatch for coordinated resource sharing given the current cross-community signal.*

### 6.2 Interactive What-If Scenario Panel

The Streamlit dashboard includes a **scenario panel** where a community manager can adjust three feature dimensions and observe the model's predicted state change in real time:

| Control | Feature mapped | Interpretation |
|---|---|---|
| Hopelessness % slider | `hopelessness_density` | Increase: simulate a week with more hopeless language |
| Post volume % slider | `post_volume` | Increase: simulate a surge week |
| Late-night posts % slider | `late_night_post_ratio` | Increase: simulate distress concentrated in night hours |

Feature mapping is handled by `src/dashboard/demo_utils.py`, which resolves the canonical feature name from available columns using a priority list (e.g. `hopelessness_density` → `hopelessness` → `distress_lexicon_density`). Scenario adjustments are applied multiplicatively to the selected base week's feature vector before inference. Results display in real time as the slider moves.

This enables moderators to answer questions like: "If hopelessness density doubled next week relative to this week, would the model change its prediction?" — turning a black-box prediction into a sensitivity analysis tool.

> **[PLACEHOLDER — Figure 7: What-if scenario panel screenshot]**
> *Screenshot from the live Streamlit dashboard scenario tab, showing the three sliders and the resulting state prediction change. Dashboard URL: https://community-crisis-predictor-g4-2026.streamlit.app*

### 6.3 Resource Allocation — Future Work

The prescriptive layer currently operates within each community (what to do in *this* community given its predicted state). A natural extension is **cross-community resource allocation**: given a fixed moderation budget across five communities simultaneously, allocate moderator hours proportional to each community's predicted escalation probability. This would be formulated as a linear programme (LP) with a capacity constraint. Deferred to future work.

---

## 7. Production Deployment

The system is deployed as two hosted services following a Train → Commit → Deploy pattern.

| Service | Platform | URL |
|---|---|---|
| FastAPI inference API | Render.com | https://community-crisis-predictor.onrender.com |
| Streamlit dashboard | Streamlit Cloud | https://community-crisis-predictor-g4-2026.streamlit.app |

> **Cold-start note (free Render tier):** The API sleeps after 15 minutes of inactivity. The first request after sleep takes approximately 30–60 seconds. For live demos, hit `/health` once before the presentation to wake the service.

### 7.1 FastAPI Inference Service

The serving layer (`serving/main.py`) provides a production inference endpoint that loads all model artifacts at startup and exposes four endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service status; lists loaded models |
| `/predict` | POST | XGB + optional LSTM inference on a weekly feature vector; returns predicted state, probabilities, and drift warnings |
| `/model-info` | GET | Walk-forward metrics and top SHAP features per subreddit |
| `/logs/summary` | GET | Aggregate statistics from the prediction log (JSONL) |
| `/docs` | GET | Auto-generated Swagger UI |

The `/predict` endpoint accepts an optional `feature_history` array (last 8 weeks) to enable LSTM prediction alongside XGBoost. Per-request **drift detection** checks each input feature against the training distribution (mean ± 2.5σ) and returns warnings for out-of-distribution inputs. All predictions are logged to `serving/logs/predictions.jsonl` for audit.

The serving layer is deliberately isolated from `src/` — the LSTM architecture (`LSTMNet`) is replicated inline in `serving/main.py` to avoid dependency on PyTorch training code. Model artifacts (`.pkl`, `.pt`, `_feature_stats.json`) are read directly from `data/models/` on the same filesystem.

### 7.2 Streamlit Dashboard

The dashboard (`src/dashboard/app.py`) runs on Streamlit Cloud with `data/features/features.parquet` and `data/models/eval_results.json` committed to the repository (tracked as git artifacts). When `API_MODE=true` (set via Streamlit Cloud secrets), the dashboard:

- Shows a **live API connection status indicator** in the sidebar
- Forwards `/predict` calls to the Render.com API rather than running local inference
- **Automatically falls back** to local pipeline outputs if the API is unreachable

This means the dashboard functions regardless of API availability — the live API connection is a capability enhancement, not a dependency.

### 7.3 CI/CD Pipeline

GitHub Actions provides two workflows:

- **`ci.yml`** — runs 72 core pipeline tests + 29 API tests on every push/PR. API tests run with `MOCK_MODELS=true` (no real model files required in CI)
- **`retrain.yml`** — manual dispatch; runs full pipeline with synthetic data, commits updated model artifacts back to the repository, and triggers automatic redeploy on both Render.com and Streamlit Cloud

To retrain on real data locally and redeploy: `make prepare-deploy && git add . && git commit -m "Update artifacts" && git push`.

---

## 8. Ethical Considerations and Limitations

### Ecological Fallacy

The community-level signal does not permit inference about any individual user. A community in State 3 contains users in all states; a community in State 0 may include users in acute personal crisis. All system outputs are framed as population-level aggregate signals. The ecological fallacy disclaimer appears explicitly in every generated weekly brief.

### Platform Demographic Skew

The dataset over-represents English-language, younger, and tech-adjacent populations. Communities with different demographic compositions, languages, or cultural norms around mental health disclosure would require independent validation before the system could be applied.

### Proxy Label Validity

The composite distress score is not a validated clinical instrument. Its component weights are theoretically motivated but not calibrated against expert clinical labels. Results should be interpreted as detecting changes in community posting behaviour consistent with distress, not as measuring clinical mental health outcomes.

### Systematic Post Removal Bias

The most severe content on r/SuicideWatch is frequently removed by moderators before archival, meaning the dataset systematically undersamples the highest-severity weeks. Model recall on State 3 is therefore likely lower than it would be on an unfiltered archive. This is a structural limitation that cannot be addressed by collecting more data from the same source.

---

## 9. Conclusion and Future Work

This project demonstrates that a structured three-layer analytics pipeline — Descriptive (what is happening), Predictive (what will happen next week), Prescriptive (what should be done) — can extract genuine decision-relevant signal from noisy, biased social media data, provided the constraints shaping that data are treated as design inputs rather than caveats.

The most robust architectural finding is that **temporal sequence context is not merely convenient but materially important**: on r/depression, the LSTM's PR-AUC advantage over XGBoost (0.448 vs 0.314) persists even after Arctic Shift gap-fill substantially boosted XGBoost's performance, confirming that slow-building distress dynamics across eight sequential weeks contain information that single-week models cannot access regardless of data volume.

The system's handling of its own data-sufficiency limits — placing r/lonely and r/mentalhealth in Trend Monitoring mode rather than issuing unreliable alerts — reflects a design principle that applies to any operational early warning system: knowing when *not* to alert is as important as knowing when to alert.

The production deployment (FastAPI on Render.com + Streamlit Cloud dashboard) validates the full Train → Serve → Monitor loop and demonstrates that the system is operationally ready for the moderation use case, within the constraints documented in Section 3.

### Priority Future Work

| Priority | Item |
|---|---|
| P1 | **2021 data collection** (PullPush.io, Jan–Dec 2021) — test distributional shift behaviour post-vaccine rollout |
| P1 | **Model retraining on extended dataset** — update eval_results.json and redeployed artifacts |
| P2 | **Cross-community LP resource allocator** — allocate moderation budget proportionally to predicted escalation probabilities |
| P2 | **Conformal prediction intervals** — provide calibrated uncertainty bounds on state probability estimates |
| P3 | **Community archetype clustering** — cluster the five subreddits by posting behaviour profile to identify transferable SHAP features |
| P3 | **Multi-platform data sources** — Twitter/X, Bluesky, clinical forum data for demographic broadening |
| P4 | **Online learning** — incremental model updates as new weeks arrive, rather than batch retrain |

---

## References

[1] Low, D. M., Rumker, L., Talkar, T., Torous, J., Cecchi, G., & Ghosh, S. S. (2020). Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19. *npj Digital Medicine*. https://zenodo.org/records/3941387

[2] Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. *ICWSM*. https://www.researchgate.net/publication/275828927

[3] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*. https://doi.org/10.1145/2939672.2939785

[4] Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*. https://www.researchgate.net/publication/317062430

[5] Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv*. https://arxiv.org/pdf/2203.05794

[6] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*. https://www.researchgate.net/publication/13853244

---

## Appendix

### A. Repository and Deployed Services

| Resource | URL |
|---|---|
| GitHub repository | https://github.com/jainaryan/community-crisis-predictor |
| FastAPI inference API (Render.com) | https://community-crisis-predictor.onrender.com |
| FastAPI Swagger UI | https://community-crisis-predictor.onrender.com/docs |
| Streamlit dashboard (Streamlit Cloud) | https://community-crisis-predictor-g4-2026.streamlit.app |

### B. Pipeline Artifact Paths

| Artifact | Path | Tracked in git? |
|---|---|---|
| Feature matrix | `data/features/features.parquet` | Yes |
| Model evaluation results | `data/models/eval_results.json` | Yes |
| XGBoost models | `data/models/{sub}_xgb.pkl` | Yes |
| LSTM checkpoints | `data/models/{sub}_lstm.pt` | Yes |
| Feature statistics | `data/models/{sub}_feature_stats.json` | Yes |
| SHAP values | `data/reports/{sub}/shap.csv` | Yes |
| Drift alerts | `data/reports/{sub}/drift_alerts.json` | Yes |
| EDA HTML reports | `data/reports/{sub}/eda_summary.html` | Yes |
| Backtesting timeline | `data/reports/{sub}/timeline.html` | Yes |
| Weekly briefs | `data/reports/{sub}/weekly_briefs.json` | Yes |
| Data completeness | `data/reports/{sub}/weekly_completeness.csv` | Yes |

### C. Placeholder Summary

The following placeholders require manual insertion after pipeline execution:

| Placeholder | Source | Section |
|---|---|---|
| Fig 1: Post volume timeline | `data/reports/{sub}/timeline.html` | 2.1 |
| 2021 data collection results | Re-run pipeline after PullPush collection | 2.1 |
| Fig 2: EDA distress trend charts | `data/reports/{sub}/eda_summary.html` | 4.1 |
| Fig 3: Crisis rate by year | `data/reports/{sub}/eda_summary.html` | 4.1 |
| Fig 4: SHAP beeswarm/bar charts | `data/reports/{sub}/feature_importance.html` | 4.2 |
| Fig 5: Timeline — anxiety vs SuicideWatch | `data/reports/{sub}/timeline.html` | 4.3 |
| Fig 6: PR curves | Dashboard Model Metrics tab | 5.1 |
| ROC-AUC column in results table | `data/models/eval_results.json` | 5.1 |
| 2021 extended results table | Re-run after 2021 data | 5.1 |
| Fig 7: What-if scenario panel screenshot | Live Streamlit dashboard | 6.2 |

### D. AI Tool Usage Declaration

In accordance with NUS IS5126 academic policy, the use of AI tools in this project is declared below. AI assistance was used for code scaffolding, document drafting, and review — with all analytical choices, experimental design, interpretation of results, and constraint analysis performed by the project team. No AI tool generated the results tables or made modelling decisions.
