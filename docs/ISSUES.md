# Issues & Future Additions

## Known Issues
*(none yet)*

## To Add / Improvements
- [ ] GRU/LSTM sequence model (train_rnn.py) for temporal dynamics
- [ ] Ensemble model combining XGBoost + RNN predictions
- [ ] Real-time monitoring mode with streaming Reddit data
- [ ] Multi-subreddit generalization testing (r/mentalhealth, r/SuicideWatch)
- [ ] SMOTE within walk-forward folds for class imbalance
- [ ] More sophisticated topic shift metrics beyond JSD
- [ ] Interactive Plotly Dash dashboard (vs static HTML)
- [ ] Automated threshold tuning to optimize recall at fixed precision floor
- [ ] Confidence intervals on predictions via bootstrapping
- [ ] Ablation study: which feature families contribute most

## Design Decisions to Revisit
- Crisis threshold at 1.5 std — may need tuning based on actual data distribution
- Distress score weights (0.4/0.35/0.25) — validate with domain experts
- BERTopic n_topics=15 — try "auto" mode after initial experiments
- Walk-forward gap of 1 week — may need 2 weeks if rolling features use 2-week windows
