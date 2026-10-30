# Strategy tournament

Mode: **mock** - 10 trials x 60 rounds, seats rotated each trial.

> Mock mode: results describe the fake model and the economy rules, not LLM behaviour.

| strategy | mean economic profit | 95% CI | mean accuracy | bankruptcy rate |
|---|---|---|---|---|
| conservative | 65.4 | 32.8 to 98.0 | 0.584 | 0.0 |
| aggressive | 77.7 | 49.5 to 105.9 | 0.668 | 0.0 |
| cost_efficient | 143.5 | 102.4 to 184.6 | 0.7098 | 0.0 |
| accuracy_focused | 49.0 | 27.0 to 71.0 | 0.7698 | 0.0 |
| lender | 95.0 | 66.0 to 124.0 | 0.5682 | 0.0 |

- `cost_efficient` earned higher economic profit than the field average (+57.4, 95% CI +15.1 to +99.7, n=10).
- `accuracy_focused` earned lower economic profit than the field average (-37.1, 95% CI -59.3 to -15.0, n=10).