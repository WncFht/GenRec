# RL Presets (Unified in rl.sh)

Use unified launcher:

```bash
bash Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl.sh --preset <name>
```

## Presets

| preset | reward_mode | token_level_prefix_advantage | token_adv_total_token_normalize | token_level_ndcg_error_token_penalty | probe_rule_with_zero_weight | equivalent old script |
|---|---|---:|---:|---:|---:|---|
| `prefix_token` (default) | `prefix_only` | true | false | false | true | `...-rl-prefix-token.sh` |
| `prefix_token_totalnorm` | `prefix_only` | true | true | false | true | `...-rl-prefix-token-totalnorm.sh` |
| `prefix_token_totalnorm_errtok` | `prefix_only` | true | true | true | true | `...-rl-prefix-token-totalnorm-errtok.sh` |
| `prefix_token_only` | `prefix_rule_only` | true | true | false | false | `...-rl-prefix-token-only.sh` |
| `prefix_seq_only` | `prefix_rule_only` | false | false | false | false | `...-rl-prefix-seq-only.sh` |
| `rule_only` | `rule_only` | false | false | false | false | `...-rl-rule-only.sh` |
| `ranking` | `ranking` | false | false | false | false | N/A |
| `ranking_only` | `ranking_only` | false | false | false | false | N/A |

## Notes

- You can still override any field explicitly, e.g.:
  - `--reward-mode ...`
  - `--token-level-prefix-adv true|false`
  - `--token-adv-total-token-normalize true|false`
  - `--token-level-ndcg-error-token-penalty true|false`
- `rl.sh` now always passes reward-related args explicitly to `trl_trainer.py`.
- All reward modes now log `rule_reward`; when a mode does not optimize it directly, it is attached as a zero-weight probe.
- Checkpoints policy remains:
  - `--save_total_limit` default `10`
  - `--save_only_model true`
