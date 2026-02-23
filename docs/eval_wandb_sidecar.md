# Remote-Local Eval Upload Pipeline

本文档描述新的两段式评测指标上报流程：

1. 远端训练机只产出 `results` 并生成 manifest。
2. 本地机器读取同步下来的 `results + manifest`，以 `wandb online` 持续增量上报。

该流程用于稳定实现“同一个模型目录写同一个 W&B run”，并避免远端离线 `resume` 限制。

## 1. 变更说明

旧流程（已替换）：

- `eval_wandb_sidecar.py` 同时做 checkpoint 发现、触发评测脚本、写 W&B。

新流程：

- `prepare-manifest`：只生成上报清单。
- `upload`：只做增量上传，不调用评测脚本、不依赖 GPU。

## 2. 关键脚本

- `GenRec/eval_wandb_sidecar.py`
- `GenRec/eval_wandb_sidecar.sh`
- `GenRec/scripts/sync_results_from_remote.sh`

## 3. Manifest 规范

默认文件：`results/.wandb_eval_manifest.json`

结构：

- `version`
- `generated_at`
- `results_root`
- `models`（数组）

每个 `models[]` 项包含：

- `model_dir`（对应 `results/<model_dir>`）
- `dataset`
- `cb_setting`
- `eval_split`
- `seed`
- `num_beams`
- `sid_levels`
- `wandb_project`
- `wandb_entity`
- `wandb_run_id`
- `wandb_run_name`

默认 run id 规则：

- `eval-<stable_hash(model_dir,dataset,cb_setting,eval_split)>`

覆盖模板：

- `GenRec/config/wandb_eval_manifest_overrides.example.json`

## 4. 远端操作（训练机）

### 4.1 生成 manifest

```bash
cd /path/to/GenRec
python eval_wandb_sidecar.py prepare-manifest \
  --results-root ./results \
  --output-manifest ./results/.wandb_eval_manifest.json \
  --default-project MIMIGenRec-Eval \
  --run-id-prefix eval
```

带覆盖配置：

```bash
python eval_wandb_sidecar.py prepare-manifest \
  --results-root ./results \
  --output-manifest ./results/.wandb_eval_manifest.json \
  --overrides ./config/wandb_eval_manifest_overrides.json
```

### 4.2 打包 results（包含 manifest）

```bash
cd /path/to/GenRec
bash scripts/sync_results_from_remote.sh pack
```

`pack` 默认会先调用远端 `prepare-manifest`。

## 5. 本地操作（有网机器）

### 5.1 解包并覆盖本地 `GenRec/results`

```bash
cd /path/to/GenRec
bash scripts/sync_results_from_remote.sh unpack /path/to/result*.tar.gz
```

### 5.2 上传一次（推荐先验证）

```bash
cd /path/to/GenRec
PYTHON_BIN=python bash eval_wandb_sidecar.sh once \
  --results-root ./results \
  --manifest-path ./results/.wandb_eval_manifest.json \
  --wandb-mode online
```

### 5.3 常驻上传（持续增量）

```bash
cd /path/to/GenRec
PYTHON_BIN=python bash eval_wandb_sidecar.sh start --instance eval_uploader \
  --results-root ./results \
  --manifest-path ./results/.wandb_eval_manifest.json \
  --wandb-mode online

bash eval_wandb_sidecar.sh status --instance eval_uploader
bash eval_wandb_sidecar.sh tail --instance eval_uploader
bash eval_wandb_sidecar.sh stop --instance eval_uploader
```

## 6. CLI 参考

### 6.1 `eval_wandb_sidecar.py prepare-manifest`

常用参数：

- `--results-root`
- `--output-manifest`
- `--overrides`
- `--default-project`
- `--default-entity`
- `--default-eval-split`
- `--run-id-prefix`

### 6.2 `eval_wandb_sidecar.py upload`

常用参数：

- `--results-root`
- `--manifest-path`
- `--state-dir`（默认 `state/wandb_eval_uploader`）
- `--once`
- `--poll-interval-seconds`
- `--model-filter`（可重复或逗号分隔）
- `--disable-wandb`
- `--wandb-mode`（默认 `online`）
- `--wandb-resume`（默认 `allow`）
- `--wandb-job-type`（默认 `eval`）

### 6.3 `eval_wandb_sidecar.sh`

命令：

- `prepare-manifest`
- `run`
- `once`
- `start`
- `stop`
- `status`
- `tail`

## 7. 状态与幂等

上传器按模型维护本地状态文件（默认目录 `state/wandb_eval_uploader`）：

- `processed_steps`
- `failed_steps`
- `table_rows`
- `last_seen_step`
- `last_update_time`

重复执行 `upload --once` 时，已处理 step 不会重复写入（幂等增量）。

## 8. 常见问题

### 8.1 为什么图上看到的 Step 不是 checkpoint 编号？

上传器使用 `run.log(..., step=checkpoint_step)`，W&B 图建议把 X 轴设置为 `checkpoint_step`。

### 8.2 如何保证同一模型持续写同一 run？

- 固定 `model_dir -> wandb_run_id`（manifest 明确指定或默认稳定哈希）。
- 同一模型目录每次上传都使用同一 `wandb_run_id`。

### 8.3 本地没有想处理的模型怎么办？

使用过滤：

```bash
PYTHON_BIN=python bash eval_wandb_sidecar.sh once \
  --results-root ./results \
  --manifest-path ./results/.wandb_eval_manifest.json \
  --model-filter Instruments-grec-sft-qwen4B-4-256-dsz0
```

### 8.4 Python 3.9 兼容吗？

兼容。脚本使用 `datetime.timezone.utc`，不依赖 `datetime.UTC`。

## 9. 迁移建议

如果你之前在远端用旧 sidecar 直接写 `wandb offline`：

1. 改为远端只产出 `results`。
2. 通过同步脚本拉回本地。
3. 本地 `upload` 统一 `online` 上报。

这样可以稳定做到“同模型不同 checkpoint 增量写入同一个云端 run”。
