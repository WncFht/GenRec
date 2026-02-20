# W&B Sidecar 评测使用说明

本文档说明如何使用以下两个脚本实现“增量评测 + 持续写入同一个 W&B eval run”：

- `eval_wandb_sidecar.py`
- `eval_wandb_sidecar.sh`

适用场景：SFT/RL 训练过程中不断产生 `checkpoint-*`，希望后台自动评测并把指标持续上传到 W&B。

## 1. 方案概览

### 1.1 你会得到什么

1. 自动检测新增 `checkpoint-*`
2. 只评测未处理的 checkpoint（增量）
3. 单模型配置对应一个长期 eval run（不是一个 checkpoint 一个 run）
4. 支持后台常驻、状态恢复、断点续跑
5. 支持指定 GPU
6. W&B 内同时记录：
   - `config`（筛选字段）
   - `history`（checkpoint_step 时间序列）
   - `summary`（best/last 指标）
   - `table`（每个 checkpoint 一行）

### 1.2 与现有评测脚本的关系

- sidecar **不替代** `evaluate_sft_3b.sh`，而是调用它做单 checkpoint 评测。
- sidecar 负责“增量检测、状态管理、W&B 上报”。

## 2. 前置依赖

### 2.1 Python 环境

1. `python3` 可用
2. 如果要上报 W&B，当前环境需安装 `wandb`

```bash
python3 -c "import wandb; print(wandb.__version__)"
```

### 2.2 路径准备

至少需要：

1. `--sft-root`：包含 `checkpoint-*` 的目录
2. `--test-data-path`
3. `--index-path`
4. `evaluate_sft_3b.sh` 可执行

## 3. 快速开始

以下命令在仓库根目录执行：`/Users/fanghaotian/Desktop/src/GenRec`

### 3.1 前台运行（推荐先用这个验证）

```bash
PYTHON_BIN=python3 bash eval_wandb_sidecar.sh run \
  --instance cb64 \
  --repo-root . \
  --sft-root /path/to/saves/Instruments-grec-sft-qwen4B-4-64-dsz0 \
  --eval-script evaluate_sft_3b.sh \
  --category Instruments_grec \
  --test-data-path /path/to/data/.../sft/test.json \
  --index-path /path/to/data/.../id2sid.json \
  --cuda-list "0" \
  --wandb-project MIMIGenRec-Eval \
  --wandb-mode online
```

### 3.2 后台常驻

```bash
PYTHON_BIN=python3 bash eval_wandb_sidecar.sh start --instance cb64 [同上参数...]
```

查看状态和日志：

```bash
bash eval_wandb_sidecar.sh status --instance cb64
bash eval_wandb_sidecar.sh tail --instance cb64
```

停止：

```bash
bash eval_wandb_sidecar.sh stop --instance cb64
```

### 3.3 只跑一轮增量后退出

```bash
PYTHON_BIN=python3 bash eval_wandb_sidecar.sh once \
  --instance cb64 \
  --repo-root . \
  --sft-root /path/to/saves/Instruments-grec-sft-qwen4B-4-64-dsz0 \
  --eval-script evaluate_sft_3b.sh \
  --category Instruments_grec \
  --test-data-path /path/to/data/.../sft/test.json \
  --index-path /path/to/data/.../id2sid.json \
  --cuda-list "0"
```

## 4. 新 checkpoint 如何被检测

sidecar 每轮会做：

1. 扫描 `sft-root` 下所有 `checkpoint-*`
2. 解析 step（如 `checkpoint-495` -> `495`）
3. 过滤掉已经在状态文件 `processed_steps` 的 checkpoint
4. 只处理“就绪”checkpoint：
   - 目录存在
   - 目录非空
   - 最近修改时间超过 `--checkpoint-ready-seconds`（默认 120 秒）

相关参数：

- `--poll-interval-seconds`：轮询间隔（默认 60）
- `--checkpoint-ready-seconds`：checkpoint 就绪等待时间（默认 120）
- `--max-pending-per-cycle`：每轮最多处理几个（默认 0 表示不限制）

## 5. 如何保证写入“同一个 W&B run”

### 5.1 默认机制（推荐）

sidecar 会根据模型配置生成稳定哈希 `model_config_key`，默认 run id 为：

- `eval-<model_config_key>`

并且把 `run_id` 写入状态文件。后续重启时会复用这个 `run_id`，实现持续写同一个 run。

### 5.2 手动指定 run

你也可以显式指定：

- `--wandb-run-id`
- `--wandb-run-name`

只要 `--wandb-run-id` 不变，就会持续写同一个 run（配合 `--wandb-resume allow`）。

### 5.3 注意事项

- 不要多个进程并发写同一个 run id
- 你不需要强制使用 group，也能稳定追踪

## 6. W&B 中写入了什么

### 6.1 config（用于筛选）

- `model_name`
- `dataset`
- `cb_setting`
- `seed`
- `num_beams`
- `sid_levels`
- `eval_split`

### 6.2 history（时间序列）

每个 checkpoint 会写入：

- `checkpoint_step`
- `eval/HR@1`
- `eval/HR@3`
- `eval/HR@5`
- `eval/HR@10`
- `eval/NDCG@1`
- `eval/NDCG@3`
- `eval/NDCG@5`
- `eval/NDCG@10`
- `eval/checkpoint_name`
- `eval/metrics_path`

### 6.3 summary（Run 列表对比）

自动维护：

- `best_hr@10`
- `best_ndcg@10`
- `best_step_for_hr@10`
- `best_step_for_ndcg@10`
- `last_hr@10`
- `last_ndcg@10`
- `last_step`

### 6.4 table（远程台账）

`eval/checkpoint_table` 每行对应一个 checkpoint，列包括：

- `checkpoint_step`
- `checkpoint_name`
- `HR@1/3/5/10`
- `NDCG@1/3/5/10`
- `metrics_path`

## 7. 环境变量与参数

## 7.1 管理脚本 `eval_wandb_sidecar.sh` 的环境变量

- `PYTHON_BIN`：Python 命令，默认 `python`
- `SIDECAR_PY`：sidecar Python 脚本路径，默认 `./eval_wandb_sidecar.py`
- `MANAGER_DIR`：管理日志/PID 目录，默认 `./log/eval_sidecar`
- `INSTANCE`：实例名（也可通过 `--instance` 指定）

## 7.2 `eval_wandb_sidecar.py` 常用参数

基础路径：

- `--repo-root`
- `--sft-root`（必填）
- `--eval-script`
- `--test-data-path`
- `--index-path`

评测相关：

- `--category`
- `--cuda-list`
- `--python-bin`
- `--batch-size`
- `--max-new-tokens`
- `--num-beams`
- `--temperature`
- `--do-sample`
- `--length-penalty`
- `--sid-levels`
- `--seed`

增量/轮询：

- `--poll-interval-seconds`
- `--checkpoint-ready-seconds`
- `--max-pending-per-cycle`
- `--force-eval`
- `--once`
- `--state-dir`

W&B：

- `--disable-wandb`
- `--wandb-project`
- `--wandb-entity`
- `--wandb-run-id`
- `--wandb-run-name`
- `--wandb-mode`（`online/offline/disabled`）
- `--wandb-resume`（默认 `allow`）
- `--wandb-job-type`（默认 `eval`）

### 7.3 传递给 `evaluate_sft_3b.sh` 的环境变量

sidecar 在执行单 checkpoint 评测时会注入：

- `CATEGORY`
- `CUDA_LIST`
- `PYTHON_BIN`
- `TEST_DATA_PATH`
- `INDEX_PATH`
- `BATCH_SIZE`
- `MAX_NEW_TOKENS`
- `NUM_BEAMS`
- `TEMPERATURE`
- `DO_SAMPLE`
- `LENGTH_PENALTY`
- `SID_LEVELS`
- `CKPT_LIST`（固定为当前待评测 checkpoint）

## 8. 本地状态与目录说明

默认状态目录：`state/wandb_eval_sidecar`

每个模型配置会生成：

1. 状态文件：`<model>_<hash>.json`
2. 锁文件：`<model>_<hash>.lock`

状态文件关键字段：

- `processed_steps`：已完成 checkpoint step
- `failed_steps`：失败记录（含时间和错误）
- `table_rows`：已收集指标行
- `run_id` / `run_name`
- `last_seen_step`
- `last_update_time`

## 9. 常见运维操作

### 9.1 切换 GPU

直接改 `--cuda-list`，例如：

- 单卡：`--cuda-list "1"`
- 多卡评测：`--cuda-list "0 1 2 3"`

### 9.2 重新处理已完成 checkpoint

加 `--force-eval`，会忽略已有 `metrics.json` 并重新跑评测。

### 9.3 只补录已有 metrics，不重新评测

不加 `--force-eval`，若 `results/<model_parent>/<checkpoint>/metrics.json` 已存在，sidecar 会直接读取并上报。

### 9.4 在线/离线上报

- 在线：`--wandb-mode online`
- 离线：`--wandb-mode offline`，后续可 `wandb sync`
- 完全关闭：`--disable-wandb`

## 10. 推荐的项目组织

建议把训练和评测分项目：

1. 训练：`MIMIGenRec-Train`
2. 评测：`MIMIGenRec-Eval`

评测项目中每个模型配置一个长期 run，便于：

1. 单模型看 checkpoint 曲线
2. 多模型同图叠加比较 `eval/HR@10` 或 `eval/NDCG@10`
3. 在 Runs 列表按 `best_hr@10` / `best_ndcg@10` 排序

## 11. 排障

### 11.1 `python: command not found`

设置 `PYTHON_BIN=python3`，或传 `--python-bin python3`。

### 11.2 `wandb is not installed`

安装 wandb，或先加 `--disable-wandb` 验证本地流程。

### 11.3 评测脚本路径错误

确认 `--repo-root` 与 `--eval-script` 组合后能定位到 `evaluate_sft_3b.sh`。

### 11.4 重复启动

sidecar 内部有锁文件机制；管理脚本也有 PID 检查。建议固定 `--instance` 命名（如 `cb64`、`cb128`）。

### 11.5 为什么没有发现新 checkpoint

检查：

1. checkpoint 命名是否符合 `checkpoint-<数字>`
2. 是否目录为空
3. 是否尚未超过 `--checkpoint-ready-seconds`
4. 是否已在 `processed_steps` 中

---

如果你后续希望把这套命令固化到 `hope/...-evaluate.sh`，可以在该脚本中封装固定参数，再通过 `eval_wandb_sidecar.sh start` 启动即可。
