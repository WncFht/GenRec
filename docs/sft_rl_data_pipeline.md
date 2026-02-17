# SFT / RL 数据生成 Pipeline 说明

本文档说明 `preprocess_data_sft_rl.py` 和 `preprocess_data_sft_rl.sh` 如何把原始交互数据转换为 GenRec 的 SFT/RL 训练数据，并结合当前仓库 `data/Industrial_and_Scientific` 的实际结果给出可核对的参考。

## 1. 范围与相关文件

- 主转换脚本：`preprocess_data_sft_rl.py`
- 运行入口脚本：`preprocess_data_sft_rl.sh`
- 当前产物目录：`data/Industrial_and_Scientific`
- SFT 数据注册：`data/dataset_info.json`

## 2. 整体流程

```text
Amazon 原始评论/元数据
  -> data/amazon18_data_process.py 生成 *.item.json / *.train.inter / *.valid.inter / *.test.inter
  -> (外部 SID 构建流程，例如 MiniOneRec) 生成 *.index.json
  -> preprocess_data_sft_rl.py
       - 提取 SID 词表 new_tokens.json
       - 导出 item_id -> [sid1,sid2,...] 的 id2sid.json
       - 生成 SFT: task1/task2/task3
       - 生成 RL : task1/task4/task5
  -> data/<CATEGORY>/{sft,rl,...}
```

## 3. 输入前置与目录约定

`preprocess_data_sft_rl.py` 读取 `data_dir/category` 下以下文件：

- `<category>.item.json`
- `<category>.train.inter`
- `<category>.valid.inter`
- `<category>.test.inter`
- `<category>.index.json`

默认参数来自 `preprocess_data_sft_rl.sh`：

- `DATA_DIR=data/Amazon18`
- `CATEGORY=Industrial_and_Scientific`
- `OUTPUT_DIR=data/${CATEGORY}`
- `TASK4_SAMPLE=10000`
- `SEED=42`

说明：

- `data/amazon18_data_process.py` 会生成 `.item.json` 与 `.train|valid|test.inter` 等文件。
- `.index.json` 不由该脚本生成，需在 SID 构建阶段额外产出。

## 4. preprocess_data_sft_rl.py 的核心步骤

### 4.1 读取与基础映射

脚本会加载：

- `item.json` -> `id2title`（带标题过滤）和 `id2title_full`（缺失标题用 `Item_<id>` 回退）
- `index.json` ->
  - `id2sid`（拼接后的 SID 字符串，支持 3/4/... 层）
  - `index_raw`（SID token 数组）

并额外导出：

- `new_tokens.json`：从 `index.json` 抽取全部 SID token 并排序
- `id2sid.json`：保留 `item_id -> [sid1, sid2, ...]` 原始结构

### 4.2 解析交互序列

`parse_inter_file` 将 `.inter` 解析成：

- `history_item_ids`
- `target_item_id`

每条样本来自一行三列（tab 分割）：

- `user_id:token`
- `item_id_list:token_seq`（空格分隔历史）
- `item_id:token`（目标）

### 4.3 任务构建与数据流向

默认情况下 5 个任务全部启用。若任一 `only_taskN=True`，则仅运行被指定的任务。

| 任务 | 构造函数 | 输入 -> 输出 | 去向 |
|---|---|---|---|
| Task1 `sid_sft` | `build_seq_samples` | 历史 SID 序列 -> 下一个 SID | SFT(train/valid/test) + RL(train/valid/test) |
| Task2 `sid_item_feat` | `build_item_qa_samples` | SID<->Title 双向问答 | SFT(train only) |
| Task3 `fusion_seq` | `build_fusion_seq_samples` | 历史 SID 序列 -> 下一个 Title | SFT(train only) |
| Task4 `hisTitle2sid` | `build_hisTitle2sid_seq_samples` | 历史 Title 序列 -> 下一个 SID | RL(train only) |
| Task5 `title_desc2sid` | `build_title_desc2sid_samples` | 标题/描述 -> SID | RL(train only) |

补充细节：

- `Task4` 会受 `seq_sample` 控制（默认 10000）；其余任务默认全量。
- `Task1` 在 test split 使用不同 prompt 文案（更偏评测语气）。
- 所有输出在写盘前会按 `seed` 打乱。
- `to_rl_format` 会把 SFT 样本包装为 RL 结构：`data_source/prompt/ability/reward_model/extra_info`。

### 4.4 输出结构

目标目录（`output_dir`）结构：

```text
data/<CATEGORY>/
├── new_tokens.json
├── id2sid.json
├── sft/
│   ├── train.json
│   ├── valid.json
│   └── test.json
└── rl/
    ├── train.json
    ├── valid.json
    └── test.json
```

## 5. 字段格式说明

### 5.1 SFT 样本

字段：

- `system`
- `instruction`
- `input`
- `output`

示例语义：

- seq rec：`history SID -> next SID`
- fusion rec：`history SID -> next Title`
- item QA：`SID <-> Title`

### 5.2 RL 样本

字段：

- `data_source`
- `prompt`（chat template，含 system + user）
- `ability`（如 `seq_rec` / `seq_title2sid` / `title_desc2sid`）
- `reward_model.ground_truth`
- `extra_info`（`split/index/task`）

### 5.3 侧产物

- `new_tokens.json`：用于 SFT 配置中的 `add_tokens_list`
- `id2sid.json`：用于评测与 RL 中的 SID 恢复/约束解码

## 6. 运行方式

最常用：

```bash
bash preprocess_data_sft_rl.sh
```

等价命令：

```bash
python3 preprocess_data_sft_rl.py \
  --data_dir data/Amazon18 \
  --category Industrial_and_Scientific \
  --output_dir data/Industrial_and_Scientific \
  --seq_sample 10000 \
  --seed 42 \
  --sid_levels -1 \
  --data_source Industrial_and_Scientific
```

仅生成某一任务（示例：只跑 Task1）：

```bash
python3 preprocess_data_sft_rl.py \
  --data_dir data/Amazon18 \
  --category Industrial_and_Scientific \
  --output_dir data/Industrial_and_Scientific \
  --only_task1 True
```

调整 Task4 采样量：

```bash
python3 preprocess_data_sft_rl.py \
  --data_dir data/Amazon18 \
  --category Industrial_and_Scientific \
  --output_dir data/Industrial_and_Scientific \
  --seq_sample -1
```

## 7. 当前仓库数据的实际统计（可回归对照）

基于 `data/Industrial_and_Scientific` 当前文件：

- `new_tokens.json`：`606` 个 token
- SFT:
  - `train.json`：`79841`
  - `valid.json`：`4532`
  - `test.json`：`4533`
- RL:
  - `train.json`：`52775`
  - `valid.json`：`4532`
  - `test.json`：`4533`

RL train 中任务分布：

- `task1_sid_sft`：`36259`
- `task4_hisTitle2sid`：`10000`
- `task5_title_desc2sid`：`6516`

SFT train 中指令分布：

- `Can you predict the next possible item that the user may expect?`：`36259`
- `Can you recommend the next item for the user based on their interaction history?`：`36259`
- `Answer the question about item identification.`：`7323`

## 8. 与训练配置的衔接

SFT 训练依赖 `data/dataset_info.json` 里的注册项。当前已注册：

- `Industrial_and_Scientific_train` -> `Industrial_and_Scientific/sft/train.json`
- `Industrial_and_Scientific_valid` -> `Industrial_and_Scientific/sft/valid.json`

若更换 `CATEGORY` 或输出目录，需要同步更新 `data/dataset_info.json`。

## 9. 常见问题与排查

- 报错 `File not found: <category>.item.json/.train.inter/...`：检查 `--data_dir` 与 `--category` 拼接路径。
- 报错找不到 `.index.json`：先完成 SID 构建流程，再运行本脚本。
- RL train 样本量与预期不符：优先检查 `--seq_sample` 是否限制了 Task4。
- 希望不打乱输出顺序：当前脚本固定 `shuffle`，需改代码。

## 10. 快速校验命令

```bash
jq 'length' data/Industrial_and_Scientific/sft/train.json
jq 'length' data/Industrial_and_Scientific/rl/train.json
jq -r 'group_by(.extra_info.task)[] | "\(.[0].extra_info.task)\t\(length)"' data/Industrial_and_Scientific/rl/train.json
```
