# 2026-04-02 Instruments 数据集复现 LC-Rec（远程训练机版）

- 记录日期：2026-04-02
- 更新日期：2026-04-03
- 记录目的：把 `Instruments` 数据集上复现 `LC-Rec` 的实际操作统一成远程训练机路径版本，覆盖 `raw data -> LC-Rec 训练输入 -> 远程 Qwen 训练 -> GenRec grec 评测`。
- 适用环境：远程训练机 `set-zw04-mlp-codelab-pc417` 上的 `genrec` conda 环境。

## 一句话结论

在远程机器上复现 `LC-Rec`，训练端需要的仍然是 `Instruments.item.json`、`Instruments.inter.json` 和 `Instruments.index.json` 三件套；评测端则直接复用远程已经存在的：

- `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47/sft/test.json`
- `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47/id2sid.json`

也就是说：

1. `LC-Rec` 用 raw `item/inter/index` 训练
2. `GenRec` 用现成 `grec` variant 做统一评测

这两边的 split 语义是对齐的，可以直接拼起来用。

## 1. 统一远程路径约定

以下命令统一假设：

```bash
export HOME_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
export GENREC_ROOT=$HOME_ROOT/GenRec
export DATA_ROOT=$HOME_ROOT/data
export LCREC_ROOT=$HOME_ROOT/LC-Rec
export BASE_MODEL=$HOME_ROOT/ckpt/base_model/Qwen2.5-3B-Instruct

export CATEGORY=Instruments
export GREC_VARIANT=Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47
export GREC_DATA_DIR=$GENREC_ROOT/data/$GREC_VARIANT
```

如果你的远程 `LC-Rec` checkout 不在 `$HOME_ROOT/LC-Rec`，只改：

```bash
export LCREC_ROOT=/your/actual/path/to/LC-Rec
```

## 2. 训练和评测分别依赖什么数据

### 2.1 LC-Rec 训练端真正依赖的三件套

如果只跑这四个任务：

- `seqrec`
- `item2index`
- `index2item`
- `fusionseqrec`

那么 `LC-Rec` 训练真正依赖的是：

- `Instruments.inter.json`
- `Instruments.item.json`
- `Instruments.index.json`

这里不需要 `*.user.json`。

### 2.2 GenRec 评测端真正依赖的两件套

`GenRec/scripts/evaluate_all_checkpoints.sh` 在 `Instruments-grec*` 场景下真正需要的是：

- `sft/test.json`
- `id2sid.json`

你刚确认远程已经有完整目录：

```text
$GREC_DATA_DIR
├── id2sid.json
├── new_tokens.json
├── rl/
└── sft/
    ├── test.json
    ├── train.json
    └── valid.json
```

所以评测端不需要重新生成。

## 3. 为什么可以直接把 LC-Rec 训练和 GenRec grec 评测拼起来

因为两边的 split 语义是一致的，都是 leave-2-out：

- train：每个用户序列的 `[:-2]`
- valid：目标是 `[-2]`
- test：目标是 `[-1]`

需要注意的只有一件事：

- `LC-Rec` 和 `GenRec/scripts/prepare_category_from_inter_json.py --split-strategy grec` 的 train 样本枚举顺序不同
- 但 train 样本集合相同，valid/test 目标相同

所以：

- 用 `LC-Rec` 从 `Instruments.inter.json` 直接做训练
- 再用 `GenRec` 的 `Instruments_grec_*` 目录做测试

这条链路是成立的。

## 4. 先检查远程 raw 数据是否齐全

`LC-Rec` 训练不能只靠 `GenRec/data/$GREC_VARIANT`，因为那个目录里没有 raw：

- `Instruments.item.json`
- `Instruments.inter.json`

训练前先检查远程 raw 目录：

```bash
ls $DATA_ROOT/$CATEGORY
tree $DATA_ROOT/$CATEGORY | head -n 40
```

你至少需要在 `$DATA_ROOT/$CATEGORY` 下看到：

- `Instruments.item.json`
- `Instruments.inter.json`
- 一个可用的 `Instruments.index*.json`

推荐直接使用和当前 grec variant 对应的那个 index：

```bash
export INDEX_PATH=$DATA_ROOT/$CATEGORY/Instruments.index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47.json
```

如果这个文件名在远程实际不同，就把 `INDEX_PATH` 改成真实路径。

## 5. 把数据摆成 LC-Rec 训练能读的目录结构

进入远程 `LC-Rec` 仓库后执行：

```bash
mkdir -p $LCREC_ROOT/data/Instruments

cp $DATA_ROOT/$CATEGORY/Instruments.item.json \
  $LCREC_ROOT/data/Instruments/

cp $DATA_ROOT/$CATEGORY/Instruments.inter.json \
  $LCREC_ROOT/data/Instruments/

cp $INDEX_PATH \
  $LCREC_ROOT/data/Instruments/Instruments.index.json
```

摆完之后目录应为：

```text
$LCREC_ROOT/data/Instruments/
├── Instruments.index.json
├── Instruments.inter.json
└── Instruments.item.json
```

## 6. 在远程 LC-Rec 上训练 Qwen2.5-3B

### 6.1 为什么输出目录名建议以 `Instruments-grec` 开头

因为后面要用：

- `GenRec/scripts/evaluate_all_checkpoints.sh`

这个脚本会按模型目录名做数据路由。为了让模型名和 `Instruments-grec` 这套评测数据直观对应，建议输出目录名显式带上：

- `Instruments-grec`

### 6.2 建议训练命令

```bash
cd $LCREC_ROOT

export OUTPUT_DIR=$LCREC_ROOT/ckpt/Instruments-grec-qwen2.5-3b

torchrun --nproc_per_node=8 --master_port=33324 finetune.py \
  --base_model "$BASE_MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --dataset Instruments \
  --data_path "$LCREC_ROOT/data" \
  --per_device_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --epochs 4 \
  --weight_decay 0.01 \
  --save_and_eval_strategy epoch \
  --deepspeed ./config/ds_z3_bf16.json \
  --bf16 \
  --only_train_response \
  --tasks seqrec,item2index,index2item,fusionseqrec \
  --train_prompt_sample_num 1,1,1,1 \
  --train_data_sample_num 0,0,0,20000 \
  --index_file .index.json
```

说明：

- `dataset Instruments` 对应的是 `$LCREC_ROOT/data/Instruments`
- `index_file .index.json` 会让 `LC-Rec` 读取 `Instruments.index.json`
- 这里只保留 `seqrec,item2index,index2item,fusionseqrec`
- `fusionseqrec` 采样仍沿用 `LC-Rec` 原始脚本里的 `20000`
- 如果你想全量训练，可以改成：
  - `--train_data_sample_num 0,0,0,0`

### 6.3 推荐日志写法

如果你想保留 shell 日志，建议用：

```bash
mkdir -p $LCREC_ROOT/log

nohup bash -lc '
  source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate genrec
  cd '"$LCREC_ROOT"'
  torchrun --nproc_per_node=8 --master_port=33324 finetune.py \
    --base_model "'"$BASE_MODEL"'" \
    --output_dir "'"$OUTPUT_DIR"'" \
    --dataset Instruments \
    --data_path "'"$LCREC_ROOT/data"'" \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --epochs 4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed ./config/ds_z3_bf16.json \
    --bf16 \
    --only_train_response \
    --tasks seqrec,item2index,index2item,fusionseqrec \
    --train_prompt_sample_num 1,1,1,1 \
    --train_data_sample_num 0,0,0,20000 \
    --index_file .index.json
' > $LCREC_ROOT/log/instruments_lc_rec_qwen2_5_3b.log 2>&1 &
```

## 7. 用远程 GenRec 脚本统一评测 LC-Rec checkpoint

### 7.1 评测思路

训练输出在：

- `$LCREC_ROOT/ckpt/Instruments-grec-qwen2.5-3b`

评测数据直接用远程现成 grec variant：

- `$GREC_DATA_DIR/sft/test.json`
- `$GREC_DATA_DIR/id2sid.json`

这里继续复用：

- `$GENREC_ROOT/scripts/evaluate_all_checkpoints.sh`

### 7.2 先 dry-run

```bash
cd $GENREC_ROOT

SFT_ROOT=$LCREC_ROOT/ckpt \
INCLUDE_SFT=1 \
INCLUDE_RL=0 \
MODEL_FILTER="Instruments-grec-qwen2.5-3b" \
AUTO_DATA_MAPPING=0 \
INSTRUMENTS_GREC_TEST_DATA_PATH=$GREC_DATA_DIR/sft/test.json \
INSTRUMENTS_GREC_INDEX_PATH=$GREC_DATA_DIR/id2sid.json \
DRY_RUN=1 \
bash scripts/evaluate_all_checkpoints.sh
```

### 7.3 正式评测

```bash
cd $GENREC_ROOT

SFT_ROOT=$LCREC_ROOT/ckpt \
INCLUDE_SFT=1 \
INCLUDE_RL=0 \
MODEL_FILTER="Instruments-grec-qwen2.5-3b" \
AUTO_DATA_MAPPING=0 \
INSTRUMENTS_GREC_TEST_DATA_PATH=$GREC_DATA_DIR/sft/test.json \
INSTRUMENTS_GREC_INDEX_PATH=$GREC_DATA_DIR/id2sid.json \
bash scripts/evaluate_all_checkpoints.sh
```

### 7.4 为什么这里仍建议显式设 `AUTO_DATA_MAPPING=0`

虽然模型名已经是 `Instruments-grec-*`，但这里的 checkpoint 不在：

- `$GENREC_ROOT/saves/...`

而是在：

- `$LCREC_ROOT/ckpt/...`

为了避免脚本去猜默认路径，最稳的是直接显式传：

- `INSTRUMENTS_GREC_TEST_DATA_PATH`
- `INSTRUMENTS_GREC_INDEX_PATH`

## 8. 最小复现顺序

如果你只想最快把远程流程打通，按下面顺序即可：

1. 在远程确认 `$DATA_ROOT/Instruments` 下 raw `item/inter/index` 齐全
2. 把三件套拷到 `$LCREC_ROOT/data/Instruments/`
3. 用 `Qwen2.5-3B-Instruct` 在远程 `LC-Rec` 里训练四个任务
4. 直接复用远程 `$GENREC_ROOT/data/$GREC_VARIANT/sft/test.json` 和 `id2sid.json`
5. 用 `$GENREC_ROOT/scripts/evaluate_all_checkpoints.sh` 评测

## 9. 当前远程已确认存在的评测目录

你已经确认这个目录在远程存在：

```text
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47
├── id2sid.json
├── new_tokens.json
├── rl
│   ├── test.json
│   ├── train.json
│   └── valid.json
└── sft
    ├── test.json
    ├── train.json
    └── valid.json
```

这意味着：

- 评测端现在已经不缺数据
- 你只需要把训练端的 `LC-Rec` 输入三件套准备好，就能直接开始跑远程复现

## 10. 如果远程 raw `Instruments.item.json` / `Instruments.inter.json` 不在

那就不能只靠 `$GREC_DATA_DIR` 完成 `LC-Rec` 训练，因为它只有：

- `id2sid.json`
- `new_tokens.json`
- `sft/*.json`
- `rl/*.json`

没有 `LC-Rec` 训练所需的 raw：

- `Instruments.item.json`
- `Instruments.inter.json`

此时需要先把 raw `Instruments` 数据同步到远程 `$DATA_ROOT/Instruments/`，再按第 5 节摆放到 `$LCREC_ROOT/data/Instruments/`。

也就是说：

- `GenRec grec variant` 只负责评测
- `LC-Rec` 训练仍然必须依赖 raw `item/inter/index`
