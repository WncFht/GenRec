我现在想要来重构和优化一下从 hope 脚本到 output name 和路经，到 evalute 的一些脚本和规则。注意你现在在本地环境运行，我们后面的脚本都是要放到远程去跑的，所以你要注意路经，同时不能太详细本地的一些文件。

我先讲一下：

1. 我们这个 sft 和 rl 训练要先用 /Users/fanghaotian/Desktop/src/GenRec/hope/Instruments/prepare_instruments_grec_sft_fixed_hint_data.sh 之类的脚本生成对应的 train valid test 等等，现在主要是有两套 index a) 一套是 lc-rec 的版本，之前本来是放在 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Arts 之类的位置，现在全部迁移到 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/LC-Rec/Arts 下面了。然后对应的生成的训练数据，原来放在 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index 现在也都迁移到 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/LC-Rec/Instruments_grec_index 下面去了。 b) 另一套就是现在的 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Arts 下面的内容。

```shell
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 GenRec]$ ls /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments
Instruments.index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47.json  Instruments.item2id
Instruments.index.json                                                                                     Instruments.item.json
Instruments.inter.json
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 GenRec]$ ls /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/LC-Rec/Instruments_grec_index
id2sid.json  new_tokens.json  rl  sft
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 GenRec]$ ls /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/LC-Rec/Instruments
Instruments.index.json  Instruments.inter.json  Instruments.item2id  Instruments.item.json  Instruments.user2id  Instruments.user.json
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 GenRec]$
```

2. 我们现在已经训练了一些模型出来，我们这次主要关注以后的一些训练，我打算是在 Instruments, Arts, Games 这三个数据集上面，用 genrec 的 index 来训练，已经放在 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments 下面了。然后主要做 sft, rule, ndcg, fixed, fixed_ce 这四个训练任务，使用 8 卡。3.现在已经训练了一些出来了:

```shell
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 GenRec]$ ls rl_outputs/
Arts-grec-fixed
Arts-grec-rule
Games-grec-grpo-rule-only-fixedhint-taskfix-b16-sft896
Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft896
ins-lc4023-rule
Instruments-grec-genrec-aligned-rule
Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495
Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495
Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495
Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-4-sft495
Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495
Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-hint-only-mixed-hintce005-sft495
Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-hint-only-mixed-sft495
Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495
Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495
Instruments-grec-lc4023-fixed
Instruments-grec-lc4023-fixed-ce
Instruments-grec-lc4023-fixed-ce005
Instruments-grec-lc4023-ndcg
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 GenRec]$ ls saves/qwen2.5-3b/full/
Arts-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu
Games-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu
Games-grec-sft-qwen4B-4-256-dsz0
Instruments-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu
Instruments-grec-hint-token-sft-fixed-b16-lr3e-4-sft495
Instruments-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu
Instruments-grec-sft-qwen4B-4-256-dsz0
```

在这里面

```shell
Arts-grec-fixed
Arts-grec-rule
ins-lc4023-rule
Instruments-grec-lc4023-fixed
Instruments-grec-lc4023-fixed-ce
Instruments-grec-lc4023-fixed-ce005
Instruments-grec-lc4023-ndcg
Arts-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu
Games-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu
Instruments-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu
```

这些是用 lc-rec 的 index 训练的，现在他们都在 LC-Rec 下面

```shell
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 GenRec]$ ls data/LC-Rec/
Arts  Arts_grec_index  Games  Games_grec_index  Instruments  Instruments_grec_index
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 GenRec]$ ls data/LC-Rec/Instruments
Instruments.index.json  Instruments.inter.json  Instruments.item2id  Instruments.item.json  Instruments.user2id  Instruments.user.json
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 GenRec]$ ls data/LC-Rec/Instruments_grec_index/
id2sid.json  new_tokens.json  rl  sft
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 GenRec]$ pwd
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
```

然后

```shell
Instruments-grec-genrec-aligned-rule
Instruments-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu
```

是用 genrec 的 index 训练的

## TODO

1. 现在有一个问题就是，我们的 hope 里面 /Users/fanghaotian/Desktop/src/GenRec/hope/Instruments 这里面的脚本是很混乱的，两种 index 的都有，我希望你把两套脚本分开来，放一个 Instruments-genrec 和 Instruments-lcrec 这样子
2. 我们的 evaluate pipeline 特别的混乱，我希望你给我一些建议， /Users/fanghaotian/Desktop/src/GenRec/hope/evaluate/evaluate.sh /Users/fanghaotian/Desktop/src/GenRec/data/eval_profile_manifest.json /Users/fanghaotian/Desktop/src/GenRec/data/eval_profile_manifest.json
3. 现在只有 /Users/fanghaotian/Desktop/src/GenRec/examples/train_full/Instruments/instruments_rec_full_sft_3b_dsz3_qwen4b_4_256_grec_genrec_aligned_8gpu.yaml 其他两个数据集的还没有

你先获取所有需要的信息，然后分析一下现状，给我讲讲你要怎么做，写到这个文件里面，不要覆盖上面的内容，写到下面

## suggestion

### 1. 我对当前现状的判断

| 维度 | 现在真实的单一事实源 | 当前问题 |
| --- | --- | --- |
| 数据 variant | `data/dataset_info.json` | 训练 YAML、manifest、evaluate 最终都会回到这里的 `file_name -> dataset key` 映射，但现在很多脚本把 `OUTPUT_DIR` / `RUN_NAME` 当成主事实源，导致名字和真实数据族脱钩。 |
| genrec / lcrec 两套 index | genrec 现在是 `data/<Dataset>` 和 `data/<Dataset>_grec_index`；lcrec 已迁到 `data/LC-Rec/<Dataset>` 和 `data/LC-Rec/<Dataset>_grec_index` | `hope/Instruments`、`hope/Arts`、`hope/Games` 里脚本仍然混用两套体系，目录层面看不出是在跑哪一套。 |
| evaluate profile | `scripts/eval_profile_manifest.py` 先读 `dataset_info.json`，再扫 `examples/train_full/*.yaml` 和 `hope/**/*.sh` 建 manifest | manifest 现在会把所有历史 shell 脚本都算进去，旧实验脚本、过时别名、临时命名都会污染评测路由。 |
| evaluate watcher | `hope/evaluate/*.sh` 实际只是薄封装，真正逻辑在 `scripts/evaluate_all_checkpoints.sh` | 外层已经在走 `ALLOW_HEURISTIC_FALLBACK=0` 的严格模式，但内层脚本还保留了大量历史 heuristic 和 fallback 路径，逻辑上是“两套系统叠在一起”。 |
| 未来主线 | 你已经明确以后主要看 `Instruments / Arts / Games` 三个数据集，优先使用 genrec index，任务集中在 `sft / rule / ndcg / fixed / fixed_ce`，默认 8 卡 | 现有目录和命名仍然以“历史跑过什么”为中心，而不是以“以后固定要维护什么矩阵”为中心。 |

我认为现在最核心的结论是：

1. `dataset_info.json` 应该继续作为数据侧单一事实源。
2. `examples/train_full/*.yaml` 应该作为 SFT 命名和输出侧单一事实源。
3. RL 和 evaluate 不应该再从整棵 `hope/` 树里猜当前活跃配置，而应该只读取“被明确声明为 canonical 的脚本 / profile”。

### 2. 当前已经暴露出来的几个具体问题

| 位置 | 现状 | 说明 |
| --- | --- | --- |
| `hope/Instruments/Qwen2_5-3B-Instruct-qwen4B-4-256-GenRecAligned-grec-sft-8.sh` | `YAML_PATH` 已经指向 genrec 8 卡 YAML，但默认 `RUN_NAME` 仍然是 `...lcrec_aligned_4gpu` 风格 | 这类漂移会直接污染日志名、W&B 名、evaluate alias。 |
| `hope/Arts/prepare_arts_grec_lcrec_aligned_data.sh` | 脚本名写的是 `lcrec_aligned`，但默认 `DATA_VARIANT` / `OUTPUT_DIR` 还是非 `LC-Rec` 路径风格 | 说明“脚本名”和“默认真实输出”已经不一致。 |
| `hope/Games/prepare_games_grec_lcrec_aligned_data.sh` | 和 Arts 一样，默认值仍然偏向非 `LC-Rec` 目录 | 同样说明 lcrec 路径迁移没有完全收口。 |
| `data/eval_profile_overrides.json` | 同时承担 legacy alias、dataset path override、手动修补 manifest 三种职责 | 文件还在工作，但角色过重，不适合继续承载新的 canonical 规则。 |
| `scripts/evaluate_all_checkpoints.sh` | 已经会在运行前重建 manifest，但脚本内部仍保留大量基于名字猜 variant 的 fallback 分支 | 这让“严格 manifest 模式”和“历史兼容 heuristic 模式”同时存在，维护成本很高。 |
| `examples/train_full` | 目前只有 `Instruments` 有 genrec-aligned 8 卡 YAML，`Arts` 和 `Games` 还缺 | 这正是你后面统一未来训练入口时的最大缺口。 |

另外，本地执行 `scripts/eval_profile_manifest.py audit` 时，大量 variant 会显示 `test_exists=false` / `index_exists=false`。这个结果更多说明“本地仓库没有远端完整数据”，不代表 manifest 设计本身错了；真正验收要在远端 `REPO_ROOT` 上做。

### 3. 我建议的目标结构

我建议把这次重构明确成 3 条主线：

| 主线 | 建议 | 原因 |
| --- | --- | --- |
| 目录拆分 | 按 `dataset + index_family` 拆开 active launcher 目录 | 让人一眼看出这套脚本对应哪种 index。 |
| 命名收口 | 所有 `output_dir` / `run_name` 显式带上 `genrec` 或 `lcrec` | 不再依赖 `4023`、`aligned`、`lc` 这类隐式历史暗号。 |
| evaluate 收口 | evaluate 只认 canonical profile，不再扫描全部历史脚本做自动猜测 | 否则旧脚本越多，评测越不可信。 |

建议的目录形态：

| 目录 | 角色 |
| --- | --- |
| `hope/Instruments-genrec/` | 以后 Instruments 的主线训练入口，默认 8 卡，任务只保留 `prepare / sft / rule / ndcg / fixed / fixed_ce` |
| `hope/Instruments-lcrec/` | 只保留复现实验和历史对照，不再作为默认入口 |
| `hope/Arts-genrec/` | 未来 Arts 主线入口 |
| `hope/Arts-lcrec/` | Arts 历史对照入口 |
| `hope/Games-genrec/` | 未来 Games 主线入口 |
| `hope/Games-lcrec/` | Games 历史对照入口 |
| `hope/archive/` | 老的 `Qwen2_5-*`、`instruments-grec-*` 这类实验脚本归档区，evaluate manifest 不再扫描它 |

如果不想一次性把 Arts / Games 也全部拆掉，最少也应该先把 `Instruments` 拆成 `Instruments-genrec` 和 `Instruments-lcrec`，因为这里最混乱，也最容易继续长出新债。

### 4. 我建议保留的内部命名规则

我建议“内部 dataset variant key”和“外部 output name”分开设计：

| 层次 | genrec | lcrec | 说明 |
| --- | --- | --- | --- |
| 内部 variant key | `Instruments_grec_index` | `Instruments_grec_index_lcrec` | 这套 key 现在已经被 `dataset_info.json` + manifest 理解，继续沿用最稳。 |
| 数据落盘路径 | `data/Instruments_grec_index/` | `data/LC-Rec/Instruments_grec_index/` | 路径显式区分 family，避免把 `lcrec` 再塞回 variant 名里当目录。 |
| SFT output name | `Instruments-grec-genrec-aligned-sft-...` | `Instruments-grec-lcrec-aligned-sft-...` | 外部名字必须能直接看出 family。 |
| RL output name | `Instruments-grec-genrec-rule-from-sft2751` 这类 | `Instruments-grec-lcrec-rule-from-sft4023` 这类 | 以后不再推荐 `lc4023-rule` 这种只有熟人看得懂的写法。 |

换句话说：

1. 内部 key 继续兼容当前 `_lcrec` 规则，少改 evaluate / dataset_info。
2. 外部 run/output 命名全面显式化，让目录名本身就能说明数据族和来源 checkpoint。

### 5. evaluate pipeline 我建议怎么收口

我建议把 evaluate 的职责重新切开：

| 层 | 建议职责 | 不该再做什么 |
| --- | --- | --- |
| `data/dataset_info.json` | 维护 `train/valid` dataset key 到数据子目录的映射 | 不负责历史 alias 兼容 |
| `examples/train_full/*.yaml` | 维护 SFT 的 `dataset / eval_dataset / output_dir / run_name` | 不再让 shell 脚本覆盖出另一套名字 |
| RL canonical scripts | 显式声明 `DATA_VARIANT_DEFAULT / MODEL_PATH / OUTPUT_DIR / RUN_NAME` | 不再让文件名和变量名互相漂移 |
| `eval_profile_manifest.json` | 只记录 canonical alias -> dataset variant | 不再承担大规模历史实验自动发现 |
| `eval_profile_overrides.json` | 仅保留 legacy alias 和少量手动兼容 | 不再往里面塞新主线规则 |

更具体地说，我建议：

1. `scripts/evaluate_all_checkpoints.sh` 的远端主用模式改成默认 strict manifest-only。
2. 历史 heuristic fallback 留一个 legacy 入口即可，不要继续作为主 watcher 的内建默认逻辑。
3. `scripts/eval_profile_manifest.py` 不再扫描整棵 `hope/**/*.sh`，而是只扫描 canonical 目录。
4. `hope/archive/`、旧实验目录、临时脚本目录应该直接从 manifest 扫描范围中排除。

否则即使你把 `hope/Instruments` 拆干净了，只要旧脚本还在 `hope/` 下面，manifest 还是会继续把它们算进去。

### 6. 未来主线训练我建议的最小配置集合

你后面明确主要维护的是 `Instruments / Arts / Games` 三个数据集，优先 genrec，默认 8 卡。那么我建议最小 canonical 集合就是下面这些：

| 数据集 | family | 任务 | 是否应有 canonical 文件 |
| --- | --- | --- | --- |
| Instruments | genrec | `prepare / sft / rule / ndcg / fixed / fixed_ce` | 是 |
| Arts | genrec | `prepare / sft / rule / ndcg / fixed / fixed_ce` | 是 |
| Games | genrec | `prepare / sft / rule / ndcg / fixed / fixed_ce` | 是 |
| Instruments | lcrec | 同上 | 保留，但标记 legacy |
| Arts | lcrec | 同上 | 保留，但标记 legacy |
| Games | lcrec | 同上 | 保留，但标记 legacy |

当前最直接缺的文件是：

1. `examples/train_full/Arts/...grec_genrec_aligned_8gpu.yaml`
2. `examples/train_full/Games/...grec_genrec_aligned_8gpu.yaml`
3. 这两个 YAML 对应的 genrec SFT launcher
4. Arts / Games 的 genrec RL canonical 脚本

### 7. 我建议的实施顺序

我建议按下面顺序改，这样风险最小：

| 阶段 | 目标 | 产物 |
| --- | --- | --- |
| Phase 1 | 先修正命名漂移，不改大逻辑 | 修正错的 `RUN_NAME` / `OUTPUT_DIR` 默认值；补齐 Arts / Games genrec 8 卡 YAML；明确 future 主线名字 |
| Phase 2 | 拆 active / legacy 目录 | 新建 `*-genrec` / `*-lcrec` 目录；把旧实验脚本挪到 `hope/archive/` |
| Phase 3 | 收口 manifest 来源 | `eval_profile_manifest.py` 只扫 canonical 目录和 YAML；`overrides` 只保留兼容项 |
| Phase 4 | 收口 evaluate 行为 | watcher 默认 strict manifest-only；把 heuristic fallback 退到 legacy 模式 |
| Phase 5 | 远端验收 | 在远端重建 manifest，跑 dry-run watcher，确认新 output name 都能正确解析到 test/index |

### 8. 如果按这个方向做，我接下来会具体怎么改

如果下一步进入真正改代码的阶段，我会按下面方式推进：

1. 先补齐 `Arts` / `Games` 的 genrec 8 卡 SFT YAML，并统一三套 genrec SFT 的命名。
2. 把 `hope/Instruments` 拆成 `hope/Instruments-genrec` 和 `hope/Instruments-lcrec`，先收口 Instruments。
3. 同步改 manifest 扫描范围，让 evaluate 只认新的 canonical 目录。
4. 再把同样的目录和命名规则推广到 `Arts` / `Games`。
5. 最后在远端做一次完整的 `build-manifest + dry-run evaluate` 验证。

我认为这样做的好处是：先把未来主线固定住，再给 legacy 留兼容壳，而不是继续在一个目录里同时维护两代约定。

## status update

### 已完成

1. 新增了两层共享入口：
   - `hope/_canonical_sft_launcher.sh`
   - `hope/_canonical_rl_launcher.sh`
2. 新建了 canonical 目录并把未来主线按 `genrec / lcrec` 分开：
   - `hope/Instruments-genrec`
   - `hope/Instruments-lcrec`
   - `hope/Arts-genrec`
   - `hope/Arts-lcrec`
   - `hope/Games-genrec`
   - `hope/Games-lcrec`
3. `Instruments / Arts / Games` 的 genrec 主线都已经补齐了 `prepare / sft / rule / ndcg / fixed / fixed_ce` 入口。
4. `Instruments / Arts / Games` 的 lcrec 侧都已经有单独目录，并且三套目录现在都补齐到 `prepare / sft / rule / ndcg / fixed / fixed_ce`。
5. 已补齐缺失的两个 genrec 8 卡 SFT YAML：
   - `examples/train_full/Arts/arts_rec_full_sft_3b_dsz3_qwen4b_4_256_grec_genrec_aligned_8gpu.yaml`
   - `examples/train_full/Games/games_rec_full_sft_3b_dsz3_qwen4b_4_256_grec_genrec_aligned_8gpu.yaml`
6. 已修正 `hope/Instruments/Qwen2_5-3B-Instruct-qwen4B-4-256-GenRecAligned-grec-sft-8.sh` 默认 `RUN_NAME` 漂移问题。
7. 新增了 `data/eval_profile_manifest_sources.json`，把 manifest 扫描范围收口到显式声明的 canonical YAML 和 canonical shell 目录。
8. `scripts/eval_profile_manifest.py` 已改成优先读取 `eval_profile_manifest_sources.json`，不再默认扫整棵 `examples/train_full` / `hope`。
9. 已重建 `data/eval_profile_manifest.json`。

### 本地验证结果

1. 运行了：
   - `python3 scripts/eval_profile_manifest.py build-manifest ...`
   - `python3 scripts/eval_profile_manifest.py resolve ...`
   - `python3 scripts/eval_profile_manifest.py audit ...`
2. 当前 manifest 已经从原先混着大量历史 variant 的状态，收口成 6 个 canonical dataset variant：
   - `Instruments_grec_index`
   - `Instruments_grec_index_lcrec`
   - `Arts_grec_index`
   - `Arts_grec_index_lcrec`
   - `Games_grec_index`
   - `Games_grec_index_lcrec`
3. 下面这些新 alias 已能被正确解析：
   - `Instruments-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu`
   - `Arts-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu`
   - `Games-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu`
   - `Instruments-grec-genrec-rule-from-sft`
   - `Arts-grec-genrec-rule-from-sft`
   - `Games-grec-genrec-rule-from-sft`
4. 本地 `audit` 里仍然显示很多 `test_exists=false / index_exists=false`，这是因为当前本地仓库没有远端完整数据目录，不代表 manifest 结构错误；最终验收仍然要在远端 `GenRec` 目录上执行。

### 还没有做的事

1. 我没有删除旧的 `hope/Instruments`、`hope/Arts`、`hope/Games` 和历史实验目录。
   - 这次是“新增 canonical 入口 + 收口 manifest 扫描范围”，不是“物理清空 legacy 脚本”。
2. 我没有删除或迁走旧的历史实验脚本。
   - 当前只是把 future canonical 入口和 manifest 扫描范围收口了，legacy 目录仍然保留在仓库里。
3. 我没有在远端实际执行训练或 evaluate watcher。
   - 本地只做了 manifest/build/resolve 级别验证。

### 我建议你在远端继续做的验收

1. 先同步本次改动到远端仓库。
2. 在远端执行：
   - `python3 scripts/eval_profile_manifest.py build-manifest --repo-root . --data-root ./data --output ./data/eval_profile_manifest.json --overrides ./data/eval_profile_overrides.json`
3. 再分别验证：
   - `python3 scripts/eval_profile_manifest.py resolve --repo-root . --data-root ./data --manifest ./data/eval_profile_manifest.json --overrides ./data/eval_profile_overrides.json --model-name Arts-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu --format json`
   - `python3 scripts/eval_profile_manifest.py resolve --repo-root . --data-root ./data --manifest ./data/eval_profile_manifest.json --overrides ./data/eval_profile_overrides.json --model-name Games-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu --format json`
   - `python3 scripts/eval_profile_manifest.py resolve --repo-root . --data-root ./data --manifest ./data/eval_profile_manifest.json --overrides ./data/eval_profile_overrides.json --model-name Instruments-grec-genrec-rule-from-sft --format json`
4. 如果这些 resolve 都正确，再去跑实际 watcher / evaluate。
