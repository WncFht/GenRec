```shell
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 Games-genrec]$ bash prepare.sh build
[INFO] MODE=build
[INFO] REPO_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data
[INFO] CATEGORY=Games
[INFO] INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games/Games.index.json
[INFO] OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index
[INFO] DATASET_SUBDIR=Games_grec_index
[INFO] DATASET_KEY_PREFIX=Games_grec_index
[INFO] DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] PYTHON_BIN=python3
[INFO] SEQ_SAMPLE=10000
[INFO] SEED=42
[INFO] SID_LEVELS=-1
[INFO] HISTORY_MAX=50
[INFO] TRAIN_ROW_ORDER=reverse
[INFO] SPLIT_STRATEGY=grec
[INFO] TRAIN_RATIO=0.8
[INFO] VALID_RATIO=0.1
[INFO] DRY_RUN=0
[CMD] env GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data CATEGORY=Games INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games/Games.index.json OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index DATASET_SUBDIR=Games_grec_index DATASET_KEY_PREFIX=Games_grec_index DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json PYTHON_BIN=python3 SEQ_SAMPLE=10000 SEED=42 SID_LEVELS=-1 HISTORY_MAX=50 TRAIN_ROW_ORDER=reverse SPLIT_STRATEGY=grec TRAIN_RATIO=0.8 VALID_RATIO=0.1 DATA_VARIANT=Games_grec_index bash /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/scripts/run_games_preprocess.sh build
[INFO] MODE=build
[INFO] GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data
[INFO] CATEGORY=Games
[INFO] CATEGORY_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games
[INFO] INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games/Games.index.json
[INFO] INDEX_STEM=index
[INFO] DATA_VARIANT=Games_grec_index
[INFO] OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index
[INFO] DATASET_SUBDIR=Games_grec_index
[INFO] DATASET_KEY_PREFIX=Games_grec_index
[INFO] DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] PYTHON_BIN=python3
[INFO] SEQ_SAMPLE=10000
[INFO] SEED=42
[INFO] SID_LEVELS=-1
[INFO] SPLIT_STRATEGY=grec
[INFO] TRAIN_RATIO=0.8
[INFO] VALID_RATIO=0.1
[INFO] DATA_VARIANT_TAG=
[INFO] RL_ONLY_TASK1=false
[INFO] RL_ONLY_TASK4=false
[INFO] RL_ONLY_TASK5=false
[STEP] Build final SFT/RL dataset
[INFO] category=Games
[INFO] item_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games/Games.item.json
[INFO] inter_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games/Games.inter.json
[INFO] index_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games/Games.index.json
[INFO] split_strategy=grec
[INFO] train_row_order=reverse
[INFO] history_max=50
[INFO] sid_levels=-1
[INFO] task3_sample=-1
[INFO] staging_category_dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/\_preprocess_input/Games
[INFO] output_dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index
[INFO] dataset_subdir=Games_grec_index
[INFO] dataset_key_prefix=Games_grec_index
[INFO] dataset_info_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] generated rows: train=246737, valid=42259, test=42259
[INFO] Updated dataset_info: Games_grec_index_train, Games_grec_index_valid
[CMD] python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/preprocess_data_sft_rl.py --data_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/\_preprocess_input --category Games --output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index --seq_sample 10000 --task3_sample -1 --seed 42 --sid_levels -1 --data_source Games
Loaded 13839 items, 13839 semantic ID mappings (sid_levels=all)
Saved 814 new tokens to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/new_tokens.json
Saved id2sid (item_id -> [sid1, sid2, ...], sid_levels=all) to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/id2sid.json
Parsed train: 246737, valid: 42259, test: 42259
Task1 SidSFT: train=246737, valid=42259, test=42259, (sft, rl, train, valid, test)
Task2 SidItemFeat: 27287 samples (sft, train only)
Task3 FusionSeqRec: train=246737 (sft, history_sids->title, sample=all)
Task4 Title2Sid: train=10000 (rl, train only)
Task5 TitleDesc2Sid: 23373 samples (rl, title2sid+desc2sid)
Saved 520761 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/sft/train.json
Saved 42259 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/sft/valid.json
Saved 42259 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/sft/test.json
Saved 280110 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/rl/train.json
Saved 42259 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/rl/valid.json
Saved 42259 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/rl/test.json

Done! SFT dataset -> /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/sft, RL dataset -> /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/rl
[DONE] Output directory: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index
[DONE] Dataset keys: Games_grec_index_train, Games_grec_index_valid
[DONE] SFT train: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/sft/train.json
[DONE] RL train: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/rl/train.json
[DONE] New tokens: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index/new_tokens.json
```

```shell
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 Instruments-genrec]$ bash prepare.sh build-data
[INFO] MODE=build-data
[INFO] REPO_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data
[INFO] CATEGORY=Instruments
[INFO] INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json
[INFO] INDEX_STEM=index
[INFO] SPLIT_STRATEGY=grec
[INFO] RESOLVED_DATA_VARIANT=Instruments_grec_index
[INFO] OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000
[INFO] DATA_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/rl
[INFO] SFT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/sft
[INFO] DATASET_SUBDIR=Instruments_grec_index
[INFO] DATASET_KEY_PREFIX=Instruments_grec_index
[INFO] DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] SEQ_SAMPLE=10000
[INFO] TASK3_SAMPLE=20000
[INFO] SEED=42
[INFO] SID_LEVELS=-1
[INFO] HISTORY_MAX=50
[INFO] TRAIN_ROW_ORDER=reverse
[INFO] RL_ONLY_TASK1=false
[INFO] RL_ONLY_TASK4=false
[INFO] RL_ONLY_TASK5=false
[INFO] MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/saves/qwen2.5-3b/full/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495
[INFO] BEAM_SIZE=16
[INFO] MAX_HINT_DEPTH=3
[INFO] UNSOLVED_DEPTH=3
[INFO] TASK_NAMES=<all tasks>
[INFO] ANALYSIS_SUMMARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/temp/rl_beam_hint/Instruments_grec_index_beam16_summary.json
[INFO] ANALYSIS_DETAILS_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/temp/rl_beam_hint/Instruments_grec_index_beam16_details.json
[INFO] FIXED_HINT_MAP_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/temp/rl_beam_hint/Instruments_grec_index_beam16_fixed_hint_map.json
[INFO] DRY_RUN=0
[CMD] env GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data CATEGORY=Instruments INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000 DATASET_SUBDIR=Instruments_grec_index DATASET_KEY_PREFIX=Instruments_grec_index DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json PYTHON_BIN=python3 SEQ_SAMPLE=10000 TASK3_SAMPLE=20000 SEED=42 SID_LEVELS=-1 HISTORY_MAX=50 TRAIN_ROW_ORDER=reverse SPLIT_STRATEGY=grec TRAIN_RATIO=0.8 VALID_RATIO=0.1 DATA_VARIANT=Instruments_grec_index DATA_VARIANT_TAG= RL_ONLY_TASK1=false RL_ONLY_TASK4=false RL_ONLY_TASK5=false bash /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/scripts/run_instruments_preprocess.sh build
[INFO] MODE=build
[INFO] GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data
[INFO] CATEGORY=Instruments
[INFO] CATEGORY_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments
[INFO] INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json
[INFO] INDEX_STEM=index
[INFO] DATA_VARIANT=Instruments_grec_index
[INFO] OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000
[INFO] DATASET_SUBDIR=Instruments_grec_index
[INFO] DATASET_KEY_PREFIX=Instruments_grec_index
[INFO] DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] PYTHON_BIN=python3
[INFO] SEQ_SAMPLE=10000
[INFO] TASK3_SAMPLE=20000
[INFO] SEED=42
[INFO] SID_LEVELS=-1
[INFO] SPLIT_STRATEGY=grec
[INFO] HISTORY_MAX=50
[INFO] TRAIN_ROW_ORDER=reverse
[INFO] TRAIN_RATIO=0.8
[INFO] VALID_RATIO=0.1
[INFO] DATA_VARIANT_TAG=
[INFO] RL_ONLY_TASK1=false
[INFO] RL_ONLY_TASK4=false
[INFO] RL_ONLY_TASK5=false
[STEP] Build final SFT/RL dataset
[INFO] category=Instruments
[INFO] item_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.item.json
[INFO] inter_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.inter.json
[INFO] index_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json
[INFO] split_strategy=grec
[INFO] train_row_order=reverse
[INFO] history_max=50
[INFO] sid_levels=-1
[INFO] task3_sample=20000
[INFO] staging_category_dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/_preprocess_input/Instruments
[INFO] output_dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000
[INFO] dataset_subdir=Instruments_grec_index
[INFO] dataset_key_prefix=Instruments_grec_index
[INFO] dataset_info_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] generated rows: train=84890, valid=17112, test=17112
[INFO] Updated dataset_info: Instruments_grec_index_train, Instruments_grec_index_valid
[CMD] python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/preprocess_data_sft_rl.py --data_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/_preprocess_input --category Instruments --output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000 --seq_sample 10000 --task3_sample 20000 --seed 42 --sid_levels -1 --data_source Instruments
Loaded 6250 items, 6250 semantic ID mappings (sid_levels=all)
Saved 762 new tokens to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/new_tokens.json
Saved id2sid (item_id -> [sid1, sid2, ...], sid_levels=all) to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/id2sid.json
Parsed train: 84890, valid: 17112, test: 17112
Task1 SidSFT: train=84890, valid=17112, test=17112, (sft, rl, train, valid, test)
Task2 SidItemFeat: 12465 samples (sft, train only)
Task3 FusionSeqRec: train=20000 (sft, history_sids->title, sample=20000)
Task4 Title2Sid: train=10000 (rl, train only)
Task5 TitleDesc2Sid: 11551 samples (rl, title2sid+desc2sid)
Saved 117355 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/sft/train.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/sft/valid.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/sft/test.json
Saved 106441 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/rl/train.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/rl/valid.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/rl/test.json

Done! SFT dataset -> /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/sft, RL dataset -> /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/rl
[DONE] Output directory: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000
[DONE] Dataset keys: Instruments_grec_index_train, Instruments_grec_index_valid
[DONE] SFT train: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/sft/train.json
[DONE] RL train: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/rl/train.json
[DONE] New tokens: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max50-seq10000-task3-20000/new_tokens.json
```

```shell
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 Instruments-genrec]$ bash prepare.sh build-data
[INFO] MODE=build-data
[INFO] REPO_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data
[INFO] CATEGORY=Instruments
[INFO] INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json
[INFO] INDEX_STEM=index
[INFO] SPLIT_STRATEGY=grec
[INFO] RESOLVED_DATA_VARIANT=Instruments_grec_index
[INFO] OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000
[INFO] DATA_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/rl
[INFO] SFT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/sft
[INFO] DATASET_SUBDIR=Instruments_grec_index
[INFO] DATASET_KEY_PREFIX=Instruments_grec_index
[INFO] DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] SEQ_SAMPLE=10000
[INFO] TASK3_SAMPLE=20000
[INFO] SEED=42
[INFO] SID_LEVELS=-1
[INFO] HISTORY_MAX=20
[INFO] TRAIN_ROW_ORDER=reverse
[INFO] RL_ONLY_TASK1=false
[INFO] RL_ONLY_TASK4=false
[INFO] RL_ONLY_TASK5=false
[INFO] MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/saves/qwen2.5-3b/full/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495
[INFO] BEAM_SIZE=16
[INFO] MAX_HINT_DEPTH=3
[INFO] UNSOLVED_DEPTH=3
[INFO] TASK_NAMES=<all tasks>
[INFO] ANALYSIS_SUMMARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/temp/rl_beam_hint/Instruments_grec_index_beam16_summary.json
[INFO] ANALYSIS_DETAILS_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/temp/rl_beam_hint/Instruments_grec_index_beam16_details.json
[INFO] FIXED_HINT_MAP_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/temp/rl_beam_hint/Instruments_grec_index_beam16_fixed_hint_map.json
[INFO] DRY_RUN=0
[CMD] env GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data CATEGORY=Instruments INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000 DATASET_SUBDIR=Instruments_grec_index DATASET_KEY_PREFIX=Instruments_grec_index DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json PYTHON_BIN=python3 SEQ_SAMPLE=10000 TASK3_SAMPLE=20000 SEED=42 SID_LEVELS=-1 HISTORY_MAX=20 TRAIN_ROW_ORDER=reverse SPLIT_STRATEGY=grec TRAIN_RATIO=0.8 VALID_RATIO=0.1 DATA_VARIANT=Instruments_grec_index DATA_VARIANT_TAG= RL_ONLY_TASK1=false RL_ONLY_TASK4=false RL_ONLY_TASK5=false bash /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/scripts/run_instruments_preprocess.sh build
[INFO] MODE=build
[INFO] GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data
[INFO] CATEGORY=Instruments
[INFO] CATEGORY_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments
[INFO] INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json
[INFO] INDEX_STEM=index
[INFO] DATA_VARIANT=Instruments_grec_index
[INFO] OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000
[INFO] DATASET_SUBDIR=Instruments_grec_index
[INFO] DATASET_KEY_PREFIX=Instruments_grec_index
[INFO] DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] PYTHON_BIN=python3
[INFO] SEQ_SAMPLE=10000
[INFO] TASK3_SAMPLE=20000
[INFO] SEED=42
[INFO] SID_LEVELS=-1
[INFO] SPLIT_STRATEGY=grec
[INFO] HISTORY_MAX=20
[INFO] TRAIN_ROW_ORDER=reverse
[INFO] TRAIN_RATIO=0.8
[INFO] VALID_RATIO=0.1
[INFO] DATA_VARIANT_TAG=
[INFO] RL_ONLY_TASK1=false
[INFO] RL_ONLY_TASK4=false
[INFO] RL_ONLY_TASK5=false
[STEP] Build final SFT/RL dataset
[INFO] category=Instruments
[INFO] item_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.item.json
[INFO] inter_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.inter.json
[INFO] index_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json
[INFO] split_strategy=grec
[INFO] train_row_order=reverse
[INFO] history_max=20
[INFO] sid_levels=-1
[INFO] task3_sample=20000
[INFO] staging_category_dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/_preprocess_input/Instruments
[INFO] output_dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000
[INFO] dataset_subdir=Instruments_grec_index
[INFO] dataset_key_prefix=Instruments_grec_index
[INFO] dataset_info_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] generated rows: train=84890, valid=17112, test=17112
[INFO] Updated dataset_info: Instruments_grec_index_train, Instruments_grec_index_valid
[CMD] python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/preprocess_data_sft_rl.py --data_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/_preprocess_input --category Instruments --output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000 --seq_sample 10000 --task3_sample 20000 --seed 42 --sid_levels -1 --data_source Instruments
Loaded 6250 items, 6250 semantic ID mappings (sid_levels=all)
Saved 762 new tokens to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/new_tokens.json
Saved id2sid (item_id -> [sid1, sid2, ...], sid_levels=all) to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/id2sid.json
Parsed train: 84890, valid: 17112, test: 17112
Task1 SidSFT: train=84890, valid=17112, test=17112, (sft, rl, train, valid, test)
Task2 SidItemFeat: 12465 samples (sft, train only)
Task3 FusionSeqRec: train=20000 (sft, history_sids->title, sample=20000)
Task4 Title2Sid: train=10000 (rl, train only)
Task5 TitleDesc2Sid: 11551 samples (rl, title2sid+desc2sid)
Saved 117355 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/sft/train.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/sft/valid.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/sft/test.json
Saved 106441 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/rl/train.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/rl/valid.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/rl/test.json

Done! SFT dataset -> /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/sft, RL dataset -> /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/rl
[DONE] Output directory: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000
[DONE] Dataset keys: Instruments_grec_index_train, Instruments_grec_index_valid
[DONE] SFT train: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/sft/train.json
[DONE] RL train: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/rl/train.json
[DONE] New tokens: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3-20000/new_tokens.json
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 Instruments-genrec]$
```

```shell
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 Instruments-genrec]$ bash prepare.sh build-data
[INFO] MODE=build-data
[INFO] REPO_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data
[INFO] CATEGORY=Instruments
[INFO] INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json
[INFO] INDEX_STEM=index
[INFO] SPLIT_STRATEGY=grec
[INFO] RESOLVED_DATA_VARIANT=Instruments_grec_index
[INFO] OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1
[INFO] DATA_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/rl
[INFO] SFT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/sft
[INFO] DATASET_SUBDIR=Instruments_grec_index
[INFO] DATASET_KEY_PREFIX=Instruments_grec_index
[INFO] DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] SEQ_SAMPLE=10000
[INFO] TASK3_SAMPLE=-1
[INFO] SEED=42
[INFO] SID_LEVELS=-1
[INFO] HISTORY_MAX=20
[INFO] TRAIN_ROW_ORDER=reverse
[INFO] RL_ONLY_TASK1=false
[INFO] RL_ONLY_TASK4=false
[INFO] RL_ONLY_TASK5=false
[INFO] MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/saves/qwen2.5-3b/full/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495
[INFO] BEAM_SIZE=16
[INFO] MAX_HINT_DEPTH=3
[INFO] UNSOLVED_DEPTH=3
[INFO] TASK_NAMES=<all tasks>
[INFO] ANALYSIS_SUMMARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/temp/rl_beam_hint/Instruments_grec_index_beam16_summary.json
[INFO] ANALYSIS_DETAILS_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/temp/rl_beam_hint/Instruments_grec_index_beam16_details.json
[INFO] FIXED_HINT_MAP_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/temp/rl_beam_hint/Instruments_grec_index_beam16_fixed_hint_map.json
[INFO] DRY_RUN=0
[CMD] env GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data CATEGORY=Instruments INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1 DATASET_SUBDIR=Instruments_grec_index DATASET_KEY_PREFIX=Instruments_grec_index DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json PYTHON_BIN=python3 SEQ_SAMPLE=10000 TASK3_SAMPLE=-1 SEED=42 SID_LEVELS=-1 HISTORY_MAX=20 TRAIN_ROW_ORDER=reverse SPLIT_STRATEGY=grec TRAIN_RATIO=0.8 VALID_RATIO=0.1 DATA_VARIANT=Instruments_grec_index DATA_VARIANT_TAG= RL_ONLY_TASK1=false RL_ONLY_TASK4=false RL_ONLY_TASK5=false bash /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/scripts/run_instruments_preprocess.sh build
[INFO] MODE=build
[INFO] GENREC_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec
[INFO] DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data
[INFO] CATEGORY=Instruments
[INFO] CATEGORY_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments
[INFO] INDEX_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json
[INFO] INDEX_STEM=index
[INFO] DATA_VARIANT=Instruments_grec_index
[INFO] OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1
[INFO] DATASET_SUBDIR=Instruments_grec_index
[INFO] DATASET_KEY_PREFIX=Instruments_grec_index
[INFO] DATASET_INFO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] PYTHON_BIN=python3
[INFO] SEQ_SAMPLE=10000
[INFO] TASK3_SAMPLE=-1
[INFO] SEED=42
[INFO] SID_LEVELS=-1
[INFO] SPLIT_STRATEGY=grec
[INFO] HISTORY_MAX=20
[INFO] TRAIN_ROW_ORDER=reverse
[INFO] TRAIN_RATIO=0.8
[INFO] VALID_RATIO=0.1
[INFO] DATA_VARIANT_TAG=
[INFO] RL_ONLY_TASK1=false
[INFO] RL_ONLY_TASK4=false
[INFO] RL_ONLY_TASK5=false
[STEP] Build final SFT/RL dataset
[INFO] category=Instruments
[INFO] item_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.item.json
[INFO] inter_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.inter.json
[INFO] index_src=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments/Instruments.index.json
[INFO] split_strategy=grec
[INFO] train_row_order=reverse
[INFO] history_max=20
[INFO] sid_levels=-1
[INFO] task3_sample=-1
[INFO] staging_category_dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/_preprocess_input/Instruments
[INFO] output_dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1
[INFO] dataset_subdir=Instruments_grec_index
[INFO] dataset_key_prefix=Instruments_grec_index
[INFO] dataset_info_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json
[INFO] generated rows: train=84890, valid=17112, test=17112
[INFO] Updated dataset_info: Instruments_grec_index_train, Instruments_grec_index_valid
[CMD] python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/preprocess_data_sft_rl.py --data_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/_preprocess_input --category Instruments --output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1 --seq_sample 10000 --task3_sample -1 --seed 42 --sid_levels -1 --data_source Instruments
Loaded 6250 items, 6250 semantic ID mappings (sid_levels=all)
Saved 762 new tokens to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/new_tokens.json
Saved id2sid (item_id -> [sid1, sid2, ...], sid_levels=all) to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/id2sid.json
Parsed train: 84890, valid: 17112, test: 17112
Task1 SidSFT: train=84890, valid=17112, test=17112, (sft, rl, train, valid, test)
Task2 SidItemFeat: 12465 samples (sft, train only)
Task3 FusionSeqRec: train=84890 (sft, history_sids->title, sample=all)
Task4 Title2Sid: train=10000 (rl, train only)
Task5 TitleDesc2Sid: 11551 samples (rl, title2sid+desc2sid)
Saved 182245 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/sft/train.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/sft/valid.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/sft/test.json
Saved 106441 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/rl/train.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/rl/valid.json
Saved 17112 samples to /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/rl/test.json

Done! SFT dataset -> /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/sft, RL dataset -> /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/rl
[DONE] Output directory: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1
[DONE] Dataset keys: Instruments_grec_index_train, Instruments_grec_index_valid
[DONE] SFT train: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/sft/train.json
[DONE] RL train: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/rl/train.json
[DONE] New tokens: /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Instruments_grec_index-max20-seq10000-task3--1/new_tokens.json
(genrec) [hadoop-hmart-poistar@set-zw04-mlp-codelab-pc241 Instruments-genrec]$
```
