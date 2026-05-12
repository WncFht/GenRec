```
INFO:root:Detailed Codebook Utilization:
INFO:root:  Layer 0: 0.1016 (52/512)
INFO:root:  Layer 1: 0.8574 (439/512)
INFO:root:  Layer 2: 0.9883 (506/512)
INFO:root:  Layer 3: 1.0000 (512/512)
INFO:root:Saving current: ./index_train_runs/Arts/index/qwen3-embedding-4B/rq4_cb512-512-512-512_sk0.0-0.0-0.0-0.003_kmtrue-lkmtrue-kmi100/May-12-2026_04-09-00/best_loss_model.pth
INFO:root:epoch 499 evaluating [time: 2.36s, collision_rate: 0.0499, avg_utilization: 0.7368]
INFO:root:Saving current: ./index_train_runs/Arts/index/qwen3-embedding-4B/rq4_cb512-512-512-512_sk0.0-0.0-0.0-0.003_kmtrue-lkmtrue-kmi100/May-12-2026_04-09-00/epoch_499_collision_0.0499_util_0.7368_model.pth
Best Loss 15.952368259429932
Best Collision Rate 0.044604927782497875
```

```
INFO:root:Detailed Codebook Utilization:
INFO:root:  Layer 0: 0.1328 (34/256)
INFO:root:  Layer 1: 0.9023 (231/256)
INFO:root:  Layer 2: 1.0000 (256/256)
INFO:root:  Layer 3: 1.0000 (256/256)
INFO:root:Saving current: ./index_train_runs/Arts/index/qwen3-embedding-4B/rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003_kmtrue-lkmtrue-kmi100/May-12-2026_04-36-38/best_loss_model.pth
INFO:root:Saving current: ./index_train_runs/Arts/index/qwen3-embedding-4B/rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003_kmtrue-lkmtrue-kmi100/May-12-2026_04-36-38/best_collision_model.pth
INFO:root:epoch 499 evaluating [time: 3.64s, collision_rate: 0.0672, avg_utilization: 0.7588]
INFO:root:Saving current: ./index_train_runs/Arts/index/qwen3-embedding-4B/rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003_kmtrue-lkmtrue-kmi100/May-12-2026_04-36-38/epoch_499_collision_0.0672_util_0.7588_model.pth
Best Loss 19.69821873307228
Best Collision Rate 0.06722599830076466
Index training started. Log file: ./log/index_train_20260512043613.log
W&B Run Name: base-bs256-lr0.001-ep500-np4
```

```
{
  "ckpt_path": "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/index_train_runs/Arts/index/qwen3-embedding-4B/rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003_kmtrue-lkmtrue-kmi100/May-12-2026_04-36-38/best_collision_model.pth",
  "collision_rate": 0.004672897196261682,
  "created_at": "2026-05-12T04:50:21",
  "cross_dataset_collision_groups_round0": null,
  "datasets": [
    "Arts"
  ],
  "max_conflicts": 4,
  "max_reencode_rounds": 20,
  "multi_output": false,
  "output_suffix": ".index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsArts_ridMay-12-2026-04-36-38.json",
  "reencode_rounds": 20,
  "total_items": 9416,
  "unique_indices": 9372
}
```
