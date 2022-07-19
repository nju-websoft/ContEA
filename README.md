# ContEA: Continual Entity Alignment

This repository is the original implementation of ContEA, the model proposed in paper [***Facing Changes: Continual Entity Alignment for Growing Knowledge Graphs***]() at ISWC2022.

## Datasets

We construct three new cross-lingual ([ZH-EN](https://github.com/nju-websoft/ContEA/tree/main/datasets/ZH-EN), [JA-EN](https://github.com/nju-websoft/ContEA/tree/main/datasets/JA-EN), and [FR-EN](https://github.com/nju-websoft/ContEA/tree/main/datasets/FR-EN)) datasets for continual entity alignment task setting, in which the to-be-aligned KGs are growing independently over time. 

The datasets can be downloaded in folder `datasets/`. Each dataset contains 6 consecutive snapshots of a growing KG. 

```
---- base       # snapshot 0
  |- batch2     # snapshot 1
  |- batch3     # snapshot 2
  |- batch4     # snapshot 3
  |- batch5     # snapshot 4
  |- batch6     # snapshot 5
  |- ent_dict   # index for entities
  |- rel_dict   # index for relations
```

The test and validation data are changeless in our continual entity alignment setting, and can be found in the first snapshot (the `base/` folder). In following snapshots (the `batchX` folder), new triples are addede into KGs which brings more potential alignment (i.e. new test data).

## Environment

The essential packages and recommened version to run the code:

- python3 (>=3.7)
- pytorch (1.11.0+cu113)
- numpy   (1.21.5)
- torch-scatter (2.0.9)
- scipy  (1.7.3)
- tabulate  (0.8.9)

## Run ContEA

We provide a demo file `src/run.sh` to run ContEA on ZH-EN dataset. The hyperparameters are the ones to reproduce the results in paper. To run the demo file, enter `src/` and run

```
$ bash run.sh
```

