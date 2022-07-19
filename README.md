# ContEA: Continual Entity Alignment

This repository is the official implementation of ContEA, the model proposed in paper [***Facing Changes: Continual Entity Alignment for Growing Knowledge Graphs***]() at ISWC2022.

## Datasets

We construct three incremental cross-lingual ([ZH-EN](https://github.com/nju-websoft/ContEA/tree/main/datasets/ZH-EN), [JA-EN](https://github.com/nju-websoft/ContEA/tree/main/datasets/JA-EN), and [FR-EN](https://github.com/nju-websoft/ContEA/tree/main/datasets/FR-EN)) datasets for continual entity alignment task, in which the to-be-aligned KGs are growing independently over time. 

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

The test and validation data are changeless in our continual entity alignment setting, and can be found in the first snapshot (under `base/` folder). In the later snapshots (under `batchX` folder), new triples are added into KGs as well as new potential alignment.

## Environment

The essential packages and recommened version to run the code:

- python3 (>=3.7)
- pytorch (1.11.0+cu113)
- numpy   (1.21.5)
- torch-scatter (2.0.9, better to install using **pip**)
- scipy  (1.7.3)
- tabulate  (0.8.9)

## Run ContEA

We provide a demo script in `src/run.sh` to run ContEA on ZH-EN dataset. The hyperparameters can reproduce the results in paper. To run the demo script, enter `src/` and run:

```
$ bash run.sh
```

In our work, we set 𝛼 = 0.1, 𝛽 = 0.1, 𝑚 = 500, 𝜆 = 2.0. Both entity and relation dimension are 100. GNN layer number of encoder is 2. We use grid search on ContEA to find optimal hyperparameters. The ranges/values of important variables are:

| Hyperparameter      | Values |
| :---        |    :----:   |  
| batch_size (t = 0)   | {512, 1024} |
| batch_size (t > 0) | 512 |
| learning_rate (t = 0) | {0.0005, 0.001, 0.01} |
| learning_rate (t > 0) | 0.001 |
| dropout_rate | 0.3 |

## Acknowledgement

ContEA is designed upon the static entity alignment model [Dual-AMN](https://github.com/MaoXinn/Dual-AMN) (implemented in tensorflow), we thanks their code open-sourced.

## Citation

```
@inproceedings{ContEA,
  title={Facing Changes: Continual Entity Alignment for Growing Knowledge Graphs},
  author={Wang, Yuxin and Cui, Yuanning and Liu, Wenqiang and Sun, Zequn and Jiang, Yiqiao and Han, Kexin and Hu, Wei},
  booktitle={ISWC},
  year={2022}
}
```
