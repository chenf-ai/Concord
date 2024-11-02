
# Concord
This is the official code for the TMLR paper: [`One by One, Continual Coordinating with Humans via Hyper-Teammate Identification`](https://openreview.net/forum?id=HVxumpoWBm).

## Algorithm projects
### Installation instructions
Please first install Pytorch and packages in requirements by using:
```
1.conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
2.pip install -r requirements.txt
```

### Running instructions
We incorporate two envs, `overcooked` and `SMAC`, and build a separate project for each in this code.
you can go into one of projects (`overcooked` or `SMAC`), and run our `Concord` by using:
```
sh bash/train_CL_Concord.sh
```

Except `Concord`, we also incorporte an implementation of other baselines like `Online EWC` and `CLEAR`.
For example, run `Online EWC` by using:
```
sh bash/train_CL_EWC.sh
```

After training the hypernet of Concord, if you want to train the recognizer, you may follow the following steps.
+ 1.Put the paths of all Concord models (12 for overcooked, 3 or 6 for SMAC) to `agent_load_path = []` in run_recognizer.py
+ 2.Train the recognizer by using:
```
sh bash/train_recognizer.sh
```

## Overcooked environment
Our experiments are performed in overcooked including four layouts: *Coordination Ring*, *Asymmetric Advantages*,*Many Orders* and *Open Asymmetric Advantages*. These layouts are named "random1", "unident\_s8", "many\_orders"and "unident\_open" respectively in the code.
### Installation instructions
Just enter the folder of `Env/HSP_main`, and follow README.md to install overcooked environment.

## SMAC environment
Our experiments are also performed on three SMAC maps, namely 3m (3 marines), 2z1s (2 zealots and 1 stalkers) and 4m (4 marines). There exist the same number and 140 type of units on both ally and enemy sides.

### Installation instructions
Please enter the folder of `Env/SMAC/`. Then, there two steps to install smac environment.
+ 1.Run the following command to download SC2 backend project into the folder of Env/SMAC/3rdparty 
```shell
bash install_sc2.sh
```
+ 2.Run the following command to install smac pip package:
```
pip install -e smac/
```

## Note
The implementation is based on [DOP](https://github.com/TonghanWang/DOP), [SMAC](https://github.com/oxwhirl/smac) and [HSP](https://github.com/samjia2000/HSP) which are open-sourced.