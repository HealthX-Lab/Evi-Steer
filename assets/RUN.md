# Training and Evaluation

Below we provide training and evaluation instructions for Evi-Steer. The same instructions applies for all other techniques.


### Training Compute
We train Evi-Steer on each dataset with a batch size of 16 using a **single** NVIDIA A100 GPU.

## Evi-Steer

#### (1) Few-shot evaluation setting

The default training settings are provided in the config files at `configs/EviSteer.yaml`. All hyper-parameters can be modified using this config file.

Below, we provide instructions to reproduce the few-shot results for Evi-Steer. 

```bash
bash scripts/fewshot.sh
```

You can evaluate the few-shot results by downloading the models from [here](), placing the models in `checkpoints`, then running the following:

```bash
bash scripts/eval_fewshot.sh
```

#### (2) Domain generalization setting

The default training settings are provided in the config files at `configs/EviSteer.yaml`. All hyper-parameters can be modified using this config file.

```bash
bash scripts/dg.sh
```

You can evaluate the domain generalization results by downloading the models from [here](), placing the models in `checkpoints`, then running the following:

```bash
bash scripts/eval_dg.sh
```