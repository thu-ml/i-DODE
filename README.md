# i-DODE
This repo is the official code for the paper [Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs](https://proceedings.mlr.press/v202/zheng23c.html) (ICML 2023).

<h3><a href="https://arxiv.org/pdf/2305.03935.pdf">Paper</a> | <a href="https://arxiv.org/abs/2305.03935">arXiv</a></h3>

The code implementation is based on [google-research/vdm](https://github.com/google-research/vdm) by Diederik P. Kingma.

--------------------

![](https://icml.cc/media/PosterPDFs/ICML%202023/23818.png)

We achieve state-of-the-art likelihood estimation results by diffusion ODEs on image datasets (2.56 on CIFAR-10, 3.43/3.69 on ImageNet-32) *without variational dequantization or data augmentation*.

Our improved techniques include:

(maximum likelihood training)

- Velocity parameterization with likelihood weighting
- Error-bounded second-order flow matching (finetuning)
- Variance reduction with importance sampling

(likelihood evaluation)

- Training-free truncated-normal dequantization

## How to run the code

### Dependencies

Following [VDM](https://github.com/google-research/vdm), the code is based on Jax/Flax instead of PyTorch. 

To run on a GPU machine, you need to find a corresponding version for your Python version and CUDA version at: https://storage.googleapis.com/jax-releases/jax_cuda_releases.html. We recommend using `python==3.8` and `cuda==11.2`. To install the required libraries:

```shell
wget https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.15+cuda11.cudnn805-cp38-none-manylinux2014_x86_64.whl
pip install jaxlib-0.3.15+cuda11.cudnn805-cp38-none-manylinux2014_x86_64.whl
pip install -r requirements.txt
```

### Datasets

The experiments are conducted on CIFAR10 and ImageNet32. We use the dataloader provided by `tensorflow_datasets`.

As stated in the paper, there are two different versions of ImageNet32 dataset. For fair comparisons, we use both versions of ImageNet32, one is downloaded from https://image-net.org/data/downsample/Imagenet32_train.zip, following Flow Matching [3], and the other is downloaded from http://image-net.org/small/train_32x32.tar (old version, no longer available), following [ScoreSDE](https://github.com/yang-song/score_sde) and [VDM](https://github.com/google-research/vdm). The former dataset applies anti-aliasing and is easier for maximum likelihood training.

For convenience, we adapt both versions of ImageNet32 to `tensorflow_datasets`.

### Configs

There are three config files, corresponding to CIFAR10, ImageNet32 (old version) and ImageNet32 (new version), respectively.

```python
configs
├── cifar10.py
├── imagenet32.py
└── imagenet32_new.py
```

Some key configurations:

```python
config.model = d(
        schedule="VP", # noise schedule, "VP" (variance preserving)/"SP" (straight line)
        second_order=False, # enable second order loss for finetuning
        velocity=True, # enable velocity parameterization
        importance=True, # enable importance sampling
        dequantization="tn", # dequantization type, "u" (uniform)/"v" (variational)/"tn" (truncated normal)
        num_importance=1, # number of importance samples for likelihood evaluation
    )
```

### Usage

#### Training

Train from scratch:

```shell
python main.py --config=configs/cifar10.py --workdir=experiments/cifar10_VP --mode=train
python main.py --config=configs/cifar10.py --workdir=experiments/cifar10_SP --mode=train --config.model.schedule="SP"
```

Continue training from a previous checkpoint:

```shell
python main.py --config=configs/cifar10.py --workdir=experiments/cifar10_VP --mode=train --config.ckpt_restore_dir=experiments/cifar10_VP/cifar10/20230101-012345/checkpoints-0
```

Finetune by the second order loss:

```shell
python main.py --config=configs/cifar10.py --workdir=experiments/cifar10_VP_finetuned --mode=train --config.ckpt_restore_dir=experiments/cifar10_VP/cifar10/20230101-012345/checkpoints-0 --config.model.second_order=True --config.training.num_steps_train=6200000
```

#### Likelihood Evaluation

Compute the SDE/ODE likelihood (measured by BPD) of a checkpoint:

```shell
python main.py --config=configs/cifar10.py --workdir=experiments/cifar10_VP --mode=eval --checkpoint=experiments/cifar10_VP/cifar10/20230101-012345/checkpoints-0
```

#### Sampling

```shell
python main.py --config=configs/cifar10.py --workdir=experiments/cifar10_VP --mode=sample --checkpoint=experiments/cifar10_VP/cifar10/20230101-012345/checkpoints-0
```

The sampled images will be saved to `workdir/samples` in the form of `.npz`.

## Pretrained checkpoints

## References

If you find the code useful for your research, please consider citing:
```bib
@inproceedings{zheng2023improved,
  title={Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs},
  author={Zheng, Kaiwen and Lu, Cheng and Chen, Jianfei and Zhu, Jun},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```

This work is built upon some previous papers which might also interest you:

[1] Diederik P Kingma, Tim Salimans, Ben Poole, Jonathan Ho. "Variational diffusion models". *Advances in Neural Information Processing Systems*, 2021.

[2] Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu. "Maximum likelihood training for score-based diffusion odes by high order denoising score matching". *International Conference on Machine Learning*, 2022.

[3] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. "Flow matching for generative modeling". *International Conference on Learning Representations*. 2022.
