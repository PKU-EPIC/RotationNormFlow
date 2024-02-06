# <div align="center">[CVPR 2023] Delving into Discrete Normalizing Flows on SO(3) Manifold for Probabilistic Rotation Modeling</div>
<!-- Official code for "**Delving into Discrete Normalizing Flows on SO(3) Manifold for Probabilistic Rotation Modeling**" *(CVPR 2023)* -->

<p align="center">
  <video controls>
  <source src="assets/symsol.mp4" type="video/mp4">
</video>
</p>

<div align="center"> 

[**Setup**](#🔧-setup) **|** [**Train**](#train) **|** [**Eval**](#evaluation) **|** [**Bib**](#bibtex) **|**  [**acknowledgement**](#🤗-acknowledgement)

[**Paper**](https://arxiv.org/abs/2304.03937) **|** [**Page**](https://pku-epic.github.io/RotationNormFlow/) **|** [**Video**](https://youtu.be/0t_GyLGsmV8) 


</div>

# 🔧 Setup

## Dependencies

* Our code are based on python 3.9, with GPU version pytorch and pytorch3d install. 
* Our code has been tested on torch 1.13.0, pytorch3d 0.7.5, nvcc 11.8.
* You can refer to [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for installation of pytorch3d.
* install other dependencies:
```
pip install scipy scikit-learn nflows healpy opencv-python wget scikit-image
conda install -y tqdm configargparse matplotlib
conda install -y -c conda-forge python-lmdb
```
## Dataset
**SYNTHETIC**

Obtain synthetic dataset used in unconditional experiments in [raw](https://drive.google.com/file/d/1db5dgRsgXnpRgDSO4xxHbO3J229zbfF0/view?usp=sharing) and link it to `data`.
```bash
unzip raw.zip
ln -s $PWD/rwa $PROJECT_PATH/data
```

**SYMSOL**

Obtain SYMSOL dataset following the same setting as in [IPDF](https://github.com/google-research/google-research/tree/master/implicit_pdf) and link it to `data`

**ModelNet10-SO(3)**

Obtain ModelNet10-SO(3) dataset from [website](https://github.com/leoshine/Spherical_Regression#modelnet10-so3-dataset) and link it to `data`

```bash
unzip ModelNet10-SO3.zip
ln -s $PWD/ModelNet10-SO3 $PROJECT_PATH/data
```

**Pascal3D+**

Obtain Pascal3D+ (release1.1) dataset from [website](https://cvgl.stanford.edu/projects/pascal3d.html) and link it to `data`

```bash
unzip PASCAL3D+_release1.1.zip
ln -s $PWD/PASCAL3D+_release1.1 $PROJECT_PATH/data
```

Obtain the synthetic data from [website](https://shapenet.cs.stanford.edu/media/syn_images_cropped_bkg_overlaid.tar) and link it to `data`
```bash
tar -xvf syn_images_cropped_bkg_overlaid.tar
ln -s $PWD/syn_images_cropped_bkg_overlaid $PROJECT_PATH/data
```

Please note that when using Pascal3D+, the data annotations will be generated during the first run of the program.

After downloading all datasets, you should have following folder tree:
```
./
├── configs/
├── data/
│   ├── ModelNet10-SO3/
│   ├── PASCAL3D+_release1.1
│   │   └── /syn_images_cropped_bkg_overlaid
│   ├── raw
│   └── symsol_dataset
├── settings/
├── utils/
...
```
# 👋 Usage

## Train

```bash
python train_uncondition.py --config=<config> [--args] # for unconditional experiments
python train.py --config=<config> [--args] # for conditional experiments
```
The training process is logged by `tensorboard`. 
### UNCONDITION:
```bash
python train_uncondition.py --config=settings/raw.yml --category=<category> 
# category can be [peak, cone, line, fisher24]
```
### SYMSOL:
```bash
python train.py --config=settings/symsol.yml # symsol I
python train.py --config=settings/symsol2.yml --category=<category> # symsol II, category can be [sphereX, cylO, tetX] 
```

### ModelNet10-SO3:
```bash
python train.py --config=settings/modelnet.yml
python train.py --config=settings/modelnet_fisher.yml
```

We use pretrained fisher as backbone to extract feature from image and also provide a good estimation for base distribution. These pretrained model can be obtained in [website](https://drive.google.com/file/d/19fKSEpfIP_0ZPtnigpXAm_RXHn2GFmdK/view?usp=share_link), and unzip it and link it to `Matrixfisher`. You should have the following folder tree:
```
./
├── configs/
├── data/
├── Matrixfisher/
│   ├── storage/
│   │   ├── modelnet/
│   │   ├── pascal/
│   ....
├── settings/
├── utils/
...
```
### Pascal3D+
```bash
python train.py --config=settings/pascal_uni.yml
python train.py --config=settings/pascal_fisher.yml
```

## Evaluation

```bash
python eval_uncondition.py <ckpt_path> --configs <config name> --eval_only [--args] # for conditional experiments
python eval.py <ckpt_path> --configs <config name> --eval_only [--args] # for unconditional experiments
```

⚡ Checkpoint can be obtained in [website](https://drive.google.com/drive/folders/1Fd3SG7x8EmG0ArQgkddJxB8fxMGi_8Yi?usp=sharing). We provide checkpoint of synthetic datasets (for unconditional experiments, in `raw` folder), and of modelnet10-SO3, SYMSOL and Pascal3d+ datasets.

### UNCONDITION:
```bash
python eval_uncondition.py <ckpt_path> --config=settings/raw.yml --category=<category> # category can be [peak, cone, line, fisher24]
```
### SYMSOL:
```bash
python eval.py <ckpt_path> --config=settings/symsol.yml # symsol I (MobiusAffine)
python eval.py <ckpt_path> --config=settings/symsol.yml --rot 16UnRot # symsol I (Ablation: MobiusRot)
python eval.py <ckpt_path> --config=settings/symsol.yml --layers 42 --last_affine 0 --rot None # symsol I (Ablation: Mobius)
python eval.py <ckpt_path> --config=settings/symsol.yml --layers 42 --dist noflow # symsol I (Ablation: Affine)
python eval.py <ckpt_path> --config=settings/symsol.yml --lu 1 # symsol I (Ablation: lu)
python eval.py <ckpt_path> --config=settings/symsol2.yml --category=<category> # symsol II, category can be [sphereX, cylO, tetX] 
```

### ModelNet10-SO3:
```bash
python eval.py <ckpt_path> --config=settings/modelnet.yml
python eval.py <ckpt_path> --config=settings/modelnet_fisher.yml
```

### Pascal3D+
```bash
python eval.py <ckpt_path> --config=settings/pascal.yml
python eval.py <ckpt_path> --config=settings/pascal_fisher.yml
```

# Bibtex
```bibtex
@inproceedings{liu2023delving,
  title={Delving into Discrete Normalizing Flows on SO (3) Manifold for Probabilistic Rotation Modeling},
  author={Liu, Yulin and Liu, Haoran and Yin, Yingda and Wang, Yang and Chen, Baoquan and Wang, He},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21264--21273},
  year={2023}
}

```

# 🤗 Acknowledgement
The code base used in this project is sourced from the repository of the [matrix Fisher distribution](https://github.com/Davmo049/Public_prob_orientation_estimation_with_matrix_fisher_distributions) and , [Implicit-PDF](https://github.com/google-research/google-research/tree/master/implicit_pdf) (for visualization), [Flow on tori and sphere](https://github.com/ryushinn/flows-on-sphere).

# ✉️ Misc

If you have any question, please feel free to mail `yul266@ucsd.edu`.