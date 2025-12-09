<p align="center">
  <h1 align="center">🎤 Register Any Point: Scaling 3D Point Cloud Registration by Flow Matching</h1>
  
  <p align="center">
    <a href="https://github.com/PRBonn/RAP"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/PRBonn/RAP"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://arxiv.org/pdf/2512.01850"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://github.com/PRBonn/RAP/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
  </p>
  
  <p align="center">
    <a href="https://www.ipb.uni-bonn.de/people/yue-pan/"><strong>Yue Pan</strong></a>
    ·
    <a href="https://taosun.io/"><strong>Tao Sun</strong></a>
    ·
    <a href="https://www.zhuliyuan.net/"><strong>Liyuan Zhu</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/lucas-nunes/"><strong>Lucas Nunes</strong></a>
    ·
    <a href="https://ir0.github.io/"><strong>Iro Armeni</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/jens-behley/"><strong>Jens Behley</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/cyrill-stachniss/"><strong>Cyrill Stachniss</strong></a>
  </p>
  <p align="center">
    <a href="https://www.ipb.uni-bonn.de"><strong>University of Bonn</strong>
    ·
    <a href="https://gradientspaces.stanford.edu/"><strong>Stanford University</strong></a>


  <h3 align="center"><a href="https://arxiv.org/pdf/2512.01850">Paper</a> | <a href="https://09c5c79a1e5f026ab5.gradio.live/">Demo</a> | <a href="https://register-any-point.github.io/">Homepage</a></h3>
  <div align="center"></div>
</p>

![rap_teaser](https://github.com/user-attachments/assets/932cc717-c2d4-4251-ba7e-7a36472a04a7)


## TODO List
   - [x] Release the inference code and RAP model v1.0.
   - [ ] Release the training code.
   - [ ] Release the training data generation code and example training data.
   - [ ] Release RAP model v1.1.


## Abstract

<details>
  <summary>[Details (click to expand)]</summary>
Point cloud registration aligns multiple unposed point clouds into a common frame, and is a core step for 3D reconstruction and robot localization. In this work, we cast registration as conditional generation: a learned continuous, point-wise velocity field transports noisy points to a registered scene, from which the pose of each view is recovered. Unlike previous methods that conduct correspondence matching to estimate the transformation between a pair of point clouds and then optimize the pairwise transformations to realize multi-view registration, our model directly generates the registered point cloud. With a lightweight local feature extractor and test-time rigidity enforcement, our approach achieves state-of-the-art results on pairwise and multi-view registration benchmarks, particularly with low overlap, and generalizes across scales and sensor modalities. It further supports downstream tasks including relocalization, multi-robot SLAM, and multi-session map merging.
</details>

## Installation

Clone the repo:
```
git clone https://github.com/PRBonn/RAP.git
cd RAP
```

Setup conda environment:
```
conda create -n py310-rap python=3.10 -y
conda activate py310-rap
```

Install the dependency:
```
bash ./scripts/install.sh
```

Download model and example data:
```
bash ./scripts/download_weights_and_demo_data.sh
```

## Run RAP

Try the demo by:

```
python app.py
```

Run batch inference after modifying the config files and the script `test_script_example.sh`:

```
bash ./scripts/test_script_example.sh
```

## Citation

<details>
  <summary>[Details (click to expand)]</summary>


If you use RAP for any academic work, please cite:

```
@article{pan2025arxiv,
  title = {{Register Any Point: Scaling 3D Point Cloud Registration by Flow Matching}},
  author = {Pan, Yue and Sun, Tao and Zhu, Liyuan and Nunes, Lucas and Armeni, Iro and Behley, Jens and Stachniss, Cyrill},
  journal = arxiv,
  volume  = {arXiv:2512.01850},
  year    = {2025}
}
```
</details>

## Contact
If you have any questions, please contact:

- Yue Pan {[yue.pan@igg.uni-bonn.de]()}


## Acknowledgement

<details>
  <summary>[Details (click to expand)]</summary>

RAP is built on top of [Rectified Point Flow (RPF)](https://github.com/GradientSpaces/Rectified-Point-Flow) and we thank the authors for the following works:

* [GARF](https://github.com/ai4ce/GARF)
* [BUFFER-X](https://github.com/MIT-SPARK/BUFFER-X)
* [VGGT](https://github.com/facebookresearch/vggt)
* [DiT](https://github.com/facebookresearch/DiT)
* [Muon](https://github.com/KellerJordan/Muon)

</details>