# DeepCA: Deep Learning-based 3D Coronary Artery Tree Reconstruction from Two 2D Non-simultaneous X-ray Angiography Projections

# 1. Overview

This is the official code repository for the [DeepCA](https://arxiv.org/abs/2407.14616) paper by Yiying Wang, Abhirup Banerjee, Robin P. Choudhury and Vicente Grau, which has been early accepted to the WACV 2025 **(early acceptance rate: 12.1%)**.

## Citation

If you find the code are useful, please consider citing the paper.

```
@article{wang2024deep,
  title={Deep Learning-based 3D Coronary Tree Reconstruction from Two 2D Non-simultaneous X-ray Angiography Projections},
  author={Wang, Yiying and Banerjee, Abhirup and Choudhury, Robin P and Grau, Vicente},
  journal={arXiv preprint arXiv:2407.14616},
  year={2024}
}
```

# 2. Introduction

Cardiovascular diseases (CVDs) are the most common cause of death worldwide. Invasive x-ray coronary angiography (ICA) is one of the most important imaging modalities for the diagnosis of CVDs. ICA typically acquires only two 2D projections, which makes the 3D geometry of coronary vessels difficult to interpret, thus requiring 3D coronary tree reconstruction from two projections. State-of-the-art approaches require significant manual interactions and cannot correct the non-rigid cardiac and respiratory motions between non-simultaneous projections. In this study, we propose a novel deep learning pipeline called DeepCA. We leverage the Wasserstein conditional generative adversarial network with gradient penalty, latent convolutional transformer layers, and a dynamic snake convolutional critic to implicitly compensate for the non-rigid motion and provide 3D coronary artery tree reconstruction. Through simulating projections from coronary computed tomography angiography (CCTA), we achieve the generalisation of 3D coronary tree reconstruction on real non-simultaneous ICA projections. We incorporate an application-specific evaluation metric to validate our proposed model on both a CCTA dataset and a real ICA dataset, together with Chamfer L1 distance. The results demonstrate promising performance of our DeepCA model in vessel topology preservation, recovery of missing features, and generalisation ability to real ICA data. To the best of our knowledge, this is the first study that leverages deep learning to achieve 3D coronary tree reconstruction from two real non-simultaneous x-ray angiography projections. 

## Our Proposed Pipeline

<p align="center">
  <img src="https://github.com/WangStephen/DeepCA/blob/main/img/workflow.pdf" width="1200" height="600">
</p>

# 3. Packages Requirement

This work requires following dependency packages:

```
python: 3.9.18
pytorch: 2.1.1
numpy: 1.23.5 
nibabel: 3.2.2
tigre 
```

# 4. Code Instructions

## Training Data Preparation

Our training data are based on the segmented CCTA data (label) from [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT).

We first split the segmented CCTA data into RCA and LAD and then use the RCA data to simulate cone-beam forward projections. Please run the functions `CCTA_split` and `generate_deformed_projections_RCA` in `projections_simulation.py` to generate your simulated deformed projections from the segmented CCTA data.

## Model Training

Run the model to start training:

```
python ./DeepCA/train_models/train_model.py 
```

# 4. License

Please see [license](https://github.com/WangStephen/DeepCA/blob/main/LICENSE).

# 5. Acknowledgement

DeepCA model architectures are revised based on [Compact Transformers](https://github.com/SHI-Labs/Compact-Transformers) and [Dynamic Snake Convolution](https://github.com/YaoleiQi/DSCNet).

Cone-beam forward projection simulation is based on [TIGRE Toolbox](https://github.com/CERN/TIGRE).
