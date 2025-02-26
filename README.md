# DIFD for few-shot medical image segmentation


### Abstract
Acquiring a large volume of annotated medical data is impractical due to time, financial, and legal constraints. Consequently, few-shot medical image segmentation is increasingly emerging as a prominent research direction. Nowadays, Medical scenarios pose two major challenges: (1) intra-class variation caused by diversity among support and query sets; (2) inter-class extreme imbalance resulting from background heterogeneity. However, existing prototypical networks struggle to tackle these obstacles effectively. To this end, we propose a Dual Interspersion and Flexible Deployment (DIFD) model. Drawing inspiration from military interspersion tactics, we design the dual Interspersion module to generate representative basis prototypes from support features. These basis prototypes are then deeply interacted with query features. Furthermore, we introduce a fusion factor to fuse and refine the basis prototypes. Ultimately, we seamlessly integrate and flexibly deploy the basis prototypes to facilitate correct matching between the query features and basis prototypes, thus conducive to improving the segmentation accuracy of the model. Extensive experiments on three publicly available medical image datasets demonstrate that our model significantly outshines other SoTAs (3.23\% higher dice score on average across all datasets), achieving a new level of performance.

### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```

### Data sets and pre-processing
Download:
1) **CHAOS-MRI**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2) **Synapse-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **CMR**: [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

### Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./script/train.sh` 

### Inference
Run `./script/evaluate.sh` 

### Citation
```
@article{cheng2025dual,
  author={Cheng, Ziming and Wang, Shidong and Yang, Long and Zhou, Tao and Zhang, Haofeng and Shao, Ling},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Dual Interspersion and Flexible Deploymen for Few-Shot Medical Image Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={}
  doi={}}
```
