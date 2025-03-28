# DualCOS: Query-Efficient Data-Free Model Stealing with Dual Clone Networks and Optimal Samples

## Introduction
This is the source code for the paper [DualCOS: Query-Efficient Data-Free Model Stealing with Dual Clone Networks and Optimal Samples](https://ieeexplore.ieee.org/abstract/document/10688153) published to ICME 2024.

## Foundation of Codebase
During the implementation of the codebase, we referred to some previous work codebases. All of them are listed as follows:
1. DFME. https://github.com/cake-lab/datafree-model-extraction
2. DFAD. https://github.com/VainF/Data-Free-Adversarial-Distillation
3. MAZE. https://github.com/sanjaykariyappa/maze
4. DisGUIDE. https://github.com/lin-tan/disguide
5. Dual Students. https://github.com/James-Beetham/dual_students
6. DFMS. https://github.com/val-iisc/Hard-Label-Model-Stealing

## Usage Steps
To run the codebase successfully, you should perform the following steps correctly:

#### Data Preparation
The relevant datasets (CIFAR-10 and CIFAR-100) need to be downloaded and placed in `dualcos/data/`.

#### Teacher (Victim) Model Preparation
- You can train a teacher model from scratch or download a pre-trained model from prior paper. Then you need to add the corresponding teacher model to `dualcos/checkpoint/teacher/`. Publicly available models are available from the original DFAD codebase: https://github.com/VainF/Data-Free-Adversarial-Distillation
- Name the teacher model appropriately. For example, `cifar10-resnet34_8x.pt` indicates that the teacher model architecture is ResNet-34, and the training dataset is CIFAR-10.

#### Perform Model Stealing Attack
```
bash run_cifar10.sh
bash run_cifar100.sh
```
