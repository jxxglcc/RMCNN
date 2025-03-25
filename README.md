# A Riemannian Convolutional Nueral Network for EEG-Based Motor Imagery Decoding

This repository contains the code for RMCNN implemented with PyTorch.

More details in paper: A Riemannian Convolutional Nueral Network for EEG-Based Motor Imagery Decoding

## Implementations of FBCSP-SVM, FBCNet, FBMSNet, Conformer,  TSFCNet, Tensor-CSPNet, Graph-CSPNet and MAtt

All these benchmark methods are implemented in Pytorch.

FBCSP-SVM is provided at [https://github.com/fbcsptoolbox/fbcsp_code](https://github.com/fbcsptoolbox/fbcsp_code)

FBCNet is provided at [https://github.com/ravikiran-mane/FBCNet](https://github.com/ravikiran-mane/FBCNet)

FBMSNet is provided at [https://github.com/ravikiran-mane/FBCNet](https://github.com/ravikiran-mane/FBCNet)

Conformer is provided at [https://github.com/eeyhsong/EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer)

TSFCNet is provided at [https://github.com/hongyizhi/TSFCNet](https://github.com/hongyizhi/TSFCNet)

Tensor-CSPNet and Graph-CSPNet is provided at [https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet](https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet)

MAtt is provided at [https://github.com/CECNL/MAtt](https://github.com/CECNL/MAtt)

### File Descriptions

* [model](https://github.com/jxxglcc/RMCNN/tree/main/model) - This file contains the model used in this repository.
* [utils](https://github.com/jxxglcc/RMCNN/tree/main/utils) - This file contains the functions used in this repository.
* [FBCSP-SVM](https://github.com/jxxglcc/RMCNN/tree/main/FBCSP-SVM) - This file contains the example code for classifying MI-EEG data using FBCSP-SVM.
* [Tensor-CSPNet and Graph-CSPNet](https://github.com/jxxglcc/RMCNN/tree/main/Tensor-CSPNet%20and%20Graph-CSPNet) - This file contains the example code for classifying MI-EEG data using Tensor-CSPNet or Graph-CSPNet.
*  [main_FBCNet.py](https://github.com/jxxglcc/RMCNN/blob/main/main_FBCNet.py) - An example code for classifying MI-EEG data using FBCNet.
* [main_FBMSNet.py](https://github.com/jxxglcc/RMCNN/blob/main/main_FBMSNnet.py) - An example code for classifying MI-EEG data using FBMSNet.
* [main_Conformer.py](https://github.com/jxxglcc/RMCNN/blob/main/main_Conformer.py) - An example code for classifying MI-EEG data using Conformer.
* [main_TSFCNet .py](https://github.com/jxxglcc/RMCNN/blob/main/main_TSFCNet.py) - An example code for classifying MI-EEG data using TSFCNet.
* [main_MAtt.py](https://github.com/jxxglcc/RMCNN/blob/main/main_MAtt.py) - An example code for classifying MI-EEG data using MAtt.
* [hold_out_benchmark.py](https://github.com/jxxglcc/RMCNN/blob/main/hold_out_benchmark.py) - holdout  code for FBCNet, FBMSNet, Conformer, TSFCNet and MAtt.

### Data Availability

The BCIC-IV-2a dataset can be downloaded in the following link: [https://www.bbci.de/competition/iv/](https://www.bbci.de/competition/iv/); The OpenBMI dataset) can be downloaded in the following link: [http://gigadb.org/dataset/100542](http://gigadb.org/dataset/100542); The High GAMMA dataset can be downloaded in the following link: https://gin.g-node.org/robintibor/high-gamma-dataset.

