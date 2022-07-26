# Adaptive-Polyphase-Sampling-Keras
Keras implementation of adaptive polyphase sampling (APS) on Resnet-20. APS enables shift-invariant downsampling.

## Table of contents
1. [Citation](#citation)
2. [Technical Overview](#technical-overview)
3. [Simple Manual](#simple-manual)
4. [Issues](#issues)

## Citation
This implementation of APS is based on the following paper from the Computer Vision and Pattern Recognition (CVPR) 2021 conference: <br>
Chaman, A., & Dokmanic, I. (2021). [**Truly Shift-Invariant Convolutional Neural Networks**](https://openaccess.thecvf.com/content/CVPR2021/html/Chaman_Truly_Shift-Invariant_Convolutional_Neural_Networks_CVPR_2021_paper.html). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 3773–3783.
```BibText
@InProceedings{Chaman_2021_CVPR,
    author    = {Chaman, Anadi and Dokmanic, Ivan},
    title     = {Truly Shift-Invariant Convolutional Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3773-3783}
}
```
Please write the above citation if you use codes from this repository.

## Technical Overview
The main goal of APS is to achieve shift-invariance on downsampling operations, which include both strided convolutions and pooling operations.
In order to achieve that goal, the main idea behind APS is two-fold. 

## Simple Manual

## Issues
