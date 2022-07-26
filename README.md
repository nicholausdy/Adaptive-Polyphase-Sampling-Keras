# Adaptive-Polyphase-Sampling-Keras
Keras implementation of adaptive polyphase sampling (APS) on Resnet-20. APS enables shift-invariant downsampling.

## Table of contents
1. [Citation](#citation)
2. [Technical Overview](#technical-overview)
3. [Getting Started](#getting-started)
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
### Concept of Adaptive Polyphase Sampling (APS)
The main goal of APS is to achieve shift-invariance on downsampling operations, which include both strided convolutions and pooling operations.
In order to achieve that goal, the main idea behind APS is two-fold. Firstly, polyphase components are obtained from the given matrix. Polyphase components are defined as downsampled matrices generated from all possible downsampling grids. For example, given a *s=2* downsampling operation on 3D matrix with the size of *i x j x k*, where  *s* is the number of strides, *i* is the number of rows, *j* is the number of columns, and *k* is the number of channels, there are 4 possible downsampling grids, henceforth called polyphase indices:
- Downsampling starting from *i=0* and *j=0* *(0,0)*
- Downsampling starting from *i=0* and *j=1* *(0,1)*
- Downsampling starting from *i=1* and *j=0* *(1,0)*
- Downsampling starting from *i=1* and *j=1* *(1,1)*

It is easier to understand the process of obtaining polyphase components by looking at the following picture taken from the aforementioned paper. <br>
![Polyphase sampling](https://github.com/nicholausdy/Adaptive-Polyphase-Sampling-Keras/blob/main/images/Screenshot%20from%202022-07-26%2019-04-02.png)

Thus, the process of obtaining polyphase components can also be generalized by the following expression: *i,j E {0,1,...,s-1}* <br>
<br>
Secondly, as the above image suggests, the polyphase component with the highest *lp norm* is selected. By doing so, we ensure that the correct polyphase component is always extracted regardless of shifts in the matrix. Therefore, we are able to achieve adaptability to shifts during the downsampling operations.<br>

### Integration of APS to Resnet
In the case of the integration of APS to Resnet, APS can be used to replace the conventional striding operations in the convolutional layer. In this case, APS can be utilized in the residual block of Resnet as shown by the following image taken from the supplemental material of the paper. <br>
![Striding Resnet](https://github.com/nicholausdy/Adaptive-Polyphase-Sampling-Keras/blob/main/images/Screenshot%20from%202022-07-26%2019-27-33.png)

The above image shows that the APS layer is used to downsample both the convolved and original input in the residual block. In order to retain shift-invariance, the same polyphase indices are used by both the APS layer responsible for downsampling the convolved input and the APS layer responsible for downsampling the original input.

## Getting Started
### Install the necessary packages
Run the following commands to create the virtual environment and install all of the necessary packages / libraries from the requirements file within that environment. Make sure to already install Python 3 and Conda beforehand.
```python
conda create -n [env-name] python=3.8.8
conda activate [env-name]
pip3 install -r requirements.txt
```
### Generate the model
Run the following command to generate the initial untrained Resnet-20 model that is integrated with APS layer.
```python
python3 generate_model.py
```
The generated untrained model will be saved inside the *models/* directory as *"aps_resnet_untrained.h5"*.

### Test classifcation consistency before training
Run the following command to test the classification consistency of the untrained model on both the unshifted and shifted dataset. The dataset that is used is the *CIFAR-10* test set consisting of *10000* images of size *32 x 32 x3*. 
```python
python3 test_before_training.py
```

### Train the model
Run the following command to train the model. The model is trained for 250 epochs on the *CIFAR-10* train set consisting of *50000* images of size *32 x 32 x 3*. The train set itself is divided into batches of size *256* each and split into *45000* images for training and *5000* images for validation. Note that shifting is not applied to the training set at all.
```python
python3 train_model.py
```
The trained model will be saved inside the *models/* directory as *"aps_resnet_trained.h5"*.

### Evaluate the trained model
Run the following command to evaluate the trained model. By running this command, the model will be evaluated on the *CIFAR-10* test set. Metrics used for the evaluation are the aforementioned classification consistency and accuracy.
```python
python3 test_after_training.py
```
## Issues
Currently, there are still several issues with the implementation that warrant further troubleshooting and improvements. Those issues are listed below.
- During training, the model is unable to converge to an adequate level of accuracy (i.e., ~10-12% accuracy at the end of training). One probable reason for the inability to converge is the extensive use of "SAME" padding on the convolutional layer, which differs significantly from the circular padding described in the paper. Since "SAME" padding adds zeroes to the edge of the tensors to ensure the output tensor has the same dimension as the input tensor, it might adversely affect the polyphase sampling process such that the appropriate polyphase component is unable to be retrieved due to the added zeroes. 
- During evaluation after training, the model is unable to achieve the same level of consistency as that before training. One probable reason for this is the shifting process that is different from what is described in the paper. In this implementation, the shifting process replaces the shifted region of the matrix with zeroes instead of the circular shifting described in the paper. Thus, spatial information might be significantly lost in the shifted dataset, which in turn reduces the classification consistency. Another probable reason is the same as the reason for the inability to converge, which is the use of "SAME" padding.  
- Integration of anti-aliasing filters with APS that is described in the paper has not been implemented yet.
- Integration of other forms of Resnet architecture (Resnet-18, Resnet-50, Resnet-56) with APS has not been implemented yet.

