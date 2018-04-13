# Reproduction of MobileNetV2 using MXNet

This is a MXNet implementation of MobileNetV2 architecture as described in the paper [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/pdf/1801.04381).

### Pretrained Models on ImageNet

We provide pretrained MobileNet models on ImageNet, which achieve slightly better accuracy rates than the original ones reported in the paper. We think the improved accuracy relies on additional augmentation strategy that use 480xN as input, and random scale between 0.533 ~ 1.0 at early training stages.

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN) on validation set:

Network|Top-1|Top-5|
:---:|:---:|:---:|
MobileNet V2| 72.45| 90.78|

### Normalization

The input images are substrated by mean RGB = [ 123.68, 116.78, 103.94 ].

### Inference

The inference python script is relatively independent from `MXNet`, it relies on `nnvm` to build a computation graph and perform the inference operations. 
Since `nnvm` is built to support neural network inference on any device enabled with OpenCL, therefore, it's quite efficient to predict on an Intel/AMD/Mali GPU. Here is an concrete example:

``` python
>> python from_mxnet.py
[14:52:11] src/runtime/opencl/opencl_device_api.cc:205: Initialize OpenCL platform 'Intel Gen OCL Driver'
[14:52:12] src/runtime/opencl/opencl_device_api.cc:230: opencl(0)='Intel(R) HD Graphics Skylake ULT GT2' cl_device_id=0x7f091bbd2bc0
elapsed: 3005.0ms (3004.5ms)
('TVM prediction top-1:', 162, 'n02088364 beagle\n')
('TVM prediction top-2:', 167, 'n02089973 English foxhound\n')
('TVM prediction top-3:', 166, 'n02089867 Walker hound, Walker foxhound\n')
('TVM prediction top-4:', 188, 'n02095314 wire-haired fox terrier\n')
('TVM prediction top-5:', 215, 'n02101388 Brittany spaniel\n')
elapsed: 80.0ms (79.3ms)
('TVM prediction top-1:', 162, 'n02088364 beagle\n')
('TVM prediction top-2:', 167, 'n02089973 English foxhound\n')
('TVM prediction top-3:', 166, 'n02089867 Walker hound, Walker foxhound\n')
('TVM prediction top-4:', 188, 'n02095314 wire-haired fox terrier\n')
('TVM prediction top-5:', 215, 'n02101388 Brittany spaniel\n')
elapsed: 80.0ms (79.3ms)
('TVM prediction top-1:', 162, 'n02088364 beagle\n')
('TVM prediction top-2:', 167, 'n02089973 English foxhound\n')
('TVM prediction top-3:', 166, 'n02089867 Walker hound, Walker foxhound\n')
('TVM prediction top-4:', 188, 'n02095314 wire-haired fox terrier\n')
('TVM prediction top-5:', 215, 'n02101388 Brittany spaniel\n')
```

### License

MIT License


