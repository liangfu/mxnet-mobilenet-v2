# Reproduction of MobileNetV2 using MXNet

This is a MXNet implementation of MobileNetV2 architecture as described in the paper [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/pdf/1801.04381).

### Pretrained Models on ImageNet

We provide pretrained MobileNet models on ImageNet, which achieve slightly lower accuracy rates than the original ones reported in the paper. We applied the augmentation strategy that use 480xN as input, and random scale between 0.533 ~ 1.0 at early training stages.

Here is the top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN) on validation set:

Network|Multiplier|Top-1|Top-5|
:---:|:---:|:---:|:---:|
MobileNet V2|1.0|71.75|90.15|
MobileNet V2|1.4|73.09|91.09|

More pretrained models with different `multiplier` settings would be uploaded later.

### Normalization

The input images are substrated by mean RGB = [ 123.68, 116.78, 103.94 ].

### Inference

The inference python script is relatively independent from `MXNet`, it relies on `nnvm` to build a computation graph and perform the inference operations. 
Since `nnvm` is built to support neural network inference on any device enabled with OpenCL, therefore, it's quite efficient to predict on an Intel/AMD/Mali GPU. Here is an concrete example:

``` python
>> python from_mxnet.py
[14:52:11] src/runtime/opencl/opencl_device_api.cc:205: Initialize OpenCL platform 'Intel Gen OCL Driver'
[14:52:12] src/runtime/opencl/opencl_device_api.cc:230: opencl(0)='Intel(R) HD Graphics Skylake ULT GT2' cl_device_id=0x7f091bbd2bc0
elapsed: 2992.1 ms (2991.7 ms)
('TVM prediction top-1:', 281, 'n02123045 tabby, tabby cat\n')
('TVM prediction top-2:', 285, 'n02124075 Egyptian cat\n')
('TVM prediction top-3:', 282, 'n02123159 tiger cat\n')
('TVM prediction top-4:', 278, 'n02119789 kit fox, Vulpes macrotis\n')
('TVM prediction top-5:', 287, 'n02127052 lynx, catamount\n')
elapsed: 63.3 ms (62.8 ms)
('TVM prediction top-1:', 281, 'n02123045 tabby, tabby cat\n')
('TVM prediction top-2:', 285, 'n02124075 Egyptian cat\n')
('TVM prediction top-3:', 282, 'n02123159 tiger cat\n')
('TVM prediction top-4:', 278, 'n02119789 kit fox, Vulpes macrotis\n')
('TVM prediction top-5:', 287, 'n02127052 lynx, catamount\n')
elapsed: 62.6 ms (62.1 ms)
('TVM prediction top-1:', 281, 'n02123045 tabby, tabby cat\n')
('TVM prediction top-2:', 285, 'n02124075 Egyptian cat\n')
('TVM prediction top-3:', 282, 'n02123159 tiger cat\n')
('TVM prediction top-4:', 278, 'n02119789 kit fox, Vulpes macrotis\n')
('TVM prediction top-5:', 287, 'n02127052 lynx, catamount\n')
```

### Known Issues

Current implementation of `dmlc/nnvm` requires a merge with the PR submission [here](https://github.com/dmlc/nnvm/pull/435). For a quick solution, you can simply add `'clip'` to the `_identity_list` variable in `frontend/mxnet.py` .

### Miscellaneous

For Gluon version of MobileNetV2, please refer to [chinakook/MobileNetV2.mxnet](https://github.com/chinakook/MobileNetV2.mxnet).

### License

MIT License
