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

### Inference with Python upon NNVM

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

### Inference with C++ upon TVM

The inference python script is relatively independent from `MXNet`, it relies on `nnvm` to build a computation graph and perform the inference operations.
Since `nnvm` is built to support neural network inference on any device enabled with OpenCL, therefore, it's quite efficient to predict on an Intel/AMD/Mali GPU. Here is an concrete example:

``` bash
$ cd tvm-predict-cpp
$ ./run_example.sh
Build the libraries..
make: Nothing to be done for 'all'.
Run the example
Run the deployment with all in one packed library...
The maximum position in output vector is: 281
```

## [NEW!] Quantized Inference (INT16)

Taking advantage of the low-bit quantization feature [#2116](https://github.com/dmlc/tvm/pull/2116) in TVM, we can now perform 16-bit inference on CPU. Both timing and accuracy results are very promissing.

```
$ python eval_quantized.py
[09:26:01] src/engine/engine.cc:55: MXNet start using engine: ThreadedEngine
INFO:root:Namespace(batch_size=1, dtype_input='int16', dtype_output='int32', global_scale=256.0, log_interval=10, model='models/imagenet1k-mnetv2-1_0', nbit_input=16, nbit_output=32, num_classes=1000, original=False, rec_val='~/.mxnet/datasets/imagenet/rec/val.rec', simulated=False, target='llvm')
qconfig(nbit_input=16, nbit_weight=16, nbit_activation=32, global_scale=256.000000, skip_k_conv==0, round_for_shift==1, store_lowbit_output==0, debug_enabled_ops==(nullptr), use_stop_fusion==1)
INFO:root:Finish building model models/imagenet1k-mnetv2-1_0...
[09:26:16] src/io/iter_image_recordio_2.cc:172: ImageRecordIOParser2: /home/liangfu/.mxnet/datasets/imagenet/rec/val.rec, use 1 threads for decoding..
INFO:root:[10 samples] validation: acc-top1=0.600000 acc-top5=0.900000, speed=16.5fps
INFO:root:[20 samples] validation: acc-top1=0.700000 acc-top5=0.950000, speed=16.5fps
INFO:root:[30 samples] validation: acc-top1=0.700000 acc-top5=0.966667, speed=16.3fps
INFO:root:[40 samples] validation: acc-top1=0.700000 acc-top5=0.950000, speed=16.4fps
INFO:root:[50 samples] validation: acc-top1=0.700000 acc-top5=0.940000, speed=16.4fps
INFO:root:[60 samples] validation: acc-top1=0.700000 acc-top5=0.916667, speed=16.4fps
INFO:root:[70 samples] validation: acc-top1=0.728571 acc-top5=0.914286, speed=16.4fps
INFO:root:[80 samples] validation: acc-top1=0.725000 acc-top5=0.912500, speed=16.4fps
INFO:root:[90 samples] validation: acc-top1=0.711111 acc-top5=0.900000, speed=16.2fps
INFO:root:[100 samples] validation: acc-top1=0.690000 acc-top5=0.910000, speed=15.5fps
INFO:root:[final] validation: acc-top1=0.690000 acc-top5=0.910000
```

### Known Issues

Current implementation of `dmlc/nnvm` requires a merge with the PR submission [here](https://github.com/dmlc/nnvm/pull/435). For a quick solution, you can simply add `'clip'` to the `_identity_list` variable in `frontend/mxnet.py` .

### Miscellaneous

For Gluon version of MobileNetV2, please refer to [chinakook/MobileNetV2.mxnet](https://github.com/chinakook/MobileNetV2.mxnet).

### License

MIT License
