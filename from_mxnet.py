"""
Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_

This article is an introductory tutorial to deploy mxnet models with NNVM.

For us to begin with, mxnet module is required to be installed.

A quick solution is
```
pip install mxnet --user
```
or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import os, sys
thisdir=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(thisdir, 'tvm_local/python'))
sys.path.insert(0, os.path.join(thisdir, 'tvm_local/nnvm/python'))
sys.path.insert(0, os.path.join(thisdir, 'tvm_local/topi/python'))

import mxnet as mx
import nnvm
import tvm
import numpy as np
import time

print(mx.__file__)
print(nnvm.__file__)
print(tvm.__file__)

target = 'opencl'
dtype = 'float32'
ctx = tvm.context(target, 0)

nnvm.compiler.build_module.BuildConfig.current = nnvm.compiler.build_module.build_config(opt_level=2)

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
# from mxnet.gluon.model_zoo.vision import get_model
from symbols.mobilenetv2 import get_symbol
from PIL import Image
from matplotlib import pyplot as plt

model_name = 'models/mobilenetv2-1_0'
img_name = 'data/cat.jpg'
synset_name = 'data/imagenet1k-synset.txt'
with open(synset_name) as f:
    synset = f.readlines()
image = Image.open(img_name).resize((224, 224))

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)
print('x', x.shape)

######################################################################
# Compile the Graph
# -----------------
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.
mx_sym, args, auxs = mx.model.load_checkpoint(model_name, 0)
# now we use the same API to get NNVM compatible symbol
nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)

###################################
## convert dtype to target dtype
for k,v in nnvm_params.items():
    nnvm_params[k] = tvm.nd.empty(v.shape, dtype, ctx).copyfrom(nnvm_params[k].asnumpy())

######################################################################
# now compile the graph
import nnvm.compiler
shape_dict = {'data': x.shape}
graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params, dtype=dtype)

######################################################################
# Deploy the model
# ---------------------------------
# Now, we dump all the compiled outputs into files, so that we can load
# them in c++ source code.

## change the `target` to `llvm`, and uncomment following lines to
## export the model for prediction over TVM.
# thisdir = os.path.dirname(os.path.abspath(__file__))
# path_lib = os.path.join(thisdir, "deploy.so")
# lib.export_library(path_lib)
# with open(os.path.join(thisdir, "deploy.json"), "w") as fo:
#     fo.write(graph.json())
# with open(os.path.join(thisdir, "deploy.params"), "wb") as fo:
#     fo.write(nnvm.compiler.save_param_dict(params))
# with open(os.path.join(thisdir, "cat.bin"), "w") as fo:
#     fo.write(x.astype(np.float32).tobytes())

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
from tvm.contrib import graph_runtime
m = graph_runtime.create(graph, lib, ctx)
m.set_input(**params)
for i in range(3):
    # set inputs
    m.set_input('data', tvm.nd.array(x.astype(dtype)))
    tic = time.time()
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0, tvm.nd.empty((1,1000), dtype))
    toc1 = time.time()
    top1 = np.argsort(np.squeeze(tvm_output.asnumpy()))[::-1][:5]
    toc2 = time.time()
    print('elapsed: %.1f ms (%.1f ms)' % ((toc2-tic)*1000.,(toc1-tic)*1000.,))
    for i in range(5):
        print('TVM prediction top-%d:'%(i+1,), top1[i], synset[top1[i]])

