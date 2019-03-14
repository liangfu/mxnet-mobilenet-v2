import os, sys
thisdir = os.path.abspath(os.path.dirname(__file__))
tvmroot = os.path.join(thisdir, 'tvm_local')
sys.path.insert(0, os.path.join(tvmroot, 'python'))
sys.path.insert(0, os.path.join(tvmroot, 'topi/python'))
sys.path.insert(0, os.path.join(tvmroot, 'nnvm/python'))

import logging
import argparse
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
import time

# Two functions for reading data from record file or raw images
def get_val_data(args,
                 rec_val,
                 batch_size,
                 num_workers=4):
    rec_val = os.path.expanduser(rec_val)
    mean_rgb = [123.68, 116.779, 103.939]
    # std_rgb = [58.393, 57.12, 57.375]
    std_rgb = [1., 1., 1.]
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    img_size = 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,
        resize              = 256,
        data_shape          = (3, img_size, img_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return val_data, batch_fn


def evaluate(args, graph, lib, params, ctx):
    """Evaluate on the validation set."""
    import tvm
    from tvm.contrib import graph_runtime

    # setup dataset.
    batch_size = args.batch_size
    val_data, batch_fn = get_val_data(args, args.rec_val, batch_size)
    # create runtime module
    m = graph_runtime.create(graph, lib, ctx)
    # m.set_input(**params)
    m.load_params(params)
    oshape = (batch_size, args.num_classes)
    out_arr = tvm.nd.empty(oshape, "float32")
    # setup evaluaiton metric
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    top1, top5 = 0., 0.
    val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    # setup timer
    elapsed = 0.
    # Execute
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, [mx.cpu(0)])
        tic = time.time()
        m.run(data=data[0].asnumpy())
        m.get_output(0, out_arr)
        toc = time.time()
        elapsed += toc - tic
        acc_top1.update(label, [mx.nd.array(out_arr.asnumpy())])
        acc_top5.update(label, [mx.nd.array(out_arr.asnumpy())])

        if args.log_interval and not (i + 1) % args.log_interval:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            nsamples = (i + 1) * batch_size
            logging.info('[%d samples] validation: acc-top1=%f acc-top5=%f, speed=%.1ffps',
                         nsamples, top1, top5, 1./(elapsed/nsamples))
        if i >= 100:
            break
    logging.info('[final] validation: acc-top1=%f acc-top5=%f', top1, top5)
    with open('record.csv', "a") as f:
        f.write('{0}, {1}, {2}, {3}, {4}\n'.format(
            args.model, args.nbit_input, args.nbit_output, args.global_scale, top1))


def quantize_model(args):
    """Build with relay."""
    import tvm
    from tvm import relay
    from tvm.relay import quantize as qtz
    img_size = 224
    data_shape = (args.batch_size, 3, img_size, img_size)
    mx_sym, mx_args, mx_auxs = mx.model.load_checkpoint(args.model, 0)
    net, params = relay.frontend.from_mxnet(mx_sym, {"data": data_shape}, arg_params=mx_args, aux_params=mx_auxs)
    target = args.target

    if args.original:
        # run original model
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(net, target, params=params)
        ctx = tvm.nd.context(target, 0)
        return graph, lib, params, ctx

    # constant folding and scale folding.
    # print('original')
    # print(net.astext(show_meta_data=False))
    with relay.build_config(opt_level=3):
        qgraph = relay.optimize(net, target, params)
    # print('after optimize')
    # print(qgraph.astext(show_meta_data=False))

    with qtz.qconfig(skip_k_conv=0,
                     nbit_input=args.nbit_input,
                     nbit_weight=args.nbit_input,
                     global_scale=args.global_scale,
                     dtype_input=args.dtype_input,
                     dtype_weight=args.dtype_input,
                     dtype_activation=args.dtype_output,
                     store_lowbit_output=False,
                     debug_enabled_ops=None):
        print(qtz.current_qconfig())
        qgraph = qtz.annotate(qgraph)
        # print('after annotate')
        # print(qgraph.astext(show_meta_data=False))
        qgraph = qtz.calibrate(qgraph)
        # print('after calibrate\n')
        # print(qgraph.astext(show_meta_data=False))
        if not args.simulated:
            qgraph = qtz.realize(qgraph)
            qgraph = relay.ir_pass.infer_type(qgraph)
            # print('after realize\n')
            # print(qgraph.astext(show_meta_data=False))

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(qgraph, target)

    ### save/load the graph, lib and params into separate files
    # save
    lib.export_library(os.path.join(thisdir, "deploy_lib.so"))
    with open(os.path.join(thisdir, "deploy_graph.json"), "w") as fo:
        fo.write(graph)
    with open(os.path.join(thisdir, "deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(params))
    # load
    graph = open(os.path.join(thisdir, "deploy_graph.json")).read()
    lib = tvm.module.load(os.path.join(thisdir, "deploy_lib.so"))
    params = bytearray(open(os.path.join(thisdir, "deploy_param.params"), "rb").read())
    
    ctx = tvm.nd.context(target, 0)
    return graph, lib, params, ctx


def main(args):
    graph, lib, params, ctx = quantize_model(args)
    logging.info("Finish building model %s...", args.model)
    evaluate(args, graph, lib, params, ctx)


def run():
    parser = argparse.ArgumentParser(description="Evaluate ImageNet validation accuracy")
    parser.add_argument("--rec-val", type=str, default="~/.mxnet/datasets/imagenet/rec/val.rec",
                        help="the validation data")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="batch size")
    parser.add_argument("--model", type=str, default="models/imagenet1k-mnetv2-1_0",
                        help="Name of the model")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="log interval")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--target", type=str, default="llvm",
                        help="target option")
    parser.add_argument("--nbit-input", type=int, default=16,
                        help="number of input bits")
    parser.add_argument("--nbit-output", type=int, default=32,
                        help="number of output bits")
    parser.add_argument("--dtype-input", type=str, default="int16",
                        help="number of input bits")
    parser.add_argument("--dtype-output", type=str, default="int32",
                        help="number of output bits")
    parser.add_argument("--global-scale", type=float, default=256.0,
                        help="global activation scale")
    parser.add_argument("--original", action="store_true",
                        help='whether to use original graph')
    parser.add_argument("--simulated", action="store_true",
                        help='whether to use simulated graph')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(args)
    main(args)

# if __name__ == "__main__":
run()
