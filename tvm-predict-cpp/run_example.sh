#!/bin/bash
DIR=$(dirname $(realpath $0))
echo "Copy deployed modules ..."
cp $DIR/../deploy/mobilenetv2-1_0/* .
echo "Build the libraries ..."
make
echo "Run the example"
export LD_LIBRARY_PATH=$DIR/tvm_local/lib:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=$DIR/tvm_local/lib:${DYLD_LIBRARY_PATH}

echo "Run the deployment with all in one packed library..."
./nnvm_run
