# Makefile Example to deploy TVM modules.
DIR=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TVM_ROOT=${DIR}/../tvm_local
NNVM_PATH=nnvm
DMLC_CORE=${TVM_ROOT}/dmlc-core

PKG_CFLAGS = -std=c++11 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/dlpack/include\

PKG_LDFLAGS = -L${TVM_ROOT}/lib -ldl -lpthread -ltvm_runtime # -ltvm

.PHONY: clean all

all: nnvm_run

# Build rule for all in one TVM package library
.obj/nnvm_run.o: nnvm_run.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^

nnvm_run: .obj/nnvm_run.o
	$(CXX) $(PKG_LDFLAGS) -o $@  $^

clean:
	rm nnvm_run.o nnvm_run
