#!/bin/bash

cmake ../cmake \
  -D BUILD_MPI=ON \
  -D PKG_GPU=ON \
  -D GPU_API=cuda \
  -D GPU_ARCH=sm_80 \
  -D GPU_PREC=mixed \
  -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..

  make -j32
