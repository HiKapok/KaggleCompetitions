CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(segment_median)

find_package(CUDA REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2 -gencode arch=compute_61,code=sm_61 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr")

# compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 ${OpenMP_CXX_FLAGS} -Wall -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")

# TensorFlow dependencies
EXECUTE_PROCESS(COMMAND python3.5 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_include(), end='', flush=True)"  OUTPUT_VARIABLE TF_INC)

EXECUTE_PROCESS(COMMAND python3.5 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_lib(), end='', flush=True)"  OUTPUT_VARIABLE TF_LIB)


MESSAGE(STATUS "Found TF_INC: " ${TF_INC})
MESSAGE(STATUS "Found TF_INC_EXTERNAL: " ${TF_INC}/external/nsync/public)
MESSAGE(STATUS "Found TF_LIB: " ${TF_LIB})


INCLUDE_DIRECTORIES(${TF_INC})
INCLUDE_DIRECTORIES(${TF_INC}/external/nsync/public)
LINK_DIRECTORIES(${TF_LIB})


# approach 1
# CUDA_ADD_LIBRARY(segment_median_gpu SHARED segment_median_op.cu OPTIONS -I$TF_INC/tensorflow/stream_executor/cuda -I/usr/local)

# ADD_LIBRARY(segment_median SHARED
#   segment_median_op.h
#   segment_median_op.cc
#   )

# TARGET_LINK_LIBRARIES(segment_median tensorflow_framework ${CUDA_LIBRARIES} segment_median_gpu)

# approach 2
CUDA_COMPILE(SEGMENT_MEDIAN_CU_O segment_median_op.cu MODULE OPTIONS -I$TF_INC/tensorflow/stream_executor/cuda -I/usr/local)

ADD_LIBRARY(segment_median SHARED
  ${SEGMENT_MEDIAN_CU_O}
  segment_median_op.h
  segment_median_op.cc
  )

TARGET_LINK_LIBRARIES(segment_median tensorflow_framework ${CUDA_LIBRARIES})


 #nvcc -std=c++11 -c -o cuda_op_kernel.cu.o ../segment_median_op.cu -I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I/usr/local/ -I$TF_INC/tensorflow/stream_executor/cuda --expt-relaxed-constexpr -gencode arch=compute_61,code=sm_61


#TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2
#nvcc -std=c++11 -c -o cuda_op_kernel.cu.o ../segment_median_op.cu -I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I/usr/local/ -I$TF_INC/tensorflow/stream_executor/cuda
#cd `python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`
#cd tensorflow/stream_executor/cuda
#curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/master/third_party/toolchains/gpus/cuda/cuda/cuda_config.h
