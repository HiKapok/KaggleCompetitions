// kernel_example.cu.cc
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "segment_median_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void SegmentMedianCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * ldg(in + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename GPUDevice, typename T>
void SegmentMedianFunctor<GPUDevice, T>::operator()(const GPUDevice& d, typename TTypes<T>::Matrix data, std::vector<uint32_t> segment_ids, typename TTypes<T>::Matrix output_median) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  const int batch_size = data.dimension(0);
  const int class_num = data.dimension(1);

  int block_count = 1024;
  int thread_per_block = 20;
  SegmentMedianCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(1, nullptr, nullptr);
  // not implement yet
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct SegmentMedianFunctor<GPUDevice, float>;
template struct SegmentMedianFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
