// kernel_example.h
#ifndef KERNEL_FOCAL_LOSS_H_
#define KERNEL_FOCAL_LOSS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

#include <vector>
#include <cstdint>

using tensorflow::TTypes;

// Functor used by SoftmaxOp to do the computations.
template <typename Device, typename T>
struct SegmentMedianFunctor {
  // Computes SegmentMedian.
  //
  // data: dim: batch_size, num_classes. (we assume logits here)
  // segment_ids: dim: batch_size.
  void operator()(const Device& d, typename TTypes<T>::Matrix data, std::vector<uint32_t> segment_ids, typename TTypes<T>::Matrix output_median);
};


#endif // KERNEL_FOCAL_LOSS_H_

