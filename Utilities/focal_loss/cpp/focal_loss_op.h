// kernel_example.h
#ifndef KERNEL_FOCAL_LOSS_H_
#define KERNEL_FOCAL_LOSS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"

using tensorflow::TTypes;
using tensorflow::OpKernelContext;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename T>
struct FocalLossFunctor {
  void operator()(OpKernelContext* context, const Device& d, typename TTypes<T>::ConstMatrix logits, typename TTypes<T>::ConstMatrix labels, \
      typename TTypes<T>::Matrix softmax, typename TTypes<T>::Vec alpha, typename TTypes<T>::ConstScalar gamma, typename TTypes<T>::Vec focal_loss);
};

template <typename Device, typename T>
struct FocalLossGradFunctor {
  void operator()(const Device& d, typename TTypes<T>::ConstMatrix logits, typename TTypes<T>::ConstVec labels, typename TTypes<T>::Matrix softmax, \
      typename TTypes<T>::Vec alpha, typename TTypes<T>::ConstScalar gamma, typename TTypes<T>::ConstVec focal_loss, typename TTypes<T>::Matrix grads);
};


#if GOOGLE_CUDA
template <typename T>
struct FocalLossFunctorGPU {
  void operator()(OpKernelContext* context, const GPUDevice& d, typename TTypes<T>::ConstMatrix logits, typename TTypes<T>::ConstMatrix labels, \
      typename TTypes<T>::Matrix softmax, typename TTypes<T>::Vec alpha, typename TTypes<T>::ConstScalar gamma, typename TTypes<T>::Vec focal_loss);
};
#endif

#if GOOGLE_CUDA
template <typename T>
struct FocalLossGradFunctorGPU {
  void operator()(const GPUDevice& device, typename TTypes<T>::ConstMatrix logits, typename TTypes<T>::ConstVec labels, typename TTypes<T>::Matrix softmax, \
      typename TTypes<T>::Vec alpha, typename TTypes<T>::ConstScalar gamma, typename TTypes<T>::ConstVec focal_loss, typename TTypes<T>::Matrix grads);
};
#endif

#endif // KERNEL_FOCAL_LOSS_H_

