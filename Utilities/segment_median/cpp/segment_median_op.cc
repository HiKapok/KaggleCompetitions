// kernel_example.cc
#include "segment_median_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>
#include <algorithm>
#include <thread>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


// in fact, we don't known the value of output shape at runtime
// no need for int32, just for test
REGISTER_OP("SegmentMedian")
    .Attr("T: {float, int32}")
    .Input("data: T")
    .Input("segment_ids: T")
    .Output("seg_median: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle data_shape = c->input(0);
      shape_inference::DimensionHandle num_per_batch = c->Dim(data_shape, 0);
      shape_inference::DimensionHandle num_classes = c->Dim(data_shape, 1);
      // use MakeShape to create a ShapeHandle from one DimensionHandle
      c->set_output(0, c->MakeShape({c->UnknownDim(), num_classes}));

      // one can use following function to make more check on input shape
      // use WithValue check DimensionHandle, and use WithRank check ShapeHandle
      // TF_RETURN_IF_ERROR(c->WithRank(logits_shape, 2, &logits_shape));
      // TF_RETURN_IF_ERROR(c->WithValue(num_per_batch, 128, &num_per_batch));
      return Status::OK();
    });


// CPU specialization of actual computation.
template <typename T>
struct SegmentMedianFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Matrix data_transposed, std::vector<uint32_t> segment_ids, typename TTypes<T>::Matrix output_median) {
    const int32_t batch_size = data_transposed.dimension(1);
    const int32_t num_classes = data_transposed.dimension(0);

    T * data_internal = data_transposed.data();
    auto find_median_func = [&](int32_t begin_index, int32_t end_index){
      for(int32_t index = begin_index;index < end_index;++index){
        int32_t seg_begin = index * batch_size;
        int32_t seg_end = index * batch_size;
        uint32_t last_seg_len = 0;
        uint32_t seg_index = 0;

        for(auto & seg_len : segment_ids){
          seg_begin += last_seg_len;
          seg_end += seg_len;
          last_seg_len = seg_len;

          std::nth_element(data_internal + seg_begin, data_internal + seg_begin + seg_len/2, data_internal + seg_end);
          T* median_ptr = data_internal + seg_begin + seg_len/2;
          T median = static_cast<T>(seg_len % 2 == 0 ? (*median_ptr + *(median_ptr-1))/2. : *median_ptr);
          output_median(seg_index++, index) = median;
        }
      }
    };
    //find_median_func(0, num_classes);
    int32_t num_core = std::max(static_cast<int32_t>(std::thread::hardware_concurrency()), 1);
    if(num_classes < num_core) num_core = num_classes;

    std::vector<std::thread> thread_vec;
    //Note: all tensorflow tensor returned matrix are RowMajor
    for(int32_t index = 0;index < num_core - 1;++index){
      thread_vec.emplace_back(find_median_func, num_classes * index / num_core, num_classes * (index+1) / num_core);
    }
    find_median_func(num_classes * (num_core - 1) / num_core, num_classes);
    for(auto & t : thread_vec) if(t.joinable()) t.join();

  }
};

// template <typename T>
// struct SegmentMedianFunctor<CPUDevice, T> {
//   void operator()(const CPUDevice& d, typename TTypes<T>::Matrix data_transposed, std::vector<uint32_t> segment_ids, typename TTypes<T>::Matrix output_median) {

//     const int batch_size = data_transposed.dimension(1);
//     const int num_classes = data_transposed.dimension(0);

//     Eigen::DSizes<int, 1> along_class(1);
//     Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
//     Eigen::DSizes<int, 2> one_by_class(1, num_classes);

//     T * data_internal = data_transposed.data();

//     auto find_median_func = [](){

//     };
//     //Note: all tensorflow tensor returned matrix are RowMajor
//     for(int32_t index = 0;index < num_classes;++index){
//       int32_t seg_begin = index * batch_size;
//       int32_t seg_end = index * batch_size;
//       uint32_t last_seg_len = 0;
//       uint32_t seg_index = 0;

//       for(auto & seg_len : segment_ids){
//         seg_begin += last_seg_len;
//         seg_end += seg_len;
//         last_seg_len = seg_len;

//         std::nth_element(data_internal + seg_begin, data_internal + seg_begin + seg_len/2, data_internal + seg_end);
//         T* median_ptr = data_internal + seg_begin + seg_len/2;
//         T median = static_cast<T>(seg_len % 2 == 0 ? (*median_ptr + *(median_ptr-1))/2. : *median_ptr);
//         output_median(seg_index++, index) = median;
//       }

//     }
//   }
// };



//OpKernel definition.
//template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class SegmentMedianOp : public OpKernel {
 public:
  explicit SegmentMedianOp(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    const Tensor& data_in = context->input(0);
    const Tensor& segment_ids_in = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(data_in.shape()),
                errors::InvalidArgument("data must be 2-dimensional"));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids_in.shape()),
                errors::InvalidArgument("segment_ids must be a vecotr"));

    const int batch_size = data_in.dim_size(0);
    const int num_classes = data_in.dim_size(1);

    OP_REQUIRES(context, segment_ids_in.dim_size(0) == data_in.dim_size(0), errors::InvalidArgument("segment_ids must be vector of size batch"));

    std::vector<uint32_t> seg_vec = num_segments(segment_ids_in.vec<T>());

    Tensor* seg_median = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {seg_vec.size(), num_classes}, &seg_median));


    SegmentMedianFunctor<Device, T> functor;

    Tensor transposed_data;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({num_classes, batch_size}), &transposed_data));

    transposed_data.matrix<T>() = data_in.matrix<T>().shuffle(Eigen::array<int, 2>{1, 0});

    functor(context->eigen_device<Device>(), transposed_data.matrix<T>(), seg_vec, seg_median->matrix<T>());

  }

private:

  std::vector<uint32_t> num_segments(typename TTypes<T>::ConstVec segment_ids){
    uint32_t num_elem = segment_ids.dimension(0);
    if(num_elem < 1) return {};
    uint32_t counter = 0;
    T last_elem = segment_ids(0);
    std::vector<uint32_t> how_many_vec;
    how_many_vec.reserve(num_elem + 1);
    for(uint32_t index = 0;index < num_elem;++index){
        if(segment_ids(index) != last_elem){
            how_many_vec.push_back(counter);
            counter = 0;
            last_elem = segment_ids(index);
        }
        counter++;
    }
    how_many_vec.push_back(counter);
    return std::move(how_many_vec);
}
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SegmentMedian").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SegmentMedianOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template SegmentMedianFunctor<GPUDevice, float>;              \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SegmentMedian").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      SegmentMedianOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
