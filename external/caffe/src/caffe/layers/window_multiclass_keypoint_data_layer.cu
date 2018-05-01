#include <vector>

#include "caffe/layers/window_multiclass_keypoint_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void WindowMulticlassKeypointDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BasePrefetchingDataLayer<Dtype>::Forward_gpu(bottom, top);
  // Copy the data
  caffe_copy(this->prefetch_filter_.count(), this->prefetch_filter_.cpu_data(),
      top[2]->mutable_gpu_data());
}

INSTANTIATE_LAYER_GPU_FORWARD(WindowMulticlassKeypointDataLayer);

}  // namespace caffe
