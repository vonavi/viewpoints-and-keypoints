#include <vector>

#include "caffe/layers/window_pose_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void WindowPoseDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BasePrefetchingDataLayer<Dtype>::Forward_gpu(bottom, top);
  // Copy the data
  caffe_copy(this->prefetch_e1_.count(), this->prefetch_e1_.cpu_data(),
      top[2]->mutable_gpu_data());
  caffe_copy(this->prefetch_e2_.count(), this->prefetch_e2_.cpu_data(),
      top[3]->mutable_gpu_data());
  caffe_copy(this->prefetch_e3_.count(), this->prefetch_e3_.cpu_data(),
      top[4]->mutable_gpu_data());
  caffe_copy(this->prefetch_e1coarse_.count(), this->prefetch_e1coarse_.cpu_data(),
      top[5]->mutable_gpu_data());
  caffe_copy(this->prefetch_e2coarse_.count(), this->prefetch_e2coarse_.cpu_data(),
      top[6]->mutable_gpu_data());
  caffe_copy(this->prefetch_e3coarse_.count(), this->prefetch_e3coarse_.cpu_data(),
      top[7]->mutable_gpu_data());
}

INSTANTIATE_LAYER_GPU_FORWARD(WindowPoseDataLayer);

}  // namespace caffe
