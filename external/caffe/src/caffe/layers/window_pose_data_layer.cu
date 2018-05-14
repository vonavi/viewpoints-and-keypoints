#include <vector>

#include "caffe/layers/window_pose_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void WindowPoseDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
    // Copy the data
    top[2]->ReshapeLike(prefetch_current_->e1_);
    top[2]->set_gpu_data(prefetch_current_->e1_.mutable_gpu_data());
    top[3]->ReshapeLike(prefetch_current_->e2_);
    top[3]->set_gpu_data(prefetch_current_->e2_.mutable_gpu_data());
    top[4]->ReshapeLike(prefetch_current_->e3_);
    top[4]->set_gpu_data(prefetch_current_->e3_.mutable_gpu_data());
    top[5]->ReshapeLike(prefetch_current_->e1coarse_);
    top[5]->set_gpu_data(prefetch_current_->e1coarse_.mutable_gpu_data());
    top[6]->ReshapeLike(prefetch_current_->e2coarse_);
    top[6]->set_gpu_data(prefetch_current_->e2coarse_.mutable_gpu_data());
    top[7]->ReshapeLike(prefetch_current_->e3coarse_);
    top[7]->set_gpu_data(prefetch_current_->e3coarse_.mutable_gpu_data());
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(WindowPoseDataLayer);

}  // namespace caffe
