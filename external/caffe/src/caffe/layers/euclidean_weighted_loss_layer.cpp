#include <vector>

#include "caffe/layers/euclidean_weighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanWeightedLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  diffPos_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanWeightedLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  //int countLabel = 0;float countPred = 0;
  //for (int i=0; i<count; ++i){
  //    countLabel+= bottom[1]->cpu_data()[i];
  //    countPred+= bottom[0]->cpu_data()[i];
  //}
  //LOG(INFO)<<count<<'\n';
  //LOG(INFO)<<countLabel<<'\n';
  //LOG(INFO)<<countPred<<'\n';

  Dtype lambda = Dtype(this->layer_param_.euclidean_weight_param().lambda());
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_mul(count,diff_.cpu_data(),bottom[1]->cpu_data(),diffPos_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype dotPos = caffe_cpu_dot(count,diffPos_.cpu_data(), diffPos_.cpu_data());
  Dtype loss = (dot+lambda*dotPos) / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanWeightedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype lambda = Dtype(this->layer_param_.euclidean_weight_param().lambda());
  for (int i = 0; i < 1; ++i) {
    if (propagate_down[i]) {
      const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
      //LOG(INFO)<<alpha;
    caffe_cpu_axpby(
            bottom[0]->count(),              // count
            alpha,     // alpha
            diff_.cpu_data(),                   // a
            lambda*alpha,         // beta
            diffPos_.mutable_cpu_data());  // b

      caffe_cpu_axpby(
          bottom[0]->count(),              // count
          Dtype(1),                              // alpha
          diffPos_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[0]->mutable_cpu_diff());  // b
    }
  }
}

INSTANTIATE_CLASS(EuclideanWeightedLossLayer);
REGISTER_LAYER_CLASS(EuclideanWeightedLoss);

}  // namespace caffe
