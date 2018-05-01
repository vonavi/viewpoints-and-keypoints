#ifndef CAFFE_WINDOW_MULTICLASS_KEYPOINT_DATA_LAYER_HPP_
#define CAFFE_WINDOW_MULTICLASS_KEYPOINT_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file. This layer is *DEPRECATED* and only kept for
 *        archival purposes for use by the original R-CNN.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class WindowMulticlassKeypointDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit WindowMulticlassKeypointDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~WindowMulticlassKeypointDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WindowMulticlassKeypointData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUMKPS, CSTART, CEND, TOTKPS, NUM };

  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
  vector<vector<int> > fg_windows_kps_;
  vector<vector<int> > bg_windows_kps_;
  vector<vector<float> > fg_windows_values_;
  vector<vector<float> > bg_windows_values_;
  vector<vector<int> > fg_windows_flipkps_;
  vector<vector<int> > bg_windows_flipkps_;

  Blob<Dtype> data_mean_;
  Blob<Dtype> prefetch_filter_;
  vector<Dtype> mean_values_;
  bool has_mean_file_;
  bool has_mean_values_;
  bool cache_images_;
  vector<std::pair<std::string, Datum > > image_database_cache_;
};

}  // namespace caffe

#endif  // CAFFE_WINDOW_MULTICLASS_KEYPOINT_DATA_LAYER_HPP_
