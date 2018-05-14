#ifndef CAFFE_WINDOW_POSE_DATA_LAYER_HPP_
#define CAFFE_WINDOW_POSE_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
class PoseBatch {
 public:
  Blob<Dtype> data_, label_, e1_, e2_, e3_, e1coarse_, e2coarse_, e3coarse_;
};

/**
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file. This layer is *DEPRECATED* and only kept for
 *        archival purposes for use by the original R-CNN.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class WindowPoseDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit WindowPoseDataLayer(const LayerParameter& param);
  virtual ~WindowPoseDataLayer();
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WindowPoseData"; }
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 8; }

 protected:
  virtual void InternalThreadEntry();
  virtual unsigned int PrefetchRand();
  virtual void load_batch(PoseBatch<Dtype>* batch);

  vector<shared_ptr<PoseBatch<Dtype> > > prefetch_;
  BlockingQueue<PoseBatch<Dtype>*> prefetch_free_;
  BlockingQueue<PoseBatch<Dtype>*> prefetch_full_;
  PoseBatch<Dtype>* prefetch_current_;
  shared_ptr<Caffe::RNG> prefetch_rng_;

  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2,
      E1, E2, E3, E1C, E2C, E3C, E1M, E2M, E3M, E1CM, E2CM, E3CM, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  bool has_mean_file_;
  bool has_mean_values_;
  bool cache_images_;
  vector<std::pair<std::string, Datum > > image_database_cache_;
};

}  // namespace caffe

#endif  // CAFFE_WINDOW_POSE_DATA_LAYER_HPP_
