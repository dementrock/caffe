#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NestedDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  p_ = this->layer_param_.nested_dropout_param().geom_rate();
  DCHECK(p_ > 0.);
  // Maybe throw a warning if the parameter is equal to one. (Useless layer)
  DCHECK(p_ <= 1.);

  const int dim = bottom[0]->count() / bottom[0]->num();
  // Scale based on the expected value of units kept per input blob (1/p_).
  // Not positive that this is the correct normalization.
  scale_ = dim * p_;
}

template <typename Dtype>
void NestedDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // This vector holds the number of units to NOT mask for each input blob.
  rand_vec_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void NestedDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask_unit_num = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int size = count / num;
  if (Caffe::phase() == Caffe::TRAIN) {
    // Create random number for each bottom.
    caffe_rng_geometric(num, p_, mask_unit_num);
    for (int i = 0; i < num; ++i) {
      // Scale or mask appropriately. Not sure if this is the best way to
      // access/change the data.
      const Dtype* current_bottom = bottom_data + bottom[0]->offset(i);
      Dtype* current_top = top_data + top[0]->offset(i);
      for (int j = 0; j < mask_unit_num[i]; ++j) {
        current_top[j] = current_bottom[j] * scale_;
      }
      for (int j = mask_unit_num[i]; j < size; ++j) {
        current_top[j] = Dtype(0);
      }
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void NestedDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int size = count / num;
    if (Caffe::phase() == Caffe::TRAIN) {
      const unsigned int* masked_unit_num = rand_vec_.cpu_data();
      for (int i = 0; i < num; ++i) {
        // Scale or mask appropriately. Not sure if this is the best way to
        // access/change the data.
        Dtype* current_bottom = bottom_diff + bottom[0]->offset(i);
        const Dtype* current_top = top_diff + top[0]->offset(i);
        for (int j = 0; j < masked_unit_num[i]; ++j) {
          current_bottom[j] = current_top[j] * scale_;
        }
        for (int j = masked_unit_num[i]; j < size; ++j) {
          current_bottom[j] = Dtype(0);
        }
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NestedDropoutLayer);
#endif

INSTANTIATE_CLASS(NestedDropoutLayer);
REGISTER_LAYER_CLASS(NESTED_DROPOUT, NestedDropoutLayer);
}  // namespace caffe
