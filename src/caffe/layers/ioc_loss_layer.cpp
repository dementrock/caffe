#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// bottom[0] is the demos, bottom[1] is the samples. Each are N x T
template <typename Dtype>
void IOCLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  T_ = bottom[0]->channels();
  nd_ = bottom[0]->num();
  ns_ = bottom[1]->num();
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  demo_counts_.Reshape(nd_, 1, 1, 1);
  sample_counts_.Reshape(ns_, 1, 1, 1);
  // temp_.Reshape(bottom[0]->num(), bottom[2]->channels(), 1, 1);
}

template <typename Dtype>
void IOCLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss(0.0);
  Dtype* dc = demo_counts_.mutable_cpu_data();
  Dtype* sc = sample_counts_.mutable_cpu_data();

  Dtype max_val;

  // Sum over time and compute max value for safe logsum.
  for (int i = 0; i < nd_; ++i) {
    demo_counts_[i] = 0;
    for (int t = 0; t < T_; ++t) {
      dc[i] += bottom[0]->data_at(i,t,0,0);
    }
    loss += dc[i];
  }
  // Divide by number of demos.
  loss /= (Dtype)nd_;

  Dtype max_val(-dc[0]);
  for (int i = 0; i < ns_; ++i) {
    sample_counts_[i] = 0;
    for (int t = 0; t < T_; ++t) {
      sc[i] += bottom[1]->data_at(i,t,0,0);
    }
    if (-sc[i] > max_val) max_val = -sc[i];
  }

  for (int i = 0; i < nd_; ++i) {
    if (-dc[i] > max_val) max_val = -dc[i];
  }

  // Do a safe log-sum-exp operation.
  for (int i = 0; i < nd_; ++i) {
    dc[i] = -dc[i] - max_val;
  }
  for (int i = 0; i < ns_; ++i) {
    sc[i] = -sc[i] - max_val;
  }

  caffe_exp(nd_, dc, dc);
  caffe_exp(ns_, sc, sc);

  Dtype temp_sum(0.0);
  for (int i = 0; i < nd_; ++i) temp_sum += dc[i];
  for (int i = 0; i < ns_; ++i) temp_sum += sc[i];

  loss -= log(temp_sum) + max_val;

  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void IOCLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  // Compute gradient w.r.t. demos


  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1. : -1.;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

      for (int n = 0; n < bottom[i]->num(); ++n) {
        for (int row = 0; row < bottom[2]->channels(); ++row) {
          Dtype temp = 0;
          for (int k = 0; k < bottom[2]->height(); ++k) {
            temp += bottom[2]->data_at(n, row, k, 0) * diff_.data_at(n, k, 0, 0);
          }
          *(bottom_diff + bottom[i]->offset(n, row)) = temp;
        }
      }
      caffe_scal(bottom[i]->count(), alpha, bottom_diff);
    }
  }
}

INSTANTIATE_CLASS(IOCLossLayer);
REGISTER_LAYER_CLASS(IOCLoss);
}  // namespace caffe
