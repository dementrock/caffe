#include <opencv2/core/core.hpp>

#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  const MemoryDataParameter param = this->layer_param_.memory_data_param();
  batch_size_ = param.data_shapes(0).dim(0);
  shuffle_ = param.shuffle();
  ioc_num_conditions_ = param.ioc_num_conditions();
  int num_top = param.data_shapes_size();
  // For more than one condition, assume that there are two tops (data & is)
  if (ioc_num_conditions_ > 1)
      num_top = 2;  // TODO - might not want to hardcode this - could remove this and change code below.
  for (int i = 0; i < num_top; ++i) {
    top[i]->Reshape(param.data_shapes(i));
    CHECK_EQ(batch_size_, param.data_shapes(i).dim(0)) <<
      "Inconsistent batch sizes across blobs in memory data layer";
    channels_.push_back(param.data_shapes(i).dim(1));
    height_.push_back(param.data_shapes(i).dim(2));
    width_.push_back(param.data_shapes(i).dim(3));
    size_.push_back(channels_[i] * height_[i] * width_[i]);
  }
  int num_bottom = num_top * ioc_num_conditions_;
  for (int i = 0; i < num_bottom; ++i) {
      data_.push_back(NULL);
  }
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(vector<Dtype*> data, int n) {
  if (!shuffle_)
    CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  for (int i = 0; i < data.size(); ++i) {
    CHECK(data[i]);
    data_[i] = data[i];
  }
  // Warn with transformation parameters since a memory array is meant to
  // be generic and no transformations are done with Reset().
  if (this->layer_param_.has_transform_param()) {
    LOG(WARNING) << this->type() << " does not transform array data on Reset()";
  }
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::SetBatchSize(int new_batch_size) {
  CHECK_GT(new_batch_size, 0) << "new batch size must be positive.";
  if (!shuffle_)
    CHECK_EQ(n_ % new_batch_size, 0) << "n must be multiple of batch size";
  batch_size_ = new_batch_size;
  pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    if (!shuffle_) {
        for (int i = 0; i < top.size(); ++i) {
            CHECK(data_[i]) << "MemoryDataLayer needs to be initalized by calling Reset";
            top[i]->Reshape(batch_size_, channels_[i], height_[i], width_[i]);
            top[i]->set_cpu_data(data_[i] + pos_ * size_[i]);
        }
        pos_ = (pos_ + batch_size_) % n_;
    } else if (shuffle_ && ioc_num_conditions_ == 0) {
        for (int b = 0; b < batch_size_; ++b) {
            int ind = rand() % n_;
            for (int i=0; i < top.size(); ++i) {
                Dtype* top_data = top[i]->mutable_cpu_data();
                if (b == 0) {
                    // Reshape top once
                    top[i]->Reshape(batch_size_, channels_[i], height_[i], width_[i]);
                }
                // TODO - will the below work??
                caffe_copy(size_[i], data_[i] + ind*size_[i], top_data + b*size_[i]);
            }
        }
    } else {  // shuffle_ && ioc_num_conditions_ > 0
        for (int b=0; b < batch_size_; ++b) {
            int ind = rand() % n_;
            // Assume that 2 inputs per condition, therefore 2 outputs.
            for (int top_i=0; top_i < 2; ++top_i) {
                int bot_i = 2 * pos_ + top_i;
                Dtype* top_data = top[top_i]->mutable_cpu_data();
                if (b==0) {
                    // Reshape each top once.
                    top[top_i]->Reshape(batch_size_, channels_[top_i], height_[top_i], width_[top_i]);
                }
                caffe_copy(size_[top_i], data_[bot_i] + ind*size_[top_i], top_data + b*size_[top_i]);
            }
        }
        // Update which condition we are on.
        pos_ = (pos_ + 1) % ioc_num_conditions_;
    }

}

INSTANTIATE_CLASS(MemoryDataLayer);
REGISTER_LAYER_CLASS(MemoryData);

}  // namespace caffe
