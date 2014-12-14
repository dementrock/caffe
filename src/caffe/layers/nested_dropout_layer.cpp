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
  // Not positive that this is the correct normalization. I don't think the
  // paper scales at all.
  // scale_ = dim * p_;
  scale_ = 1.0;
  unit_num_ = 0;
  converge_thresh_ = 1e-3;
}

template <typename Dtype>
void NestedDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // This vector holds the number of units to NOT mask for each input blob.
	// For each element e, this layer will dropout all but the first e channels/units.
  rand_vec_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void NestedDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int* mask_unit_num = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // For a fc layer output, num_pix should be one.
  const int num_pix = bottom[0]->width() * bottom[0]->height();
  const int num_channels = bottom[0]->channels();

  if (Caffe::phase() == Caffe::TRAIN) {
    // Create random number for each bottom.
    caffe_rng_geometric(num, p_, mask_unit_num, unit_num_);
    for (int i = 0; i < num; ++i) {
      // Scale or mask appropriately. Not sure if this is the best way to
      // access/change the data.
      // TODO - Vectorize this operation. (Construct a vector, mask and then
      // use axbpy to multiply the mask by the bottom data to produce the
      // top data.
      // For conv outputs, bottom_data will be a 4-D blob of num*c*w*h,
			// dropout by channel.
      // First scale the channels that are not being dropped out.
      if (mask_unit_num[i] > num_channels) {
        mask_unit_num[i] = num_channels;
      }
      for (int j = 0; j < mask_unit_num[i]; ++j) {
        Dtype* current_channel = top_data + top[0]->offset(i, j);
        const Dtype* current_bottom_channel = bottom_data + bottom[0]->offset(i, j);
        for (int k = 0; k < num_pix; ++k) {
          current_channel[k] = current_bottom_channel[k] * scale_;
        }
      }
      // Next set the rest of the channels to 0.
      for (int j = mask_unit_num[i]; j < num_channels; ++j) {
        Dtype* current_channel = top_data + top[0]->offset(i, j);
        for (int k = 0; k < num_pix; ++k) {
          current_channel[k] = Dtype(0);
        }
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
  std::cout << "Starting backward pass\n";
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    // For a fc layer output, num_pix should be one.
    const int num_pix = bottom[0]->width() * bottom[0]->height();

    if (Caffe::phase() == Caffe::TRAIN) {
      const int* mask_unit_num = rand_vec_.cpu_data();
      for (int i = 0; i < num; ++i) {
        // Set diff to 0 for first unit_num_ gradients, scale the middle, the mask the rest
				for (int j = 0; j < unit_num_; ++j) {
          Dtype* current_channel = bottom_diff + bottom[0]->offset(i, j);
					for (int k = 0; k < num_pix; ++k) {
            current_channel[k] = Dtype(0);
          }
				}
        for (int j = unit_num_; j < mask_unit_num[i]; ++j) {
          Dtype* current_channel = bottom_diff + bottom[0]->offset(i, j);
          const Dtype* current_top_channel =  top_diff + top[0]->offset(i, j);
          for (int k = 0; k < num_pix; ++k) {
            current_channel[k] = current_top_channel[k] * scale_;
          }
        }
        // Next set the rest of the channels to 0.
        for (int j = mask_unit_num[i]; j < top[0]->channels(); ++j) {
          Dtype* current_channel = bottom_diff + bottom[0]->offset(i, j);
          for (int k = 0; k < num_pix; ++k) {
            current_channel[k] = Dtype(0);
          }
        }
      }
			// ---- Code for unit sweeping ----
      // First check for converge of the channel/unit with number unit_num_:
      // If any of the gradients/diffs is larger than thresh, then we haven't
      // converged.
      bool converged = true;
      for (int i = 0; i < num; ++i) {
        const Dtype* top_unit_i = top_diff + top[0]->offset(i, unit_num_);
        if (caffe_cpu_asum(num_pix, top_unit_i) > converge_thresh_ * num_pix) {
          std::cout << "\nDid not converge, diff value:" << caffe_cpu_asum(num_pix, top_unit_i) << "\n";
          converged = false;
          break;
        }
        else {
          std::cout << "diff: " << caffe_cpu_asum(num_pix, top_unit_i) << ", ";
        }
      }
      if (converged) {
        std::cout << "Unit " << unit_num_ << " converged. :)\n";
        // Only increase if we have more channels that haven't converged.
        if (unit_num_ < bottom[0]->channels() - 1) {
          unit_num_++;
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
