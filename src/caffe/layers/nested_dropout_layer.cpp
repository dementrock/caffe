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
	// For a fc layer output, num_pix should be one.
	const int num_pix = bottom[0]->width() * bottom[0]->height();

  if (Caffe::phase() == Caffe::TRAIN) {
    // Create random number for each bottom.
    caffe_rng_geometric(num, p_, mask_unit_num, unit_num_);
    for (int i = 0; i < num; ++i) {
      // Scale or mask appropriately. Not sure if this is the best way to
      // access/change the data.
			// TODO - Vectorize this operation. (Construct a vector, mask and then
			// use axbpy to multiply the mask by the bottom data to produce the
			// top data.
			// Note: this assumes bottom_data to be a 2-D blob of num*d dimension.
			// For conv outputs, bottom_data will be a 4-D blob of num*c*w*h.
			// In this case, we want to dropout by channel rather than by d.
      /* const Dtype* current_bottom = bottom_data + bottom[0]->offset(i);
      Dtype* current_top = top_data + top[0]->offset(i);
      for (int j = 0; j < mask_unit_num[i]; ++j) {
        current_top[j] = current_bottom[j] * scale_;
      }
      for (int j = mask_unit_num[i]; j < size; ++j) {
        current_top[j] = Dtype(0);
      } */
			// New code for conv:
			// First scale the channels that are not being dropped out.
      for (int j = 0; j < mask_unit_num[i]; ++j) {
				Dtype* current_channel = top_data + top[0]->offset(i, j);
				const Dtype* current_bottom_channel = bottom_data + bottom[0]->offset(i, j);
				for (int k = 0; k < num_pix; ++k) {
					current_channel[k] = current_bottom_channel[k] * scale_;
				}
			}
			// Next set the rest of the channels to 0.
			for (int j = mask_unit_num[i]; j < size; ++j) {
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
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
		CHECK(num == 1); // Assuming this doesn't fail, TODO remove loop. If it fails, fix convergence check.
    const int size = count / num;
  	// For a fc layer output, num_pix should be one.
	  const int num_pix = bottom[0]->width() * bottom[0]->height();

    if (Caffe::phase() == Caffe::TRAIN) {
			// First check for converge of the channel/unit with number unit_num_:
			const Dtype* first_top_unit = top_diff + top[0]->offset(0, unit_num_);
			if (caffe_cpu_asum(num_pix, first_top_unit) < converge_thresh_ * num_pix) {
				DLOG(INFO) << "Unit " << unit_num_ << " converged. :)";
				unit_num_++;
			}

      const unsigned int* masked_unit_num = rand_vec_.cpu_data();
      for (int i = 0; i < num; ++i) {
        // Scale or mask appropriately. Not sure if this is the best way to
        // access/change the data.
        /* Dtype* current_bottom = bottom_diff + bottom[0]->offset(i);
        const Dtype* current_top = top_diff + top[0]->offset(i);
        for (int j = 0; j < masked_unit_num[i]; ++j) {
          current_bottom[j] = current_top[j] * scale_;
        }
        for (int j = masked_unit_num[i]; j < size; ++j) {
          current_bottom[j] = Dtype(0);
        } */
				// New code for conv layer (but also still works with fc layer)
      	for (int j = 0; j < mask_unit_num[i]; ++j) {
					Dtype* current_channel = bottom_diff + bottom[0]->offset(i, j);
					const Dtype* current_top_channel =  top_diff + top[0]->offset(i, j);
					for (int k = 0; k < num_pix; ++k) {
						current_channel[k] = current_top_channel[k] * scale_;
					}
				}
				// Next set the rest of the channels to 0.
				for (int j = mask_unit_num[i]; j < size; ++j) {
					Dtype* current_channel = bottom_diff + bottom[0]->offset(i, j);
					for (int k = 0; k < num_pix; ++k) {
						current_channel[k] = Dtype(0);
					}
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
