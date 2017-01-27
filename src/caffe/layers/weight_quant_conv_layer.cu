#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/weight_quant_conv_layer.hpp"

namespace caffe {

__global__ void sync_conv_groups4() { }

template <typename Dtype>
void WeightQuantConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	//bit, option
	Dtype* quant_opt= 
        (this->bias_term_?this->blobs_[2]:this->blobs_[1])->mutable_cpu_data();
	const int opt = static_cast<int>(*(quant_opt));
	const int num_level = static_cast<int>(*(quant_opt+1));
	Dtype& updated = *(quant_opt+2);

	// quantization
	switch(opt)
	{
		case 0: 
			if(updated != 0)
			{
				updated = 0;
				
				eh_weight_quant_update<Dtype>(num_level,this->num_ch_, this->len_ch_,
					*(this->blobs_[0]), *(this->bias_term_?this->blobs_[4]:this->blobs_[3]),
					this->scale_, this->mult_tmp_, this->bdr_rep_,
					*(this->bias_term_?this->blobs_[3]:this->blobs_[2]), this->slice_);	
			}
			
			break;
		case 1:
		default:
			assert(0);
	}	
		
	const Dtype* weight_quant = (this->bias_term_?this->blobs_[4]:this->blobs_[3])->gpu_data();
	for (int i = 0; i < bottom.size(); ++i) 
	{
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();

		// Forward through cuDNN in parallel over groups.
		for (int g = 0; g < this->group_; g++) {
			// Filters.
			CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
				cudnn::dataType<Dtype>::one,
				bottom_descs_[i], bottom_data + bottom_offset_ * g,
				filter_desc_, weight_quant + this->weight_offset_ * g,
				conv_descs_[i],
				fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
				cudnn::dataType<Dtype>::zero,
				top_descs_[i], top_data + top_offset_ * g));

			// Bias.
			if (this->bias_term_) 
			{
				const Dtype* bias_data = this->blobs_[1]->gpu_data();
				CUDNN_CHECK(cudnnAddTensor(handle_[g],
					cudnn::dataType<Dtype>::one,
					bias_desc_, bias_data + bias_offset_ * g,
					cudnn::dataType<Dtype>::one,
					top_descs_[i], top_data + top_offset_ * g));
			}
		}

		// Synchronize the work across groups, each of which went into its own
		// stream, by launching an empty kernel into the default (null) stream.
		// NOLINT_NEXT_LINE(whitespace/operators)
		sync_conv_groups4<<<1, 1>>>();
	}
}

template <typename Dtype>
void WeightQuantConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	const Dtype* weight_quant = NULL;
	Dtype* weight_diff = NULL;

	if (this->param_propagate_down_[0]) {
		weight_quant = (this->bias_term_?this->blobs_[4]:this->blobs_[3])->gpu_data();
		weight_diff = this->blobs_[0]->mutable_gpu_diff();
	}

	Dtype* bias_diff = NULL;
	if (this->bias_term_ && this->param_propagate_down_[1]) {
		bias_diff = this->blobs_[1]->mutable_gpu_diff();
	}

	for (int i = 0; i < top.size(); ++i) 
	{
		const Dtype* top_diff = top[i]->gpu_diff();

		// Backward through cuDNN in parallel over groups and gradients.
		for (int g = 0; g < this->group_; g++) 
		{
			// Gradient w.r.t. bias.
			if (this->bias_term_ && this->param_propagate_down_[1]) 
			{
				CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
					cudnn::dataType<Dtype>::one,
					top_descs_[i],  top_diff + top_offset_ * g,
					cudnn::dataType<Dtype>::one,
					bias_desc_, bias_diff + bias_offset_ * g));
			}

			// Gradient w.r.t. weights.
    	  	if (this->param_propagate_down_[0]) 
		  	{
		  		// updated = 1
				(this->bias_term_?this->blobs_[2]:this->blobs_[1])->mutable_cpu_data()[2] = 1;
				const Dtype* bottom_data = bottom[i]->gpu_data();

				CUDNN_CHECK(cudnnConvolutionBackwardFilter(
					handle_[1*this->group_ + g],
					cudnn::dataType<Dtype>::one,
					bottom_descs_[i], bottom_data + bottom_offset_ * g,
					top_descs_[i],    top_diff + top_offset_ * g,
					conv_descs_[i],
					bwd_filter_algo_[i], workspace[1*this->group_ + g],
					workspace_bwd_filter_sizes_[i],
					cudnn::dataType<Dtype>::one,
					filter_desc_, weight_diff + this->weight_offset_ * g));
			}

			// Gradient w.r.t. bottom data.
			if (propagate_down[i]) 
			{
				if (weight_quant == NULL) 
				{
					weight_quant = (this->bias_term_?this->blobs_[4]:this->blobs_[3])->gpu_data();
				}

				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
				CUDNN_CHECK(cudnnConvolutionBackwardData(
					handle_[2*this->group_ + g],
					cudnn::dataType<Dtype>::one,
					filter_desc_, weight_quant + this->weight_offset_ * g,
					top_descs_[i], top_diff + top_offset_ * g,
					conv_descs_[i],
					bwd_data_algo_[i], workspace[2*this->group_ + g],
					workspace_bwd_data_sizes_[i],
					cudnn::dataType<Dtype>::zero,
					bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      		}
    	}

	    // Synchronize the work across groups, each of which went into its own
	    // stream, by launching an empty kernel into the default (null) stream.
	    // NOLINT_NEXT_LINE(whitespace/operators)
	    sync_conv_groups4<<<1, 1>>>();
  	}
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightQuantConvolutionLayer);

}  // namespace caffe
#endif
