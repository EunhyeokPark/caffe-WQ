#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/weight_quant_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightQuantInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
	    //bit, option
	Dtype* quant_opt= 
        (this->bias_term_?this->blobs_[2]:this->blobs_[1])->mutable_cpu_data();
	const int opt = static_cast<int>(*(quant_opt));
	const int num_level = static_cast<int>(*(quant_opt+1));	// or num_level
	Dtype& updated = *(quant_opt+2);

	// quantization
	switch(opt)
	{
		case 0: 
			if(updated != 0)
			{
				updated = 0;
				
				eh_weight_quant_update<Dtype>(num_level, this->num_ch_, this->len_ch_, 
					*(this->blobs_[0]), *(this->bias_term_?this->blobs_[4]:this->blobs_[3]), 
					this->scale_, this->mult_tmp_, this->bdr_rep_, 
					*(this->bias_term_?this->blobs_[3]:this->blobs_[2]), this->slice_);
			}

			break;
		case 1:
		default:
			assert(0);
	}

	
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* weight_quant = 
		(this->bias_term_?this->blobs_[4]:this->blobs_[3])->gpu_data();

	if (M_ == 1) 
	{
		caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
			weight_quant, bottom_data, (Dtype)0., top_data);

		if (bias_term_)
			caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
			this->blobs_[1]->gpu_data(), top_data);
	}
	else 
	{
		caffe_gpu_gemm<Dtype>(CblasNoTrans,
			transpose_ ? CblasNoTrans : CblasTrans,
			M_, N_, K_, (Dtype)1.,
			bottom_data, weight_quant, (Dtype)0., top_data);

		if (bias_term_)
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
			bias_multiplier_.gpu_data(),
			this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
	}
}

template <typename Dtype>
void WeightQuantInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
	if (this->param_propagate_down_[0]) 
	{
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* bottom_data = bottom[0]->gpu_data();

		// updated = 1
		(this->bias_term_?this->blobs_[2]:this->blobs_[1])->mutable_cpu_data()[2] = 1;
		// Gradient with respect to weight
	    if (transpose_) {
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				K_, N_, M_,
				(Dtype)1., bottom_data, top_diff,
				(Dtype)1., this->blobs_[0]->mutable_gpu_diff());
		} 
		else {
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				N_, K_, M_,
				(Dtype)1., top_diff, bottom_data,
				(Dtype)1., this->blobs_[0]->mutable_gpu_diff());
		}
	}
	
	if (bias_term_ && this->param_propagate_down_[1]) 
	{
		const Dtype* top_diff = top[0]->gpu_diff();

		// Gradient with respect to bias
		caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
			bias_multiplier_.gpu_data(), (Dtype)1.,
			this->blobs_[1]->mutable_gpu_diff());
	}

	if (propagate_down[0]) 
	{
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* weight_quant_data = 
			(this->bias_term_?this->blobs_[4]:this->blobs_[3])->gpu_data();

		// Gradient with respect to bottom data
		if (transpose_) 
		{
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
				M_, K_, N_,
				(Dtype)1., top_diff, weight_quant_data,
				(Dtype)0., bottom[0]->mutable_gpu_diff());
		} 
		else {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
				M_, K_, N_,
				(Dtype)1., top_diff, weight_quant_data,
				(Dtype)0., bottom[0]->mutable_gpu_diff());
		}
  	}
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightQuantInnerProductLayer);

}  // namespace caffe
