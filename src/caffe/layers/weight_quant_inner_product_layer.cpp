#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/weight_quant_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightQuantInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

	// EH_MODIFY
	// quantize_initialization

	this->num_ch_ = this->blobs_[0]->shape()[0];
	this->len_ch_ = this->blobs_[0]->count() / this->num_ch_;
	
	this->scale_.Reshape({this->num_ch_});
	this->mult_tmp_.Reshape({this->len_ch_});
	
	caffe_set<Dtype>(this->mult_tmp_.count(), 1.,
		this->mult_tmp_.mutable_cpu_data());
	
	if(this->bias_term_ )
	{
		if(this->blobs_.size() == 2)
		{
		  	this->blobs_.resize(5);
			
			// weight bit, option
			this->blobs_[2].reset(new Blob<Dtype>({3}));	

			Dtype* quant_opt = this->blobs_[2]->mutable_cpu_data();
			quant_opt[0] = 0;	// option
			quant_opt[1] = 8;	// num_level
			quant_opt[2] = 1;	// updated

			// store slice information
			this->blobs_[3].reset(new Blob<Dtype>({64}));
			this->blobs_[4].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
		}		
		this->param_propagate_down_.resize(2,true);
		this->param_propagate_down_.push_back(false);	// opt is not update through back prop
		this->param_propagate_down_.push_back(false);	// quantize_base is not update through back prop
		this->param_propagate_down_.push_back(false);
		
	}
	else
	{
		if(this->blobs_.size() == 1)
		{
			this->blobs_.resize(4);

			// weight bit, option
			this->blobs_[1].reset(new Blob<Dtype>({3}));	

			Dtype* quant_opt = this->blobs_[1]->mutable_cpu_data();
			quant_opt[0] = 0;	// option
			quant_opt[1] = 8;	// num_level
			quant_opt[2] = 1;	// updated

			// store slice information
			this->blobs_[2].reset(new Blob<Dtype>({64}));
			this->blobs_[3].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
		}
		
		this->param_propagate_down_.resize(1,true);
		this->param_propagate_down_.push_back(false);	// opt is not update through back prop
		this->param_propagate_down_.push_back(false);	// quantize_base is not update through back prop
		this->param_propagate_down_.push_back(false);
		
	}  
}

template <typename Dtype>
void WeightQuantInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

	// EH_MODIFY
  	// num_level
	const Dtype* quant_opt= 
        (this->bias_term_?this->blobs_[2]:this->blobs_[1])->cpu_data();
	const int num_level = static_cast<int>(*(quant_opt+1));

	int opt = static_cast<int>(*(quant_opt));

	if(opt == 0)
	{	
		int half_level_ = (num_level+1)>>1;
		this->slice_.resize(half_level_);
		this->bdr_rep_.Reshape({half_level_});
	}
	else
	{
		this->slice_.resize(num_level);
		this->bdr_rep_.Reshape({num_level});
	}
}

template <typename Dtype>
void WeightQuantInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{	assert(0); // cpu version is not implemented
	
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void WeightQuantInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
	assert(0); // cpu version is not implemented
	
	if (this->param_propagate_down_[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightQuantInnerProductLayer);
#endif

INSTANTIATE_CLASS(WeightQuantInnerProductLayer);
REGISTER_LAYER_CLASS(WeightQuantInnerProduct);

}  // namespace caffe
