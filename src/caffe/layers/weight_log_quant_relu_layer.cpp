#include <algorithm>
#include <vector>

#include "caffe/layers/weight_log_quant_relu_layer.hpp"

namespace caffe {


template <typename Dtype>
void WeightLogQuantReLULayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	
	this->blobs_.resize(2);
	this->blobs_[0].reset(new Blob<Dtype>({2}));	// num_level, updated
	this->blobs_[1].reset(new Blob<Dtype>({256}));	// bdr, rep
	
	Dtype* data = this->blobs_[0]->mutable_cpu_data();
	data[0] = 8;	// default num_level
	data[1] = 1;	// uninitialized

	this->hist_.Reshape({1024});
	
	this->param_propagate_down_.push_back(false);
	this->param_propagate_down_.push_back(false);
}

template <typename Dtype>
void WeightLogQuantReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	NeuronLayer<Dtype>::Reshape(bottom, top);

}



INSTANTIATE_CLASS(WeightLogQuantReLULayer);
REGISTER_LAYER_CLASS(WeightLogQuantReLU);


}  // namespace caffe
