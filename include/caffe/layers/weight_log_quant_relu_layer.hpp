#ifndef CAFFE_WEIGHT_LOG_QUANT_RELU_LAYER_HPP_
#define CAFFE_WEIGHT_LOG_QUANT_RELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Rectified Linear Unit non-linearity @f$ y = \max(0, x) @f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */
template <typename Dtype>
class WeightLogQuantReLULayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit WeightLogQuantReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "WeightLogQuantReLU"; }
  
  	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	 	const vector<Blob<Dtype>*>& top);

 protected:
	 

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){assert(0);}
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
		{assert(0);}
		


  Blob<int> hist_;

  
};

}  // namespace caffe

#endif  // CAFFE_RELU_LAYER_HPP_
