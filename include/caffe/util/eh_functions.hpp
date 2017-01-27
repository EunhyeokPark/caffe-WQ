#ifndef CAFFE_UTIL_EH_FUNCTIONS_H_
#define CAFFE_UTIL_EH_FUNCTIONS_H_

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe
{	
	template <typename Dtype>
	void eh_weight_quant_update(
		const int num_level, const int num_ch, const int len_ch,
		Blob<Dtype>& in_blob, Blob<Dtype>& out_blob,
		Blob<Dtype>& scale_blob, Blob<Dtype>& mult_tmp_blob, 
		Blob<Dtype>& bdr_rep_blob, Blob<Dtype>& slice_blob, std::vector<int>& slice_vec);


}
#endif
