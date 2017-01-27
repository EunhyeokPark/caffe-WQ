#include "caffe/util/eh_functions.hpp"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

#include<iostream>

namespace caffe
{


template <typename Dtype>
struct QuantInfo
{
	Dtype* sum_;
	int* slice_;
	int num_level_;
	int num_param_;

	QuantInfo(Dtype* sum, int* slice, int num_level, int num_param):
		sum_(sum), slice_(slice), num_level_(num_level), num_param_(num_param){}

	int half_level(){return (num_level_+1)/2;}
	Dtype p(int n)
	{
		if(n == 0)
			return ((Dtype)slice_[0]) / num_param_;
		else
			return ((Dtype)(slice_[n] - slice_[n-1])) / num_param_;
	}
	
	Dtype wp(int n)
	{
		if(n==0)
			return sum_[slice_[0]-1] / num_param_;
		else
			return (sum_[slice_[n]-1] - sum_[slice_[n-1]-1]) / num_param_;
	}

	Dtype w(int n)
	{
		if(n==0)
			return sum_[slice_[0]-1] / (slice_[0]);
		else
			return (sum_[slice_[n]-1] - sum_[slice_[n-1]-1]) / (slice_[n] - slice_[n-1]);
	}

	Dtype wEnt()
	{
		Dtype ent = 0;

		for(int i = (num_level_%2) ; i < half_level(); ++i)
		{
			ent -= wp(i)*log(p(i)/2);
		}
		return ent;
	}

	Dtype pEnt(int n)
	{
		Dtype ent = 0;

		for(int i = max(n,(num_level_%2)) ; i < n+2; ++i)
		{
			ent -= wp(i)*log(p(i)/2);
		}
		return ent;
	}
};


template <typename Dtype>
void recUpdate(int n, QuantInfo<Dtype>* info, int begin = 0, int end = 0)
{
	if((begin == 0) && (end == 0))
	{
		if(n==0)
		{
			int new_begin = 0;
			int new_end = info->slice_[1]-1;

			recUpdate(n, info, new_begin, new_end);
		}
		else
		{
			int new_begin = info->slice_[n-1];
			int new_end = info->slice_[n+1]-1;

			recUpdate(n, info, new_begin, new_end);
		}
	}
	else if( (end-begin) == 1)
	{
		info->slice_[n] = begin;
		Dtype b_ent = info->pEnt(n);
		info->slice_[n] = end;
		Dtype e_ent = info->pEnt(n);

		info->slice_[n] = (b_ent > e_ent) ? begin : end;
	}
	else
	{
		int center = (begin+end)/2;
		info->slice_[n] = center;
		Dtype f_ent = info->pEnt(n);
		info->slice_[n] = center+1;
		Dtype b_ent = info->pEnt(n);
		
		if(f_ent > b_ent)
		{	recUpdate(n, info, begin, center); }
		else
		{	recUpdate(n, info, center, end); }
		
	}
}

template <typename Dtype>
__global__ void eh_channel_wise_scale_kernel(
	const int n, const int len_ch, const Dtype* scale, Dtype* data)
{
	CUDA_KERNEL_LOOP(i, n) 	
	{
		int ch_idx = i / len_ch;
		data[i] /= scale[ch_idx];
	}
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, Dtype* data)
{
	CUDA_KERNEL_LOOP(i, n) 	
	{
		data[i] = sqrt(data[i]);
	}
	
}

template <typename Dtype>
__global__ void eh_weight_quant_scale_forward_kernel(
	const int n, const int half_level, const int len_ch, const Dtype* scale, 
	const Dtype* in, Dtype* out, const Dtype* quant_bdr, const Dtype* quant_rep)
{
	CUDA_KERNEL_LOOP(i, n) 	
	{		
		int ch_idx = i/len_ch;
		
		Dtype sign = in[i] >= 0 ? 1 : -1;
		Dtype value = in[i] * sign / scale[ch_idx];
		Dtype out_temp = 0;
		
		for(int j = 0; j < half_level; ++j)
		{
			if(value >= quant_bdr[j])
			{
				out_temp = quant_rep[j];
			}
		}	
		
		out[i] = out_temp * sign * scale[ch_idx];
	}
}

/*
// original L2-L2 optimization

template <typename Dtype>
void eh_weight_quant_update(
	const int num_level, const int num_ch, const int len_ch,
	Blob<Dtype>& in_blob, Blob<Dtype>& out_blob,
	Blob<Dtype>& scale_blob, Blob<Dtype>& mult_tmp_blob, 
	Blob<Dtype>& bdr_rep_blob, Blob<Dtype>& slice_blob, std::vector<int>& slice_vec)
{
	int half_level = (num_level+1) /2;
	int n = in_blob.count(); 

	// get scale value	
	const Dtype* in = in_blob.gpu_data();
	Dtype* temp = out_blob.mutable_gpu_data();
	Dtype* scale = scale_blob.mutable_gpu_data();
	const Dtype* mult_tmp = mult_tmp_blob.gpu_data();

	caffe_gpu_powx<Dtype>(n, in, 2, temp);
	caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ch, len_ch, 1.,
		temp, mult_tmp, 0., scale);

	caffe_gpu_scal<Dtype>(num_ch, 1./len_ch, scale);

	// channel_wise scaling
	eh_channel_wise_scale_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>
		(n, len_ch, scale, temp);

	// find optimum slice
	thrust::device_ptr<Dtype> temp_d = thrust::device_pointer_cast(temp);
	thrust::sort(temp_d, temp_d + n);
	thrust::inclusive_scan(temp_d, temp_d + n, temp_d);	// sum1

	temp = out_blob.mutable_cpu_data();

	Dtype* slice = slice_blob.mutable_cpu_data();
	
	for(int i = 0 ; i < half_level; ++i)
	{
		slice_vec[i] = static_cast<int>(slice[i]);
	}

	QuantInfo<Dtype> info(temp, slice_vec.data(), num_level, n);	
	Dtype new_ent = info.wEnt();
	Dtype prev_ent = 0;

	while(new_ent > prev_ent)
	{
		for(int i = 0; i < half_level-1; ++i)
		{
			recUpdate(i, &info);
		}

		prev_ent = new_ent;
		new_ent = info.wEnt();
	}

	// slice update
	for(int i = 0 ; i < half_level; ++i)
	{
		slice[i] = static_cast<Dtype>(slice_vec[i]);
	}
	
	// get boundary and representative value
	Dtype* bdr = bdr_rep_blob.mutable_cpu_data();
	Dtype* rep = bdr_rep_blob.mutable_cpu_diff();

	rep[0] = 0;
	for(int i = (num_level%2) ; i < half_level; ++i)
	{
		rep[i] = sqrt(info.w(i));
	}

	bdr[0] = 0;
	for(int i = 0 ; i < half_level-1 ; ++i)
	{
		bdr[i+1] = sqrt(temp[slice_vec[i]] - temp[slice_vec[i]-1]);
	}	
		
	// apply forward kernel 
	sqrt_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_ch), CAFFE_CUDA_NUM_THREADS>>>(num_ch, scale);
	bdr = bdr_rep_blob.mutable_gpu_data();
	rep = bdr_rep_blob.mutable_gpu_diff();

	Dtype* out = out_blob.mutable_gpu_data();

	eh_weight_quant_scale_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>
		(n, half_level, len_ch, scale, in, out, bdr, rep);	
}
*/

template <typename Dtype>
void eh_weight_quant_update(
	const int num_level, const int num_ch, const int len_ch,
	Blob<Dtype>& in_blob, Blob<Dtype>& out_blob,
	Blob<Dtype>& scale_blob, Blob<Dtype>& mult_tmp_blob, 
	Blob<Dtype>& bdr_rep_blob, Blob<Dtype>& slice_blob, std::vector<int>& slice_vec)
{
	int half_level = (num_level+1) /2;
	int n = in_blob.count(); 

	// get scale value	
	const Dtype* in = in_blob.gpu_data();
	Dtype* temp = out_blob.mutable_gpu_data();
	Dtype* scale = scale_blob.mutable_gpu_data();

	caffe_gpu_powx<Dtype>(n, in, 2, temp);

	caffe_gpu_set<Dtype>(num_ch, 1., scale);

	// find optimum slice
	thrust::device_ptr<Dtype> temp_d = thrust::device_pointer_cast(temp);
	thrust::sort(temp_d, temp_d + n);
	thrust::inclusive_scan(temp_d, temp_d + n, temp_d);	// sum1

	temp = out_blob.mutable_cpu_data();

	Dtype* slice = slice_blob.mutable_cpu_data();
	
	for(int i = 0 ; i < half_level; ++i)
	{
		slice_vec[i] = static_cast<int>(slice[i]);
	}

	QuantInfo<Dtype> info(temp, slice_vec.data(), num_level, n);	
	Dtype new_ent = info.wEnt();
	Dtype prev_ent = 0;

	while(new_ent > prev_ent)
	{
		for(int i = 0; i < half_level-1; ++i)
		{
			recUpdate(i, &info);
		}

		prev_ent = new_ent;
		new_ent = info.wEnt();
	}

	// slice update
	for(int i = 0 ; i < half_level; ++i)
	{
		slice[i] = static_cast<Dtype>(slice_vec[i]);
	}
	
	// get boundary and representative value
	Dtype* bdr = bdr_rep_blob.mutable_cpu_data();
	Dtype* rep = bdr_rep_blob.mutable_cpu_diff();

	rep[0] = 0;
	for(int i = (num_level%2) ; i < half_level; ++i)
	{
		rep[i] = sqrt(info.w(i));
	}

	bdr[0] = 0;
	for(int i = 0 ; i < half_level-1 ; ++i)
	{
		bdr[i+1] = sqrt(temp[slice_vec[i]] - temp[slice_vec[i]-1]);
	}	
		
	// apply forward kernel 
	bdr = bdr_rep_blob.mutable_gpu_data();
	rep = bdr_rep_blob.mutable_gpu_diff();

	Dtype* out = out_blob.mutable_gpu_data();

	eh_weight_quant_scale_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>
		(n, half_level, len_ch, scale, in, out, bdr, rep);	
}

template
void eh_weight_quant_update<float>(
	const int num_level, const int num_ch, const int len_ch,
	Blob<float>& in_blob, Blob<float>& out_blob,
	Blob<float>& scale_blob, Blob<float>& mult_tmp_blob, 
	Blob<float>& bdr_rep_blob, Blob<float>& slice_blob, std::vector<int>& slice_vec);

template
void eh_weight_quant_update<double>(
	const int num_level, const int num_ch, const int len_ch,
	Blob<double>& in_blob, Blob<double>& out_blob,
	Blob<double>& scale_blob, Blob<double>& mult_tmp_blob, 
	Blob<double>& bdr_rep_blob, Blob<double>& slice_blob, std::vector<int>& slice_vec);


}
