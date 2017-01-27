#include <algorithm>
#include <vector>

#include "caffe/layers/weight_log_quant_relu_layer.hpp"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cfloat>

#define BASE_SIZE 16
#define BIN_SIZE 1024

namespace caffe {
	

template <typename Dtype>
__global__ void eh_histogram(const int n, const int offset, const Dtype* data, int* hist)
{
	__shared__ int s[BIN_SIZE];
	
	for(int i = threadIdx.x; i < BIN_SIZE; i+= blockDim.x)
		s[i] = 0;
	
	__syncthreads();

	CUDA_KERNEL_LOOP(i, n) 	
	{
		if(data[i] > 0)
		{
			int quant = (int)floor(log2(data[i])*BASE_SIZE) - offset;
			
			if(quant < 0)
				atomicAdd(&s[0], 1);
			else if (quant >= BIN_SIZE)
				atomicAdd(&s[BIN_SIZE-1], 1);
			else
				atomicAdd(&s[quant], 1);
		}
	}
	
	__syncthreads();
	
	for(int i = threadIdx.x; i < BIN_SIZE; i+= blockDim.x)
		atomicAdd(&hist[i], s[i]);	
}

// step: hist는 2^(1/16) 기준으로 구해짐. 그러므로 2^(step/16) 이 LogQuant의 base 역할을 수행. 
// step은 2의 배수여야 함. 
template <typename Dtype>
struct LQInfo
{
	int num_level_;
	int num_weight_;
	int offset_;
	int* hist_scan_;
	
	// idx == 0 은 zero level에 사용됨
	Dtype p(int idx, int fsr, int step)
	{
		if(idx == num_level_-1)
		{
			int begin = fsr + (step>>1)*(2*idx-3);
			return static_cast<Dtype>(num_weight_-hist_scan_[begin-1])/num_weight_;
		}
		else
		{
			int begin = fsr + (step>>1)*(2*idx-3);
			int end = fsr + (step>>1)*(2*idx-1);
			return static_cast<Dtype>(hist_scan_[end-1] - hist_scan_[begin-1])/num_weight_;
		}
	}

	Dtype w(int idx, int fsr, int step)
	{
		return pow(2, static_cast<Dtype>(fsr+(idx-1)*step+offset_)/BASE_SIZE);		
	}

	Dtype wEnt(int fsr, int step)
	{
		Dtype ent = 0;

		for(int i = 1 ; i < num_level_; ++i)
		{
			Dtype prob = p(i, fsr, step);
			if(prob>0)
				ent -= w(i, fsr, step)*prob*log(prob);
		}
		return ent;
	}
};

template <typename Dtype>
__global__ void WLQReLUForward2(const int n, const int num_level,
	const Dtype* in, Dtype* out, const Dtype offset, const Dtype step, int phase)
{	
	CUDA_KERNEL_LOOP(index, n)
	{		
		Dtype oTemp;
		Dtype in_data = in[index];
		
		if(in_data <= 0)
		{
			oTemp = 0;
		}
		else
		{
			Dtype mod_idx = min( (Dtype)(num_level-2), 
				round((log2(in_data)*BASE_SIZE - offset)/step) );

			if(mod_idx < 0)
			{
				oTemp = (phase == 0) ? FLT_MIN : 0;
			}
			else
			{
				oTemp = powf(2., (offset + mod_idx*step)/BASE_SIZE);			
			}
		}
		out[index] = oTemp;
	}
}

template <typename Dtype>
void WeightLogQuantReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
	Dtype* option = this->blobs_[0]->mutable_cpu_data();
	int num_level = static_cast<int>(option[0]);
	int updated = static_cast<int>(option[1]);
	
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const int n = bottom[0]->count();

	//////// update offset	
	if(this->phase_ == 0 || updated != 0)
	{
		option[1] = 0;
		int offset =  - 640;
		
		// get histogram
		int* hist = this->hist_.mutable_gpu_data();
		caffe_gpu_set<int>(BIN_SIZE, 0, hist);
		eh_histogram<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
			n, offset, bottom_data, hist);
		thrust::device_ptr<int> hist_d = thrust::device_pointer_cast(hist);
		thrust::inclusive_scan(hist_d, hist_d+BIN_SIZE, hist_d);
		hist = this->hist_.mutable_cpu_data();

		// reduce search dimension
		int min_offset = -1;
		int max_offset = -1;
		int max_value = -1;

		for(int i = 0 ; i < BIN_SIZE; ++i)
		{
			if(hist[i] ==0)
				min_offset = i;

			if(hist[i] > max_value)
			{
				max_offset = i;
				max_value = hist[i];
			}
		}

		int length = max_offset -min_offset + 1;

		hist += min_offset;
		offset += min_offset;	

		// find maximum configurations
		Dtype max_ent = 0;
		int max_half_step = 0;
		int max_fsr = 0;

		LQInfo<Dtype> info = {num_level, hist[length-1], offset, hist};

		for(int half_step = 1; half_step < 17; ++half_step)
		{
			for(int fsr = 40; fsr < length - half_step*(2*num_level - 5); ++fsr)
			{
				Dtype ent = info.wEnt(fsr, 2*half_step);

				if(ent > max_ent)
				{
					max_ent = ent;
					max_half_step = half_step ;
					max_fsr = fsr;
				}
			}			
		}
		
		Dtype* bdr = this->blobs_[1]->mutable_cpu_data();
		Dtype* rep = bdr + num_level;
		bdr[0] = rep[0] = 0;

		for(int i = 1 ; i < num_level; ++i)
		{
			//rep[i] = info.w(i, max_fsr, max_half_step*2);
			rep[i] = pow(2, static_cast<Dtype>(max_fsr + offset + 2*(i-1)*max_half_step)/BASE_SIZE);
			bdr[i] = pow(2, static_cast<Dtype>(max_fsr + offset + (2*i-1)*max_half_step)/BASE_SIZE);
		}

		//////////////
		rep[0] = max_fsr + offset;
		bdr[0] = 2*max_half_step;
	}
	
		
	// WLQ ReLU forward
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* bdr = this->blobs_[1]->mutable_cpu_data();
	Dtype* rep = bdr + num_level;

	WLQReLUForward2<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>
		(n, num_level, bottom_data, top_data, rep[0], bdr[0], this->phase_);
    
	CUDA_POST_KERNEL_CHECK;
	// << " count: " << count << " bottom_data: "
	//     << (unsigned long)bottom_data
	//     << " top_data: " << (unsigned long)top_data
	//     << " blocks: " << CAFFE_GET_BLOCKS(count)
	//     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Dtype>
void WeightLogQuantReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) { 

	Dtype* option = this->blobs_[0]->mutable_cpu_data();
	option[2] = static_cast<Dtype>(1);
	
	if (propagate_down[0]) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
		// NOLINT_NEXT_LINE(whitespace/operators)
		ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, top_diff, bottom_data, bottom_diff, negative_slope);
		CUDA_POST_KERNEL_CHECK;
	}
}


INSTANTIATE_LAYER_GPU_FUNCS(WeightLogQuantReLULayer);


}  // namespace caffe
