#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <torch/extension.h>
#include <math.h>
#include "THC/THC.h"
#include <ATen/cuda/CUDAContext.h>

const int UNROLL = 4;

__device__ half my_float2half(const float f) {
    half val;
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
    return val;
}

template <
          typename scalar_t,
          typename accscalar_t,
          typename IndexType
         >
__global__ void my_element_wise_kernel(scalar_t const                *inputs,
                                       scalar_t                      *outputs,
                                       IndexType                      totalElements
                                      ) 
{
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType rounded_size = ((totalElements - 1)/(blockDim.x * gridDim.x * UNROLL)+1) * blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx;
       linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x*UNROLL) {
       scalar_t src[UNROLL];
       #pragma unroll
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
               src[ii]     = inputs[li];
           }
       }
       #pragma unroll
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
	           outputs[li] = src[ii] + __float2half(2.0);
           }
       }
  }
}

template <
          typename scalar_t,
          typename accscalar_t,
          typename IndexType
         >
void my_element_wise_cuda(scalar_t const *inputs,
                          scalar_t       *outputs,
                          IndexType       totalElements
		                 )
{
  int block_size = 256;
  dim3 dim_block(block_size);
  dim3 grid((totalElements + block_size -1)/block_size);
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);

  my_element_wise_kernel<scalar_t, accscalar_t, IndexType><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(inputs, outputs, totalElements);
  THCudaCheck(cudaGetLastError());
}

torch::Tensor element_wise_cuda(torch::Tensor const& inputs, torch::Tensor &outputs) {

	const int total_elems = inputs.size(0) * inputs.size(1);
	//torch::Tensor outputs = torch::empty_like(inputs, inputs.options());

    my_element_wise_cuda<half,float,uint32_t>(static_cast<half const*>(inputs.data_ptr()), 
                                              static_cast<half*>(outputs.data_ptr()), 
                                              total_elems);

    return outputs;
}

template <
          typename scalar_t,
          typename accscalar_t,
          typename IndexType
         >
__global__ void my_element_wise_vect_kernel(scalar_t const                *inputs,
                                            scalar_t                      *outputs,
                                            IndexType                      totalElements
                                           ) 
{
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType rounded_size = ((totalElements - 1)/(blockDim.x * gridDim.x * UNROLL)+1) * blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx;
       linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x*UNROLL) {
       scalar_t src[UNROLL];
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
               src[ii]     = inputs[li];
           }
       }
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
	           outputs[li] = src[ii] + __float2half(2.0);
           }
       }
  }
}

template <
          typename scalar_t,
          typename accscalar_t,
          typename IndexType
         >
void my_element_wise_vect_cuda(scalar_t const *inputs,
                               scalar_t       *outputs,
                               IndexType       totalElements
		                      )
{
  int block_size = 256;
  dim3 dim_block(block_size);
  dim3 grid((totalElements + block_size -1)/block_size);
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);

  my_element_wise_vect_kernel<scalar_t, accscalar_t, IndexType><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(inputs, outputs, totalElements);
  THCudaCheck(cudaGetLastError());
}

torch::Tensor element_wise_vect_cuda(torch::Tensor const& inputs, torch::Tensor &outputs) {

	const int total_elems = inputs.size(0) * inputs.size(1);
	//torch::Tensor outputs = torch::empty_like(inputs, inputs.options());

    my_element_wise_vect_cuda<half,float,uint32_t>(static_cast<half const*>(inputs.data_ptr()), 
                                                   static_cast<half*>(outputs.data_ptr()), 
                                                   total_elems);

    return outputs;
}
