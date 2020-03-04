#include <torch/extension.h>
#include <vector>

torch::Tensor reduction_cuda(torch::Tensor const& inputs, torch::Tensor &outputs);
torch::Tensor reduction_shared_cuda(torch::Tensor const& inputs, torch::Tensor &outputs);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor reduction(torch::Tensor const& inputs, torch::Tensor &outputs)
{
  AT_ASSERTM(inputs.dim()         == 2, "expected 2D tensor");
  AT_ASSERTM(outputs.dim()        == 1, "expected 1D tensor");

  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(outputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  
  return reduction_cuda(inputs, outputs);
}

torch::Tensor reduction_shared(torch::Tensor const& inputs, torch::Tensor &outputs)
{
  AT_ASSERTM(inputs.dim()         == 2, "expected 2D tensor");
  AT_ASSERTM(outputs.dim()        == 1, "expected 1D tensor");

  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(outputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  
  return reduction_shared_cuda(inputs, outputs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("exec", &reduction, "Reduction exec.");
        m.def("exec_shared", &reduction_shared, "Reduction exec.");
}
