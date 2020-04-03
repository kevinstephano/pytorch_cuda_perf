#include <torch/extension.h>
#include <vector>

torch::Tensor element_wise_cuda(torch::Tensor const& inputs, torch::Tensor &outputs);
torch::Tensor element_wise_vect_cuda(torch::Tensor const& inputs, torch::Tensor &outputs);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor element_wise(torch::Tensor const& inputs, torch::Tensor &outputs)
{
  AT_ASSERTM(inputs.dim()         == 2, "expected 2D tensor");
  AT_ASSERTM(outputs.dim()        == 2, "expected 2D tensor");

  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(outputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  
  return element_wise_cuda(inputs, outputs);
}

torch::Tensor element_wise_vect(torch::Tensor const& inputs, torch::Tensor &outputs)
{
  AT_ASSERTM(inputs.dim()         == 2, "expected 2D tensor");
  AT_ASSERTM(outputs.dim()        == 2, "expected 2D tensor");

  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(outputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  
  return element_wise_vect_cuda(inputs, outputs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("exec", &element_wise, "Element wise exec.");
        m.def("exec", &element_wise_vect, "Element wise exec.");
}
