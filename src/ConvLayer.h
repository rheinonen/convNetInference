#ifndef __ConvLayer__
#define __ConvLayer__

using namespace std;

namespace Layers {

class ConvLayer {
private:
  int* input_shape;
  int* kernel_shape, stride, padding;
  int* output_shape, im2col_output_shape;

  float* weights;
  float* biases;

  void im2col(const float* input, float* output) {};

  void col2im(const float* input, float* output) {};

  void filterMatrix(const float* fm) {};
public:
  ConvLayer (
    const int* input_shape,
    const int* kernel_shape,
    const int* stride,
    const int* padding,
    const int* output_shape,
    const float* weights,
    const float* biases
  ) {};

  void forwardProp(int dim_x, int dim_y, int n_channels, float &input) {};

  void forwardPropThreaded(int dim_x, int dim_y, int n_channels, float &input) {};

  void forwardPropGPU(int dim_x, int dim_y, int n_channels, float &input) {};
}

}
