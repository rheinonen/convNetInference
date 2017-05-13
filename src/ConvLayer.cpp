#include "ConvLayer.h"

ConvLayer::ConvLayer (int &shape, int stride, float &w, float &b) :
  // weights and biases initialization are copy ops so lightweight
  weights(w), biases(b), stride(stride)
{
  // shape will contain x and y dimensions of the convolutional filters in the
  // zeroth and first positions respectively and the number of filters in the
  // second position
  fs_x = shape[0];
  fs_y = shape[1];
  depth = shape[2];
}

void ConvLayer::forwardProp(int dim_x, int dim_y, int n_channels, float &input) {
  // implement me
}

void ConvLayer::forwardPropThreaded(int dim_x, int dim_y, int n_channels, float &input) {
  // implement me
}

void ConvLayer::forwardPropGPU(int dim_x, int dim_y, int n_channels, float &input) {
  // implement me
}
