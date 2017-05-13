#ifndef __ConvLayer__
#define __ConvLayer__

using namespace std;

namespace Layers {

class ConvLayer {
private:
  unsigned int fs_x;
  unsigned int fs_y;
  unsigned int stride;
  unsigned int depth;
  float* weights;
  float* biases;
public:
  ConvLayer (int &shape, int stride, float &w, float &b) {};

  void forwardProp(int dim_x, int dim_y, int n_channels, float &input) {};

  void forwardPropThreaded(int dim_x, int dim_y, int n_channels, float &input) {};

  void forwardPropGPU(int dim_x, int dim_y, int n_channels, float &input) {};
}

}
