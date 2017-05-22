#include "Layer.h"

Layer::Layer(
  const int* input_shape,
  const int* output_shape,
  const char* prev,
  const char* next
):
  input_shape(input_shape),
  output_shape(output_shape),
  prev(prev),
  next(next)
{
  output_size = output_shape[0] * output_shape[1] * output_shape[2];
};

char* Layer::getNext() {
  return next;
};

int Layer::getOutputSize() {
  return output_size;
};
