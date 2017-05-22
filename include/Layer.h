class Layer {
private:
  int* input_shape, output_shape;
  int output_size;

  char* prev, next;
public:
  Layer(const int* input_shape, const int* output_shape, const char* prev, const char* next) {};

  char* getNext() {};

  int getOutputSize() {};

  // Pure virtual function. This method should be implemented in every layer type.
  virtual void forwardProp(const float* input, float* output) =0;
};
