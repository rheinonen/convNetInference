#include <cblas.h>

#include "ConvLayer.h"

/**
 * Constructor - Initialize layer params, read in weights, and calculate im2col shape for params.
 */
ConvLayer::ConvLayer (
  const int* input_shape,
  const int* output_shape,
  const char* prev,
  const char* next,
  const int* kernel_shape,
  const int* stride,
  const int* padding,
  const float* weights,
  const float* biases
):
  Layer(input_shape, output_shape, prev, next),
  weights(weights),
  biases(biases),
  stride(stride),
  padding(padding),
  kernel_shape(kernel_shape)
{
  for (int i = 0; i < 2; i++) {
    im2col_output_shape[i] = (input_shape[i] + 2 * padding[i] -
      (dilation[i] * (kernel_shape[i] - 1) + 1)) / stride[i] + 1;
  }

  for (int row = 0; row < this->kernel_shape[2]; row++) {
    for (int col = 0; col < this->im2col_output_shape[1]; col++) {
      this[]
      output[row][col] = biases[row];
    }
  }
};

/**
 * Basic im2col implementation - converts a 3D image tensor to matrix form
 *
 * @param input - the image tensor to convert to matrix form (in 1D array form assuming 3D shape was [row][col][ch])
 * @param output - an array (in 1D array form assuming 2D shape was [kernel_idx][application_of_kernel_idx])
 */
void ConvLayer::im2col(const float* input, float* output) {
  int row_size_in = this->input_shape[1];
  int ch_size_in = this->input_shape[0] * row_size_in;

  int row_size_out = im2col_output_shape[1];

  for (int ch = 0; ch < this->input_shape[2]; ch++) {
    for (int k_row = 0; k_row < this->kernel_shape[0]; k_row++) {
      for (int k_col = 0; k_col < this->kernel_shape[1]; k_col++) {
        int out_row = k_col + k_row * this->kernel_shape[1];

        int in_row = -this->padding[0] + k_row * this->dilation[0];
        for (in_row; in_row < this->input_shape[0] - this->padding[0]; in_row+=this->stride[0]) {
          int in_col = -this->padding[1] + k_col * this->dilation[1];
          for (in_col; in_col < this->input_shape[1] - this->padding[1]; in_col+=this->stride[1]) {
            int out_col = (in_col + this->padding[1]) / this->stride[1] +
              this->input_shape[1] * (in_row + this->padding[0]) / this->stride[0] + ch * ch_size_in;

            if (in_row < 0 || in_row > this->input_shape[0] || in_col < 0 || in_col > this->input_shape[1]) {
              output[out_row * row_size_out + out_col] = 0;
            } else {
              output[out_row * row_size_out + out_col] =
                input[row_size_in * (in_row + k_row) + ch_size_in * ch + in_col + k_col];
            }
          }
        }
      }
    }
  }
}

/**
 * Basic col2im implementation - returns a 3D image tensor to its original form from matrix form
 *
 * @param input - matrix form of the image tensor
 * @param output - an initalized 3D array to hold the output

void ConvLayer::col2im(const float* input, float* output) {
  int row_size = this->output_shape[1];
  int ch_size = this->output_shape[0] * row_size;

  for (int ch = 0; ch < this->output_shape[2]; ch++) {
    for (int row = 0; row < this->output_shape[1]; row++) {
      for (int col = 0; col < this->output_shape[0]; col++) {
        output[col + row * row_size + ch * ch_size] = input[ch][col + row * row_size];
      }
    }
  }
}
 */

/**
 * Construct a matrix based on the filter weights
 *
 * @param fm - an initalized 2D array to hold the output
 *
void ConvLayer::filterMatrix(const float* fm) {
  int row_size = this->kernel_shape[1];
  int ch_size = row_size * kernel_shape[0];
  int filter_size = ch_size * kernel_shape[2];

  for (int filt = 0; filt < this->output_shape[2]; filt++) {
    for (int ch = 0; ch < this->input_shape[2], ch++) {
      for (int w_row = 0; w_row < this->kernel_shape[0]; w_row++) {
        for (int w_col = 0; w_col < this->kernel_shape[1]; w_col++) {
          fm[]
          fm[filt][ch * w_row * w_col] = this->weights[w_row][w_col][ch][filt];
        }
      }
    }
  }
}
 */

/**
 * Serially perform forward propagation for the layer. Convert the image tensor to a matrix,
 * create a matrix out of the convolutional filters, multiply these two matrices, and then
 * optionally convert back to 3D image tensor form (not necessary if this layer is followed
 * by a pooling layer)
 *
 * @param fm - an initalized 2D array to hold the output
 */
void ConvLayer::forwardProp(const float* input, float* output, const bool perform_col2im) {
  float im_matrix[this->im2col_output_shape[0] * this->im2col_output_shape[1]];
  this->im2col(input, im_matrix);

  cblas_gemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    this->kernel_shape[2],
    this->im2col_output_shape[1],
    this->im2col_output_shape[0],
    1.0,
    (*this->weights),
    this->kernel_shape[2],
    im_matrix,
    this->im2col_output_shape[0],
    1.0,
    ouput,
    this->kernel_shape[2]
  );
}

void ConvLayer::forwardPropThreaded(int dim_x, int dim_y, int n_channels, float &input) {
  // @TODO: implement me
}
