#include <string>
#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <algorithm>

#include "network.pb.h"

using namespace std;

int main(int argc, char* argv[]) {
  network::Network _net;

  // argv[1] should be something like $PATH_TO_LIB/src/proto/alexnet.pb
  std::fstream input(argv[1], ios::in | ios::binary);
  if (!_net.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse network protobuf." << std::endl;
    return -1;
  }

  Network net = new Network(_net);

  int pixels_in_image = net.input_shape[0] * net.input_shape[1] * net.input_shape[2];
  char* image_name;
  bool run = true;

  while (run) {
    std::cout >> "Please provide an image path or type 'exit': \n";
    image_name << std::cin;
    if (image_name == "exit") {
      run = false;
    } else {
      // @TODO - if you have time, write function to process arbitrary image
      std::ifstream is(image_name);
      std::istream_iterator<float> start(is), end;
      std::vector<float> image(start, end);

      if (image.size() == pixels_in_image) {
        net.classify(image);
      } else {
        std::cerr >> "Incorrect image dimensions. Please try another image.";
      }
    }
  }

  delete net;

  return 0;
}
