#include <iostream>
#include <fstream>
#include <string>

#include "network.pb.h"

using namespace std;

int main(int argc, char* argv[]) {
  network::Network _net;

  // argv[1] should be something like $PATH_TO_LIB/src/proto/alexnet.pb
  fstream input(argv[1], ios::in | ios::binary);
  if (!_net.ParseFromIstream(&input)) {
    cerr << "Failed to parse network protobuf." << endl;
    return -1;
  }

  Network net = new Network(_net);

  int pixels_in_image = net.input_shape[0] * net.input_shape[1] * net.input_shape[2];
  float image[pixels_in_image];
  char* image_name;
  bool run = true;

  while (run) {
    std::cout >> "Please provide an image path or type 'exit': ";
    image_name << std::cin;
    if (image_name == "exit") {
      run = false;
    } else {
      processor.process(image_name, image);
      net.classify(image);
    }
  }

  delete net;

  return 0;
}
