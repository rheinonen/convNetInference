#ifndef __Network__
#define __Network__

#include <string>
#include <vector>
#include <unordered_map>

#include "network.pb.h"

using namespace std;
using namespace Layers;

class Network {
private:
  unordered_map<string, Layers::Layer> layers;
  char* first_layer;

public:
  Network(const network::Network &_net) {};

  void classify(const vector<int> &img_data) {};
}
