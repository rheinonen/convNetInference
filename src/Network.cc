Network::Network(network::Network _net) {
  for (auto l : _net.layers) {
    if (l.is_first_layer) first_layer = l.name;

    // @TODO - construct all of the other layer types
    switch (layer.type) {
      case "data":
        layers[l.name] = new DataLayer();
        break;
      case "conv":
        layers[l.name] = new ConvLayer(
          l.params.input_shape,
          l.params.output_shape,
          l.prev,
          l.next,
          l.params.kernel_shape,
          l.params.stride,
          l.params.padding,
          l.params.weights,
          l.params.biases
        );
        break;
      case "lrn":
        layers[l.name] = new LRNLayer();
        break;
      case "relu":
        layers[l.name] = new RELULayer();
        break;
      case "pool":
        layers[l.name] = new PoolingLayer();
        break;
      case "fc":
        layers[l.name] = new FCLayer();
        break;
      case "softmax":
        layers[l.name] = new SoftmaxLayer();
        break;
    }
  }
};

void Network::classify(vector<float> input) {
  vector<float> output;
  Layer* layer = &layers[first_layer];
  bool next_layer_exists = true;

  while (next_layer_exists) {
    output.resize(layer->getOutputSize());

    layer->forwardProp(input, output);

    input = output;

    if (layer->getNext() == "eon") {
      next_layer_exists = false;
    } else {
      layer = &layers[layer->getNext()];
    }
  }

};
