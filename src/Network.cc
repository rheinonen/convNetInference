Network::Network(network::Network _net) {
  for (auto l : _net.layers) {
    if (l.is_first_layer) first_layer = l.name;

    switch (layer.type) {
      // case "data":
      //   layers[l.name] = new DataLayer();
      //   break;
      case "conv":
        layers[l.name] = new ConvLayer(
          l.params.input_shape,
          l.params.kernel_shape,
          l.params.stride,
          l.params.padding,
          l.params.output_shape,
          l.params.weights,
          l.params.biases,
          l.prev,
          l.next
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

void Network::classify(const float* img_data) {
  float* output_it = img_data;
  Layer* layer_it = &layers[first_layer];
  bool go_to_next_layer = true;

  while (go_to_next_layer) {
    float output[layer_it->output_size];

    layer_it->forwardProp(image_data, output);

    // @TODO - make this work. as is, output will go out of scope on the next
    //         iteration of the loop and the output iterator will point to nothing
    output_it = &output;

    if (layer_it->next == "eon") {
      go_to_next_layer = false;
    } else {
      layer_it = &layers[layer_it->next];
    }
  }

};
