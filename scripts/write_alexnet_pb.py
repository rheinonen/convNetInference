import network_pb2
import numpy as np

# Constants
LAYER_FIELD_NAMES = ["name", "type", "prev", "next"]
LAYER_PARAMETER_NAMES = ["input_shape", "output_shape", "weights", "biases", "kernel_shape", "stride",
                         "padding", "dilation", "apply_col2im", "local_size", "alpha", "beta", "class_names"]

alex_net_weights = np.load("../assets/bvlc_alexnet.npy").item()

# TODO - finish adding layers
architecture = [{
    "name": "data",
    "type": "data",
    "params": {
        "input_shape": [227, 227, 3],
        "output_shape": [227, 227, 3]
    },
    "prev": ["data"],
    "next": ["conv1"]
}, {
    "name": "conv1",
    "type": "conv",
    "params": {
        "input_shape": [227, 227, 3],
        "kernel_shape": [11, 11, 3],
        "output_shape": [55, 55, 96],
        "stride": [4, 4],
        "padding": [0, 0],
        "dilation": 1
    },
    "prev": ["data"],
    "next": ["lrn1"]
}, {
    "name": "lrn1",
    "type": "lrn",
    "params": {
        "input_shape": [55, 55, 96],
        "output_shape": [55, 55, 96],
        "local_size": 5,
        "alpha": 0.0001,
        "beta": 0.75
    },
    "prev": ["conv1"],
    "next": ["pool1"]
}, {
    "name": "pool1",
    "type": "pool",
    "params": {
        "input_shape": [55, 55, 96],
        "output_shape": [27, 27, 96],
        "kernel_shape": [3, 3, 1],
        "stride": [2, 2],
        "pool_fn": "max"
    },
    "prev": ["lrn1"],
    "next": ["conv2"]
}, {
    "name": "conv2",
    "type": "conv",
    "params": {
        "input_shape": [27, 27, 96],
        "kernel_shape": [5, 5, 96],
        "output_shape": [27, 27, 256],
        "stride": [1, 1],
        "padding": [2, 2],
        "dilation": 1
    },
    "prev": ["pool1"],
    "next": ["lrn2"]
}, {
    "name": "lrn2",
    "type": "lrn",
    "params": {
        "input_shape": [27, 27, 256],
        "output_shape": [27, 27, 256],
        "local_size": 5,
        "alpha": 0.0001,
        "beta": 0.75
    },
    "prev": ["conv2"],
    "next": ["pool2"]
}, {
    "name": "pool2",
    "type": "pool",
    "params": {
        "input_shape": [27, 27, 256],
        "output_shape": [13, 13, 256],
        "kernel_shape": [2, 2, 1],
        "stride": [2, 2],
        "pool_fn": "max"
    },
    "prev": ["lrn1"],
    "next": ["conv2"]
}]

for layer in architecture:
    if layer["name"] in alex_net_weights:
        weights = alex_net_weights[layer["name"]][0]
        layer["params"]["weights"] = [float(w) for w in np.reshape(weights, weights.size)]
        layer["params"]["biases"] = [float(b) for b in alex_net_weights[layer["name"]][1]]

net = network_pb2.Network()
net.name = 'AlexNet'
net.num_layers = len(architecture)

for l in architecture:
    layer = net.layers.add()

    # Set top level info
    for field_name in LAYER_FIELD_NAMES:
        if isinstance(l[field_name], list):
            getattr(layer, field_name).extend(l[field_name])
        else:
            setattr(layer, field_name, l[field_name])

    # Set layer params
    for param_name, p in l["params"].iteritems():
        if isinstance(p, list):
            getattr(layer.params, param_name).extend(p)
        else:
            setattr(layer.params, param_name, p)

with open("../src/proto/alexnet.pb", "wb") as f:
    f.write(net.SerializeToString())
