import network_pb2
import numpy as np
import os

# Constants
LAYER_FIELD_NAMES = ["name", "type", "prev", "next"]
LAYER_PARAMETER_NAMES = ["input_shape", "output_shape", "weights", "biases", "kernel_shape", "stride",
                         "padding", "dilation", "apply_col2im", "local_size", "alpha", "beta", "class_names"]


alex_net_weights = np.load(os.path.join(os.path.dirname(__file__), "bvlc_alexnet.npy")).item()

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
        "dilation": 1,
        "duplicate_kernel": False
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
    "next": ["relu1"]
}, {
    "name": "relu1",
    "type": "relu",
    "params": {
        "input_shape": [55, 55, 96],
        "output_shape": [55, 55, 96]
    },
    "prev": ["lrn1"],
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
        "kernel_shape": [5, 5, 48],
        "output_shape": [27, 27, 256],
        "stride": [1, 1],
        "padding": [2, 2],
        "dilation": 1,
        "duplicate_kernel": True
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
    "next": ["relu2"]
}, {
    "name": "relu2",
    "type": "relu",
    "params": {
        "input_shape": [27, 27, 256],
        "output_shape": [27, 27, 256]
    },
    "prev": ["lrn2"],
    "next": ["pool2"]
}, {
    "name": "pool2",
    "type": "pool",
    "params": {
        "input_shape": [27, 27, 256],
        "output_shape": [13, 13, 256],
        "kernel_shape": [3, 3, 1],
        "stride": [2, 2],
        "pool_fn": "max"
    },
    "prev": ["lrn2"],
    "next": ["conv3"]
}, {
    "name": "conv3",
    "type": "conv",
    "params": {
        "input_shape": [13, 13, 256],
        "kernel_shape": [3, 3, 384],
        "output_shape": [13, 13, 384],
        "stride": [1, 1],
        "padding": [1, 1],
        "dilation": 1,
        "duplicate_kernel": False
    },
    "prev": ["pool2"],
    "next": ["conv4"]
}, {
    "name": "relu3",
    "type": "relu",
    "params": {
        "input_shape": [13, 13, 384],
        "output_shape": [13, 13, 384]
    },
    "prev": ["conv3"],
    "next": ["conv4"]
}, {
    "name": "conv4",
    "type": "conv",
    "params": {
        "input_shape": [13, 13, 384],
        "kernel_shape": [3, 3, 192],
        "output_shape": [13, 13, 384],
        "stride": [1, 1],
        "padding": [1, 1],
        "dilation": 1,
        "duplicate_kernel": True
    },
    "prev": ["relu3"],
    "next": ["relu4"]
}, {
    "name": "relu4",
    "type": "relu",
    "params": {
        "input_shape": [13, 13, 384],
        "output_shape": [13, 13, 384]
    },
    "prev": ["conv4"],
    "next": ["conv5"]
}, {
    "name": "conv5",
    "type": "conv",
    "params": {
        "input_shape": [13, 13, 384],
        "kernel_shape": [3, 3, 128],
        "output_shape": [13, 13, 256],
        "stride": [1, 1],
        "padding": [1, 1],
        "dilation": 1,
        "duplicate_kernel": True
    },
    "prev": ["relu4"],
    "next": ["relu5"]
}, {
    "name": "relu5",
    "type": "relu",
    "params": {
        "input_shape": [13, 13, 256],
        "output_shape": [13, 13, 256]
    },
    "prev": ["conv5"],
    "next": ["pool5"]
}, {
    "name": "pool5",
    "type": "conv",
    "params": {
        "input_shape": [13, 13, 256],
        "output_shape": [6, 6, 256],
        "kernel_shape": [3, 3, 1],
        "stride": [2, 2],
        "pool_fn": "max"
    },
    "prev": ["conv4"],
    "next": ["fc6"]
}, {
    "name": "fc6",
    "type": "fc",
    "params": {
        "input_shape": [6, 6, 256],
        "output_shape": [1, 1, 4096]
    },
    "prev": ["pool5"],
    "next": ["fc7"]
}, {
    "name": "fc7",
    "type": "fc",
    "params": {
        "input_shape": [1, 1, 4096],
        "output_shape": [1, 1, 4096]
    },
    "prev": ["fc6"],
    "next": ["fc8"]
}, {
    "name": "fc8",
    "type": "fc",
    "params": {
        "input_shape": [1, 1, 4096],
        "output_shape": [1, 1, 1000]
    },
    "prev": ["fc7"],
    "next": ["softmax"]
}, {
    "name": "softmax",
    "type": "softmax",
    "params": {
        "input_shape": [1, 1, 1000],
        "output_shape": [1, 1, 1000]
    },
    "prev": ["fc8"],
    "next": ["eon"]
}]

for layer in architecture:
    if layer["name"] in alex_net_weights:
        weights = alex_net_weights[layer["name"]][0]
        layer["params"]["weights"] = [float(w) for w in np.reshape(weights, weights.size)]
        layer["params"]["biases"] = [float(b) for b in alex_net_weights[layer["name"]][1]]

        # Training was split on multiple GPUs in original AlexNet. We need to apply the same filter twice.
        if "duplicate_kernel" in layer["params"]:
            if layer["params"]["duplicate_kernel"]:
                layer["params"]["weights"] += layer["params"]["weights"]
                layer["params"]["kernel_shape"][2] *= 2
            del layer["params"]["duplicate_kernel"]

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

output_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "src", "proto", "alexnet.pb"))

with open(output_path, "wb") as f:
    f.write(net.SerializeToString())
