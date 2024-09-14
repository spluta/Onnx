import torch
import onnx
from torch import nn
import numpy as np
import onnxruntime.backend as backend
import numpy as np

torch.manual_seed(0)

layer_count = 4

model = nn.Sequential(
    nn.LSTM(1, 32, num_layers=1),
    nn.Linear(32, 1)
)
model = nn.LSTM(1, 32, 1)

input = torch.randn(1, 1, 1)


h0 = torch.randn(1, 1, 32)
c0 = torch.randn(1, 1, 32)
model
model.forward(from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

# The serialization
with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# display
print(onnx_model)input, [h0, c0])

a[0].shape

model.eval()

with torch.no_grad():
    input = torch.randn(1, 1, 1)
    h0 = torch.randn(layer_count * 2, 3, 20)
    c0 = torch.randn(layer_count * 2, 3, 20)
    output, (hn, cn) = model(input, (h0, c0))

    # default export
    torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx')
    onnx_model = onnx.load('lstm.onnx')
    # input shape [5, 3, 10]
    print(onnx_model.graph.input[0])

    # export with `dynamic_axes`
    torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx',
                    input_names=['input', 'h0', 'c0'],
                    output_names=['output', 'hn', 'cn'],
                    dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}})
    onnx_model = onnx.load('lstm.onnx')
