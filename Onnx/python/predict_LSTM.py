import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from typing import List


# if using mps, you need to define the device
mps_device = torch.device("mps")

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, bias_fl=True,
                 num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rec = nn.LSTM(1, 32, 1)
        self.lin = nn.Linear(32, 1)
        self.hidden = None

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        x, (hn, cn) = self.rec(x, (h0.detach(), c0.detach()))

        return self.lin(x)


model = SimpleRNN()

for name, param in model.named_parameters():
    print(name)
    print(param.data)
    param.data = torch.randn_like(param.data)

model.rec.weight_ih_l0

weights = model.rec.weight_ih_l0

# model.rec.weight_ih_l0 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).reshape(8, 1).requires_grad_(True)
# model.rec.weight_ih_l0 = torch.rand(128, 1).requires_grad_(True)
# print(weights)

# model


model.forward(torch.randn(1, 1, 1))

scripted_model = torch.jit.trace(model, torch.randn(1, 1, 1))
scripted_model.save("model.pt")


# torch.save(cpu_model, "4Osc_trainings/small_predict4_cpu_model_65536_" + str(int(loss.item() * 1000000)))

torch_input = torch.randn(1, 1, 1)
onnx_program = torch.onnx.dynamo_export(model, torch_input)

onnx_program.save("4Osc.onnx")


from onnx import numpy_helper
import onnxruntime as ort

weights = onnx_model.graph.initializer
numpy_helper.to_array(weights[4])

ort_session = ort.InferenceSession("/Users/spluta1/Library/Application Support/SuperCollider/Extensions/MyPlugins/Onnx/Onnx/python/model_from_ortbuilder.onnx", None)

ort_session.get_inputs()[0].shape
ort_session.get_inputs()[1].shape
ort_session.get_inputs()[2].shape

ort_session.get_outputs()[0].shape
# ort_session.get_modelmeta

x = np.array([0.5, 0.2], dtype=np.float32)
hin = np.random.rand(3,24).astype(np.float32)
cin = np.random.rand(3,24).astype(np.float32)

ort_session.run(None, {"x": x, "hin": hin, "cin": cin})

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
    
lstm = LSTMModel(1, 100, 1, 10)

lstm.forward(torch.randn(100, 28, 1))

# # Save the torchscript model
scripted = torch.jit.trace(lstm, torch.randn(100, 28, 1))
