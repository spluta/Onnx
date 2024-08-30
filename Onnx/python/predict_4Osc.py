import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from typing import List

import matplotlib.pyplot as plt

# if using mps, you need to define the device
mps_device = torch.device("mps")

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPRegressor, self).__init__()
        self._methods = []
        self._attributes = ["none"]
        self.fc1 = nn.Linear(input_size, hidden_size//2)  # Modified line
        self.fc2 = nn.Linear(hidden_size//2, hidden_size)
        self.fc2b = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc4b = nn.Linear(hidden_size//4, hidden_size//8)
        self.fc5 = nn.Linear(hidden_size//8, output_size)

        # this is how NN.ar sees the input and output sizes of the forward method
        self.register_buffer(
            f'forward_params',
            torch.tensor([
                input_size,
                1,
                output_size,
                1,
            ]))

        self._methods.append("forward")

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc2b(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc4b(x))
        x = self.fc5(x)
        return x
    
    # I would think that this is the way to expose the methods and attributes of the model
    # these methods are in the nn_tilde module 
    # but I'm not sure NN actually uses this information
    @torch.jit.export
    def get_methods(self):
        return self._methods

    @torch.jit.export
    def get_attributes(self) -> List[str]:
        return self._attributes


# make the target data
# these are four waveforms, a sine, triangle, square, and sawtooth wave

buf_size = 65536

sin = torch.sin(torch.linspace(0, 1-(1./buf_size), buf_size)*2*torch.pi)
tri = torch.cat((torch.linspace(0, 1, buf_size//4), torch.linspace(1, -1, buf_size//2+1)[1:], torch.linspace(-1, 0-(1/buf_size), buf_size//4+1)[1:]))
square = torch.cat((torch.linspace(0, 1, 2), torch.linspace(1, 1, buf_size//2-2), torch.linspace(1, -1, 2), torch.linspace(-1, -1, buf_size//2-2)))
saw = torch.cat((torch.linspace(0, 1, 2), torch.linspace(1, -1, buf_size-1)[1:]))

y_train = torch.cat((sin, tri, square, saw))

num_waves = 4

# make the input data
# in this case the nn will get the phase of a waveform, then a value between 0 and 1 to interpolate between the waveforms, 0 should be the first waveform, 1 the last

X_train = np.linspace(0, 1, buf_size)

X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1)

X_train_a = torch.cat((X_train, torch.zeros_like(X_train)), dim=1)
X_train_b = torch.cat((X_train, torch.zeros_like(X_train)+1/(num_waves-1)), dim=1)
X_train_c = torch.cat((X_train, torch.zeros_like(X_train)+2/(num_waves-1)), dim=1)
X_train_l = torch.cat((X_train, torch.ones_like(X_train)), dim=1)

# concatenate the training data and move the data to the gpu
X_train = torch.cat((X_train_a, X_train_b, X_train_c, X_train_l), dim=0).to(mps_device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(mps_device)

# Create the model and define the loss function and optimizer
model = MLPRegressor(2, 256, 1).to(mps_device)
criterion = nn.MSELoss()


last_time = time.time()

for nums in [[0.001,10000], [0.0001,10000], [0.00001, 30000]]:
    optimizer = optim.Adam(model.parameters(), lr=nums[0])

    # Train the model
    for epoch in range(nums[1]):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        if epoch % 100 == 0:
            elapsed_time = time.time() - last_time
            last_time = time.time()
            print(epoch, loss.item(), elapsed_time)
        loss.backward()
        optimizer.step()

# Generate predictions using the trained model
predicted_sin = model(X_train).detach().to('cpu').numpy()

# Print the training loss
print("Training loss:", loss.item())

# Save the gpu model
torch.save(model, "4Osc_trainings/small_predict4_gpu_model_65536_" + str(int(loss.item() * 1000000)))

# Save the cpu model
cpu_model = model.to('cpu')

torch.save(cpu_model, "4Osc_trainings/small_predict4_cpu_model_65536_" + str(int(loss.item() * 1000000)))

torch_input = torch.randn(1, 2)
onnx_program = torch.onnx.dynamo_export(cpu_model, torch_input)

onnx_program.save("4Osc.onnx")

# # Save the torchscript model
# scripted = torch.jit.trace(cpu_model)
# scripted.save("aa_saw_model2.pt")

# predicted_sin = model(X_train).detach().to('cpu').numpy()
# plt.plot(predicted_sin)
# plt.show()