import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from typing import List

# if using mps, you need to define the device
mps_device = torch.device("mps")

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPRegressor, self).__init__()
        self._methods = []
        self._attributes = ["none"]
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])  # Modified line
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], output_size)
       
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

X_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0,1.0], [1.0,0.0]], dtype=torch.float32).to(mps_device)

y_train = torch.cat([torch.rand(1, 10, dtype=torch.float32), torch.rand(1, 10, dtype=torch.float32), torch.rand(1, 10, dtype=torch.float32), torch.rand(1, 10, dtype=torch.float32)]).to(mps_device)

model = MLPRegressor(2, [3, 5, 7, 9], 10).to(mps_device)
criterion = nn.MSELoss()
last_time = time.time()

for nums in [[0.001,10000]]:
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

# Save the cpu model
cpu_model = model.to('cpu')

# Save the gpu model
torch.save(model, "mlp_" + str(int(loss.item() * 1000000)))

torch_input = torch.randn(1, 2)
onnx_program = torch.onnx.dynamo_export(cpu_model, torch_input)

onnx_program.save("mlp.onnx")