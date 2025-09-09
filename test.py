import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size):
        super(ESN, self).__init__()
        self.reservoir_size = reservoir_size
        self.W_in = nn.Linear(input_size, reservoir_size)
        self.W_res = nn.Linear(reservoir_size, reservoir_size)
        self.W_out = nn.Linear(reservoir_size, output_size)

    def forward(self, input, return_states=False):
        reservoir = torch.zeros((input.size(0), self.reservoir_size))
        states = []  # store states over time

        for i in range(input.size(1)):
            input_t = input[:, i, :]
            reservoir = torch.tanh(self.W_in(input_t) + self.W_res(reservoir))
            states.append(reservoir.detach().clone())  # save copy

        output = self.W_out(reservoir)
        
        if return_states:
            return output, torch.stack(states, dim=1)  # (batch, seq_len, reservoir_size)
        return output

# Example input
batch_size, seq_len, input_size = 1, 10, 2
reservoir_size, output_size = 5, 1

model = ESN(input_size, reservoir_size, output_size)

# Create a toy sequence of shape (1, seq_len, 2)
x = torch.randn(batch_size, seq_len, input_size)

# Get output + reservoir states
y, states = model(x, return_states=True)  # states shape: (1, seq_len, reservoir_size)

# Take the single sequence (batch=1)
states = states[0].numpy()  # (seq_len, reservoir_size)

# Plot reservoir activations over time
plt.figure(figsize=(8, 5))
for i in range(reservoir_size):
    plt.plot(range(seq_len), states[:, i], label=f"Neuron {i+1}")
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.title("Reservoir States Over Time")
plt.legend()
plt.show()
