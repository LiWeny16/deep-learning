import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

batch_size = 5

# nueural network model
class DuelingDQN(nn.Module):
    """ Dueling Deep Q-Network """
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        # Feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        # forward function
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


# create model and check the outpur 
model = DuelingDQN(state_dim=10, action_dim=4)
x = torch.randn(1, 10)  # set state_dim = 10
y = model(x)
print("model output :", y)
# make_dot(y, params=dict(model.named_parameters())).render("./images/dueling_dqn", format="png")


# Network parameters
state_dim = 10  # Example: dimensionality of the state space
action_dim = 4  # Example: number of actions in the action space
num_epochs = 500  # Number of training epochs

# Instantiate the network, optimizer, and loss function
dueling_dqn = DuelingDQN(state_dim, action_dim)
optimizer = optim.Adam(dueling_dqn.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# Loss storage for plotting
losses = []
mean_weights = []

def smooth(y, box_pts):
    # Create a box filter
    box = np.ones(box_pts)/box_pts
     # Apply convolution of 'y' with the box filter, keeping output size same as input
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_mean_weights(net):
    total_weight_sum = 0
    total_weight_count = 0
     # Iterate through each parameter in the network
    for param in net.parameters():
        total_weight_sum += param.data.mean().item()
        total_weight_count += 1
    # Calculate the average mean weight across all parameters
    return total_weight_sum / total_weight_count

# Main training loop
for epoch in range(num_epochs):
    # Generate dummy data for testing
    dummy_states = torch.randn(batch_size, state_dim)
    dummy_next_states = torch.randn(batch_size, state_dim)
    dummy_rewards = torch.randn(batch_size, 1)
    dummy_dones = torch.zeros(batch_size, 1)
    dummy_actions = torch.randint(0, action_dim, (batch_size, 1))

    # Forward pass
    current_q_values = dueling_dqn(dummy_states).gather(1, dummy_actions)
    next_q_values = dueling_dqn(dummy_next_states).max(1)[0].detach().unsqueeze(1)
    expected_q_values = dummy_rewards + (0.99 * next_q_values * (1 - dummy_dones))

    # Loss computation
    loss = loss_function(current_q_values, expected_q_values)
    losses.append(loss.item())
    mean_weights.append(get_mean_weights(dueling_dqn))

    # Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# Plot the loss over epochs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
smoothed_losses = smooth(losses, 10)
plt.plot(smoothed_losses, label='Smoothed Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs with Smoothing')

# Plot the average weight value change
plt.subplot(1, 2, 2)
plt.plot(mean_weights, label='Average Weight Value per Epoch', color='red')
plt.xlabel('Epoch')
plt.ylabel('Average Weight Value')
plt.title('Average Weight Value Change Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
