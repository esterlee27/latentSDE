import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import copy
import matplotlib.pyplot as plt
import tqdm
import os

# Constants
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # Temperature parameter
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
HIDDEN_SIZE = 256
REPLAY_BUFFER_SIZE = 1000000
_STEPS = 100
_DT = 0.1
INI_STATE = [-0.067, -0.55, -0.35, 0.53]

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.next_state = np.zeros((capacity, state_dim))
        self.reward = np.zeros((capacity, 1))
        self.done = np.zeros((capacity, 1))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )
        
        # Mean and log_std outputs
        self.mean = nn.Linear(HIDDEN_SIZE, action_dim)
        self.log_std = nn.Linear(HIDDEN_SIZE, action_dim)
        
        self.max_action = max_action
        self.action_dim = action_dim
        
        # Initialize weights
        self.apply(self._weights_init)
        
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = self.net(state)
        
        # Get mean and constrain it
        mean = self.mean(x)
        
        # Get log_std and constrain it
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        
        # Squash to [-max_action, max_action]
        y_t = torch.tanh(x_t)
        action = self.max_action * y_t
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        
        # Apply correction for tanh transformation
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
        
        # Initialize weights
        self.apply(self._weights_init)
        
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        
        return q1, q2


class CartpoleDynamics:
    def __init__(self):
        pass
        
    def _f(self, x, u):
        """
        Return the control-independent part of the control-affine dynamics.
        """
        # Set up a tensor for holding the result
        f = np.zeros(4)

        pos = x[0]
        theta = x[1]
        v = x[2]
        theta_dot = x[3]

        # 确保u是标量
        if isinstance(u, np.ndarray):
            u = u.item()  # 转换为标量值

        f[0] = v
        f[1] = theta_dot
        f[2] = (u + np.sin(theta)*(theta_dot**2 - np.cos(theta))) / (1 + np.sin(theta)**2)
        f[3] = (
                u*np.cos(theta) + 
                theta_dot**2 * np.cos(theta) * np.sin(theta) -
                2*np.sin(theta)) / (1 + np.sin(theta)**2)

        return f

    def next_state(self, x, u, ctrl_step):
        """Simulate the system dynamics using a small step size"""
        simulate_step = 1e-2
        steps = int(ctrl_step/simulate_step)
        
        # 确保u是标量
        if isinstance(u, np.ndarray):
            u = u.item()  # 转换为标量值
            
        for i in range(steps):
            x = x + simulate_step * (self._f(x, u))
        return x
    
    def reward_function(self, state, action):
        """Calculate reward based on state and action"""
        # 确保action是标量
        if isinstance(action, np.ndarray):
            action = action.item()
            
        # Penalize distance from origin
        position_penalty = -0.1 * abs(state[0])
        
        # Penalize angle deviation from upright
        angle_penalty = -1.0 * (state[1]**2)
        
        # Penalize velocity
        velocity_penalty = -0.1 * (state[2]**2)
        
        # Penalize angular velocity
        angular_velocity_penalty = -0.1 * (state[3]**2)
        
        # Penalize control effort
        control_penalty = -0.01 * (action**2)
        
        reward = position_penalty + angle_penalty + velocity_penalty + angular_velocity_penalty + control_penalty
        return reward


class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize actor
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        # Initialize critics
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # Initialize target critic
        self.critic_target = copy.deepcopy(self.critic)
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_target.parameters():
            p.requires_grad = False
            
        # Initialize automatic entropy tuning
        self.target_entropy = -action_dim  # -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ACTOR)
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate:
            # Use mean action for evaluation
            mean, _ = self.actor(state)
            return torch.tanh(mean).cpu().data.numpy().flatten()
        else:
            # Sample from distribution for training
            action, _ = self.actor.sample(state)
            return action.cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=BATCH_SIZE):
        # Sample replay buffer 
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        
        # ---------- Update critic ----------
        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute target Q value
        with torch.no_grad():
            # Sample action from the target policy
            next_action, next_log_prob = self.actor.sample(next_state)
            
            # Get target Q estimates
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            
            # Take minimum of the two Q-values to prevent overestimation
            target_q = torch.min(target_q1, target_q2)
            
            # Add entropy term
            target_q = target_q - self.alpha * next_log_prob
            
            # Compute target with Bellman equation
            target_q = reward + (1 - done) * GAMMA * target_q
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------- Update actor ----------
        # Compute actor loss
        actions_pi, log_prob_pi = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.alpha * log_prob_pi - q_pi).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ---------- Update alpha ----------
        alpha_loss = -(self.log_alpha * (log_prob_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # ---------- Update target networks ----------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        }


def evaluate_policy(policy, eval_episodes=10):
    """Run multiple evaluation episodes and return average reward"""
    env = CartpoleDynamics()
    avg_reward = 0.
    trajectories = []
    
    for _ in range(eval_episodes):
        total_reward = 0
        trajectory = []
        
        state = copy.copy(INI_STATE)
        state[0] += np.random.uniform(-0.1, 0.1)
        state[1] += np.random.uniform(-0.05, 0.05)
        
        for t in range(_STEPS):
            trajectory.append(copy.copy(state))
            action = policy.select_action(state, evaluate=True)
            next_state = env.next_state(state, action, _DT)
            reward = env.reward_function(state, action)
            
            state = next_state
            total_reward += reward
        
        trajectories.append(trajectory)
        avg_reward += total_reward
        
    avg_reward /= eval_episodes
    
    return float(avg_reward), trajectories  # 确保返回标量值


def main(train_dir='./train/cartpole_sac/'):
    # Create directory
    os.makedirs(train_dir, exist_ok=True)
    
    # Env and agent setup
    env = CartpoleDynamics()
    state_dim = 4  # [position, angle, velocity, angular_velocity]
    action_dim = 1
    max_action = 5.0
    
    # Initialize SAC agent
    policy = SAC(state_dim, action_dim, max_action)
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, state_dim, action_dim)
    
    # Training loop
    total_timesteps = 100000
    batch_size = 256
    eval_freq = 5000
    update_freq = 50
    updates_per_step = 1
    
    state = copy.copy(INI_STATE)
    episode_timesteps = 0
    episode_num = 0
    episode_reward = 0
    done = False
    
    # Initialize logging
    metrics = {'train_rewards': [], 'eval_rewards': [], 'critic_losses': [], 'actor_losses': [], 'alphas': []}
    
    print("Starting training...")
    for t in tqdm.tqdm(range(1, total_timesteps + 1)):
        episode_timesteps += 1
        
        # Select action with noise
        action = policy.select_action(state)
        
        # Take action in environment
        next_state = env.next_state(state, action[0], _DT)
        
        # Add noise to simulate realistic conditions
        r_noise = 0.03 * np.random.normal(0, np.sqrt(_DT))
        v_noise = 0.03 * np.random.normal(0, np.sqrt(_DT))
        diffusion = np.array([r_noise, v_noise, 0, 0])
        next_state = next_state + diffusion
        
        # Compute reward
        reward = env.reward_function(state, action[0])
        
        # Check if episode is done (e.g., pole angle too large or cart position out of bounds)
        done_bool = 0
        if abs(next_state[0]) > 2.0 or abs(next_state[1]) > np.pi/2:
            done_bool = 1
            done = True
        
        # Store transition in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        
        state = next_state
        episode_reward += reward
        
        # Train agent after collecting enough data
        if t > batch_size:
            if t % update_freq == 0:
                for _ in range(update_freq * updates_per_step):
                    train_metrics = policy.train(replay_buffer, batch_size)
                    
                    metrics['critic_losses'].append(train_metrics['critic_loss'])
                    metrics['actor_losses'].append(train_metrics['actor_loss'])
                    metrics['alphas'].append(train_metrics['alpha'])
        
        if done or episode_timesteps >= _STEPS:
            metrics['train_rewards'].append(episode_reward)
            
            print(f"Episode: {episode_num+1}, Total Timesteps: {t}, Episode Steps: {episode_timesteps}, Reward: {float(episode_reward):.2f}")
            
            # Reset environment
            state = copy.copy(INI_STATE)
            state[0] += np.random.uniform(-0.1, 0.1)
            state[1] += np.random.uniform(-0.05, 0.05)
            
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            done = False
        
        # Evaluate policy
        if t % eval_freq == 0:
            eval_reward, trajectories = evaluate_policy(policy)
            metrics['eval_rewards'].append(eval_reward)
            
            print(f"Evaluation at timestep {t}: Average Reward {eval_reward:.2f}")
            
            # Plot a sample trajectory
            traj = np.array(trajectories[0])
            plt.figure(figsize=(10, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(traj[:, 0])
            plt.title('Position')
            
            plt.subplot(2, 2, 2)
            plt.plot(traj[:, 1])
            plt.title('Angle')
            
            plt.subplot(2, 2, 3)
            plt.plot(traj[:, 2])
            plt.title('Velocity')
            
            plt.subplot(2, 2, 4)
            plt.plot(traj[:, 3])
            plt.title('Angular Velocity')
            
            plt.tight_layout()
            plt.savefig(f'{train_dir}/trajectory_{t}.png')
            plt.close()
            
            # Save model
            torch.save(policy.actor.state_dict(), f'{train_dir}/actor_{t}.pth')
            torch.save(policy.critic.state_dict(), f'{train_dir}/critic_{t}.pth')
            
            # Plot metrics
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(metrics['train_rewards'])
            plt.title('Training Rewards')
            
            plt.subplot(2, 2, 2)
            plt.plot(metrics['eval_rewards'])
            plt.title('Evaluation Rewards')
            
            plt.subplot(2, 2, 3)
            plt.plot(metrics['critic_losses'][-1000:])
            plt.title('Critic Loss (last 1000)')
            
            plt.subplot(2, 2, 4)
            plt.plot(metrics['actor_losses'][-1000:])
            plt.title('Actor Loss (last 1000)')
            
            plt.tight_layout()
            plt.savefig(f'{train_dir}/metrics_{t}.png')
            plt.close()


if __name__ == "__main__":
    main()