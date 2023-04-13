import torch
from torch import nn


class TrajectoryOptimizerPolicy:
    def __init__(self, models, *args, n_planning_steps=10, **kwargs):
        self.encoder = models["encoder"]
        self.transition_model = models["transition_model"]
        self.reward_model = models["reward_model"]
        self.n_planning_steps = n_planning_steps
        self.constraint_multiplier = 0.1
        self.mse_loss = nn.MSELoss()

    def __call__(self, initial_state):
        # Initialize state and action parameters
        states_param = torch.rand(
            (self.n_planning_steps, self.transition_model.state_dim)
        )
        states_param = nn.Parameter(states_param)  # Make it a parameter
        actions = torch.rand((self.n_planning_steps, self.transition_model.action_dim))
        actions_param = nn.Parameter(actions)

        # Redefine to use
        states = torch.cat((initial_state, states_param), dim=0)  # Add initial state
        actions = actions_param

        if torch.cuda.is_available():
            states_param.cuda()
            actions_param.cuda()
            states.cuda()
            actions.cuda()

        # Optimize trajectory
        optimizer = torch.optim.Adam([states_param, actions_param], lr=1e-3)
        for _ in range(self.n_planning_steps):
            result = self.trajectory_eval_fn(states, actions)
            optimizer.zero_grad()
            result.backward()
            optimizer.step()

        return actions[0].detach().cpu().numpy()

    def trajectory_eval_fn(self, states, actions):
        return self.reward_fn(states) + self.constraint_multiplier * self.constraint(
            states, actions
        )

    def reward_fn(self, states):
        # Evaluate the trajectory
        rewards = self.reward_model(states).sum()
        return rewards

    def constraint(self, states, actions):
        # Evaluate the trajectory
        next_states = self.transition_model(states[:-1], actions)
        loss = self.mse_loss(states[1:], next_states)
        return loss
