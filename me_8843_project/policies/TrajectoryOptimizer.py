import logging
import time

import torch
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrajectoryOptimizerPolicy:
    def __init__(
        self,
        models,
        *args,
        threshold_flag=False,
        convergence_threshold=0.01,
        lr=1e-3,
        n_planning_steps=10,
        initial_constraint_multiplier=1.0,
        **kwargs,
    ):
        self.encoder = models["encoder"]
        self.transition_model = models["transition"]
        self.reward_model = models["reward"]
        self.threshold_flag = threshold_flag
        self.convergence_threshold = convergence_threshold
        self.lr = lr
        self.n_planning_steps = n_planning_steps
        self.initial_constraint_multiplier = initial_constraint_multiplier
        self.mse_loss = nn.MSELoss()

    def __call__(self, initial_state):
        # Initialize state and action parameters
        states = torch.rand((self.n_planning_steps, self.transition_model.state_dim))
        actions = torch.rand((self.n_planning_steps, self.transition_model.action_dim))

        # Shift to GPU if available
        if torch.cuda.is_available():
            initial_state = initial_state.cuda()
            states = states.cuda()
            actions = actions.cuda()

        # Make them parameters
        states_param = nn.Parameter(states)
        actions_param = nn.Parameter(actions)

        # Redefine to use
        states = torch.cat((initial_state, states_param), dim=0)  # Add initial state
        actions = actions_param

        # Initialize lambda constraint
        constraint_multiplier = nn.Parameter(
            torch.Tensor([self.initial_constraint_multiplier])
        )

        # Optimize trajectory
        trajectory_optimizer = torch.optim.Adam(
            [states_param, actions_param], lr=self.lr
        )
        constraint_optimizer = torch.optim.Adam([constraint_multiplier], lr=self.lr)

        # Continue until delta is small enough
        start_time = time.time()
        plan_count = 0
        prev_result = None
        original_result = None
        delta = float("-inf")
        # while self.convergence_check(delta, plan_count):
        for plan_count in tqdm(range(self.n_planning_steps)):
            raw_result = self.trajectory_eval_fn(
                states, actions, constraint_multiplier
            )  # Minimize negative
            result = -1 * raw_result

            # On even steps, optimize trajectory
            # On odd steps, optimize constraint multiplier
            if plan_count % 2 == 0:
                trajectory_optimizer.zero_grad()
                result.backward()
                trajectory_optimizer.step()
            else:
                constraint_optimizer.zero_grad()
                raw_result.backward()
                constraint_optimizer.step()

            if prev_result is None:
                prev_result = result
                original_raw_result = raw_result
            else:
                delta = result - prev_result
                prev_result = result

            plan_count += 1

        logger.info(
            f"Trajectory optimization elapsed time: {time.time() - start_time} seconds"
        )

        # logger.info(
        #     f"Original result: {original_raw_result}, Final result: {raw_result}, difference: {raw_result - original_raw_result}"
        # )
        return actions[0].detach().cpu().numpy()

    def convergence_check(self, delta, plan_count):
        if self.threshold_flag:
            return delta < self.convergence_threshold
        else:
            return plan_count < self.n_planning_steps

    def trajectory_eval_fn(self, states, actions, constraint_multiplier):
        return self.reward_fn(states) + constraint_multiplier * self.constraint(
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
