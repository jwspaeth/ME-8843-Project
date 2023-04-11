class RandomPolicy:
    def __init__(self, env, *args, **kwargs):
        self.env = env

    def __call__(self, *args, **kwargs):
        return self.env.action_space.sample()
