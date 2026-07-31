from typing import Optional

import flax.linen as nn
import jax.numpy as jnp


class ImpalaStack(nn.Module):
    num_features: int
    num_blocks: int

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        x = nn.Conv(
            self.num_features,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=initializer,
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        for _ in range(self.num_blocks):
            identity = x
            x = nn.relu(x)
            x = nn.Conv(
                self.num_features,
                kernel_size=(3, 3),
                padding="SAME",
                kernel_init=initializer,
            )(x)
            x = nn.relu(x)
            x = nn.Conv(
                self.num_features,
                kernel_size=(3, 3),
                padding="SAME",
                kernel_init=initializer,
            )(x)
            x += identity
        return x


class ImpalaEncoder(nn.Module):
    width: int = 1
    stack_sizes: tuple[int, ...] = (16, 32, 32)
    num_blocks: int = 1
    mlp_hidden_dims: tuple[int, ...] = (512,)

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0
        for stack_size in self.stack_sizes:
            x = ImpalaStack(stack_size * self.width, self.num_blocks)(x)
        x = nn.relu(x)
        x = x.reshape((*x.shape[:-3], -1))
        for hidden_dim in self.mlp_hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.gelu(x)
        return x


class GCEncoder(nn.Module):
    """Helper module to handle inputs to goal-conditioned networks.

    It takes in observations (s) and goals (g) and returns the concatenation of `state_encoder(s)`, `goal_encoder(g)`,
    and `concat_encoder([s, g])`. It ignores the encoders that are not provided. This way, the module can handle both
    early and late fusion (or their variants) of state and goal information.
    """

    state_encoder: Optional[nn.Module] = None
    goal_encoder: Optional[nn.Module] = None
    concat_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
        reps = jnp.concatenate(reps, axis=-1)
        return reps
