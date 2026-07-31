from __future__ import annotations

from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from hsvl.utils.actors import GCActor
from hsvl.utils.common import create_log_bins, get_activation_fn
from hsvl.utils.critics import GCSurvivalValue
from hsvl.utils.datasets import HGCDataset_sample
from hsvl.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from hsvl.utils.survival import create_perbin_nll_fn, create_perbin_value_fn


class FlatSVL(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    value_from_logits: Any = nonpytree_field()
    nll_from_logits: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params):
        pair1, pair2 = self.network.select("critic")(
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
            params=grad_params,
        )
        nll1 = self.nll_from_logits(
            pair1[0],
            pair1[1],
            batch["surv_is_event"],
            batch["surv_time"],
            batch["surv_censor_time"],
        )
        nll2 = self.nll_from_logits(
            pair2[0],
            pair2[1],
            batch["surv_is_event"],
            batch["surv_time"],
            batch["surv_censor_time"],
        )
        valid = batch["surv_valid"]
        loss = (valid * (nll1 + nll2)).sum() / jnp.maximum(valid.sum(), 1.0)
        return loss, {"critic/nll": loss}

    def actor_loss(self, batch, grad_params):
        dist = self.network.select("actor")(
            batch["observations"], batch["actor_goals"], params=grad_params
        )
        actor_actions = jnp.clip(dist.mode(), -1, 1)
        pair1, pair2 = self.network.select("critic")(
            batch["observations"], batch["actor_goals"], actor_actions
        )
        q1 = self.value_from_logits(pair1[0], pair1[1])
        q2 = self.value_from_logits(pair2[0], pair2[1])
        q = jnp.minimum(q1, q2)
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(batch["actions"])
        bc_loss = -(float(self.config["actor_alpha"]) * log_prob).mean()
        loss = q_loss + bc_loss
        return loss, {
            "actor/loss": loss,
            "actor/q_loss": q_loss,
            "actor/bc_loss": bc_loss,
            "actor/q_mean": q.mean(),
            "actor/bc_log_prob": log_prob.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params):
        critic_loss, info = self.critic_loss(batch, grad_params)
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        info.update(actor_info)
        info["total_loss"] = critic_loss + actor_loss
        return critic_loss + actor_loss, info

    @jax.jit
    def update(self, batch):
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params)

        network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=network), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        dist = self.network.select("actor")(
            observations, goals, temperature=temperature
        )
        return jnp.clip(dist.sample(seed=seed), -1, 1)

    def sample_and_update(self, dataset, batch_size: int, dataset_size: int):
        batch, rng = HGCDataset_sample(
            self.rng, dataset, batch_size, dataset_size, self.config
        )
        agent, info = self.update(batch)
        return agent.replace(rng=rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        activation = get_activation_fn(config["activation_fn"])
        bins, num_bins = create_log_bins(
            int(config["surv_horizon"]), int(config["num_log_bins"])
        )
        critic = GCSurvivalValue(
            hidden_dims=config["value_hidden_dims"],
            num_bins=int(num_bins),
            k_basis=config["k_basis"],
            num_basis_sets=config["num_basis_sets"],
            layer_norm=config["layer_norm"],
            activation=activation,
            ensemble=True,
            use_residual=False,
        )
        actor = GCActor(
            hidden_dims=(512, 512, 512, 512, 512, 512),
            action_dim=int(ex_actions.shape[-1]),
            state_dependent_std=False,
            const_std=True,
            activation=activation,
            use_residual=False,
            layer_norm=False,
        )
        ex_goals = ex_observations
        network_info = {
            "critic": (critic, (ex_observations, ex_goals, ex_actions)),
            "actor": (actor, (ex_observations, ex_goals)),
        }
        network_def = ModuleDict({name: item[0] for name, item in network_info.items()})
        network_args = {name: item[1] for name, item in network_info.items()}
        params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(
            model_def=network_def,
            params=params,
            tx=optax.adam(float(config["lr"])),
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            value_from_logits=create_perbin_value_fn(
                bins, float(config["discount"]), bool(config["add_tail"])
            ),
            nll_from_logits=create_perbin_nll_fn(bins),
        )
