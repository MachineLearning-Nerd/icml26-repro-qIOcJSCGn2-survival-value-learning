from __future__ import annotations


def configure_visual_hsvl(config: dict) -> dict:
    config = dict(config)
    config.update(
        batch_size=256,
        discount=0.995,
        encoder="impala_small",
        low_actor_rep_grad=True,
        p_aug=0.5,
    )
    return config
