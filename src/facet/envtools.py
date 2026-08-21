"""Environment construction and featurization helpers."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
)


def make_env(layout: str, horizon: int = 400):
    """Create an OvercookedEnv with old (2019 human-data) dynamics."""
    mdp = OvercookedGridworld.from_layout_name(layout, old_dynamics=True)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    # Force construction of the MediumLevelActionManager used by
    # featurize_state_mdp so later calls are cheap and deterministic.
    env.featurize_state_mdp(env.state)
    return env


def featurize(env, state) -> tuple:
    """(obs_p0, obs_p1) 96-dim float arrays for both players."""
    return env.featurize_state_mdp(state)


def parse_state(mdp_configured_first, raw):
    """Parse a dataframe 'state' cell into an OvercookedState.

    The layout's OvercookedGridworld must have been constructed first so
    the Recipe class is configured.
    """
    import json

    if isinstance(raw, str):
        raw = json.loads(raw)
    return OvercookedState.from_dict(raw)


def parse_joint_action(raw) -> tuple:
    import json

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.decoder.JSONDecodeError:
            raw = eval(raw)  # 'INTERACT' single-quote quirk in the data
    out = []
    for a in raw:
        if isinstance(a, list):
            out.append(tuple(a))
        elif isinstance(a, str):
            out.append(a.lower())
        else:
            out.append(a)
    return tuple(out)
