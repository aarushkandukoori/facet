"""FACET: Fictitious Agent Co-training for Emergent Teamwork.

Population-based reinforcement learning for human-compatible multiplayer
agents, evaluated on the Overcooked-AI benchmark with real human gameplay
data. Research code for Jewel Labs (jewellabs.org).
"""

LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "random0",  # canonical name: forced_coordination
    "random3",  # canonical name: counter_circuit
]

CANONICAL_NAMES = {
    "cramped_room": "Cramped Room",
    "asymmetric_advantages": "Asymmetric Advantages",
    "coordination_ring": "Coordination Ring",
    "random0": "Forced Coordination",
    "random3": "Counter Circuit",
}

OBS_DIM = 96
N_ACTIONS = 6
HORIZON = 400
