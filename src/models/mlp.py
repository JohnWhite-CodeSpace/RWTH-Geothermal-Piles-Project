"""
Fully-connected network architecture used as the PINN's underlying net_u.
"""

from typing import Sequence, Tuple, Type, Union

import torch.nn as nn

# Single source of truth for the project's default architecture (6
# hidden layers of 64 neurons). Every script that builds a "standard"
# GeothermalPINN should import these rather than repeating the numbers.
DEFAULT_HIDDEN_LAYERS = 6
DEFAULT_HIDDEN_WIDTH = 64
DEFAULT_ARCHITECTURE: Tuple[int, ...] = (DEFAULT_HIDDEN_WIDTH,) * DEFAULT_HIDDEN_LAYERS


def build_mlp(
    in_dim: int = 2,
    out_dim: int = 2,
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
    hidden_width: Union[int, Sequence[int]] = DEFAULT_HIDDEN_WIDTH,
    activation: Type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    """
    Build a fully-connected feedforward network for the PINN.

    Uses Tanh activations rather than ReLU: the PDE residual needs second
    derivatives (T_rr, u_rr) computed through the network via autograd,
    which requires an activation that stays smooth (nonzero second
    derivative), unlike ReLU.

    Args:
        in_dim: Number of input features (r, t).
        out_dim: Number of output features (T, u).
        hidden_layers: Number of hidden layers. Ignored when
            hidden_width is a sequence -- its length determines the
            number of hidden layers instead.
        hidden_width: Either a single int (uniform width for every
            hidden layer) or a sequence of per-layer widths, e.g.
            [128, 64, 32, 64, 128] for a non-uniform ("hourglass")
            architecture.
        activation: Activation module class applied after each hidden layer.

    Returns:
        An nn.Sequential mapping (N, in_dim) -> (N, out_dim).
    """
    if isinstance(hidden_width, int):
        widths = [hidden_width] * hidden_layers
    else:
        widths = list(hidden_width)

    layers = []
    prev_dim = in_dim
    for width in widths:
        layers += [nn.Linear(prev_dim, width), activation()]
        prev_dim = width
    layers.append(nn.Linear(prev_dim, out_dim))

    net = nn.Sequential(*layers)
    _init_weights(net)
    return net


def _init_weights(net: nn.Sequential) -> None:
    """Apply Xavier/Glorot initialization to all linear layers."""
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
