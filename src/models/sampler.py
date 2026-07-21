import math
from typing import List, Tuple

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube


def generate_grid_sample(n_samples: int, spans: List[Tuple[float, float]]) -> torch.Tensor:
    """
    Generate uniform grid samples across given dimensions.

    Args:
        n_samples: Total number of desired points.
        spans: A list of (min, max) bounds for each dimension.

    Returns:
        A PyTorch tensor containing the generated grid points.
    """
    n_dims = len(spans)
    nums_each_dim = int(np.ceil(n_samples ** (1.0 / n_dims)))

    grids = [torch.linspace(span[0], span[1], nums_each_dim) for span in spans]
    grid = torch.meshgrid(*grids, indexing="ij")
    grid_points = torch.stack([g.flatten() for g in grid], dim=1)

    return grid_points[:n_samples]


def generate_lhs_sample(n_samples: int, spans: List[Tuple[float, float]]) -> torch.Tensor:
    """
    Generate Latin Hypercube samples across given dimensions.

    Args:
        n_samples: Total number of desired points.
        spans: A list of (min, max) bounds for each dimension.

    Returns:
        A PyTorch tensor containing the LHS points, shape (n_samples, len(spans)).
    """
    lhs = LatinHypercube(d=len(spans))
    lhs_samples = lhs.random(n=n_samples)

    # Scale each column to the corresponding physical span
    for i, span in enumerate(spans):
        lhs_samples[:, i] = span[0] + (span[1] - span[0]) * lhs_samples[:, i]

    return torch.tensor(lhs_samples, dtype=torch.float32)


def generate_random_sample(n_samples: int, spans: List[Tuple[float, float]]) -> torch.Tensor:
    """
    Generate uniform random samples across given dimensions.

    Args:
        n_samples: Total number of desired points.
        spans: A list of (min, max) bounds for each dimension.

    Returns:
        A PyTorch tensor containing the random points.
    """
    rand = torch.rand(n_samples, len(spans))

    # Scale each column to the corresponding physical span
    for i, span in enumerate(spans):
        rand[:, i] = rand[:, i] * (span[1] - span[0]) + span[0]

    return rand


class GeothermalSampler:
    """Data sampler for the Geothermal PINN domain, initial, and boundary conditions.

    This class handles the generation of collocation points for dimensionless radius (r*) 
    and time (t*) using various sampling strategies (Grid, LHS, Random).
    """

    def __init__(self, spans: List[Tuple[float, float]]):
        """
        Initialize the Sampler.

        Args:
            spans: A list containing tuples of (min, max) for each dimension.
                Expected format: [(r_min, r_max), (t_min, t_max)].
        """
        self.spans = spans

    def sample(
        self,
        num_dom: int,
        num_ic: int,
        num_bc: int,
        methods: Tuple[str, str, str] = ("lhs", "grid", "grid"),
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate all required collocation points for the PINN.

        Args:
            num_dom: Number of interior domain points.
            num_ic: Number of initial condition points.
            num_bc: Number of boundary points (per boundary).
            methods: Sampling methods for [domain, IC, BCs]. Defaults to ("lhs", "grid", "grid").
            device: Device to store the tensors on.

        Returns:
            Tuple containing (domain_points, ic_points, bc_pile_points, bc_far_points).
        """
        dom_points = self.sample_domain_points(num_dom, method=methods[0], device=device)
        ic_points = self.sample_ic_points(num_ic, method=methods[1], device=device)
        bc_pile_points, bc_far_points = self.sample_bc_points(num_bc, method=methods[2], device=device)

        return dom_points, ic_points, bc_pile_points, bc_far_points

    def sample_domain_points(
        self,
        num_points: int,
        method: str = "lhs",
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Sample interior domain points.

        Points are slightly offset from boundaries to avoid singularities during
        PDE residual calculations (e.g., dividing by zero at r=0).

        Args:
            num_points: Number of domain points to generate.
            method: Sampling method ('grid', 'lhs', or 'random').
            device: Target device for the tensor.

        Returns:
            PyTorch tensor of domain points, shape (num_points, 2).
        """
        # Offset to prevent dividing by zero or strictly overlapping with BCs/ICs
        safe_spans = [(span[0] + 1e-5, span[1] - 1e-5) for span in self.spans]

        if method == "grid":
            points = generate_grid_sample(num_points, safe_spans)
        elif method == "lhs":
            points = generate_lhs_sample(num_points, safe_spans)
        elif method == "random":
            points = generate_random_sample(num_points, safe_spans)
        else:
            raise ValueError("Method must be 'grid', 'lhs', or 'random'.")
            
        return points.to(torch.device(device))

    def sample_ic_points(
        self,
        num_points: int,
        method: str = "grid",
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Sample initial condition points (t* = 0).

        Args:
            num_points: Number of IC points to generate.
            method: Sampling method ('grid', 'lhs', or 'random').
            device: Target device for the tensor.

        Returns:
            PyTorch tensor of IC points, shape (num_points, 2).
        """
        r_span = [(self.spans[0][0], self.spans[0][1])]

        if method == "grid":
            r = torch.linspace(r_span[0][0], r_span[0][1], num_points).reshape(-1, 1)
        elif method == "lhs":
            r = generate_lhs_sample(num_points, r_span)
        elif method == "random":
            r = generate_random_sample(num_points, r_span)
        else:
            raise ValueError("Method must be 'grid', 'lhs', or 'random'.")

        t = torch.zeros_like(r)
        
        return torch.cat((r, t), dim=1).to(torch.device(device))

    def sample_bc_points(
        self,
        num_points: int,
        method: str = "grid",
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample boundary condition points at the inner pile (r_min) and far-field (r_max).

        Args:
            num_points: Number of points to generate per boundary.
            method: Sampling method ('grid', 'lhs', or 'random').
            device: Target device for the tensor.

        Returns:
            Tuple containing (pile_boundary_points, far_field_boundary_points).
        """
        t_span = [(self.spans[1][0], self.spans[1][1])]

        if method == "grid":
            t = torch.linspace(t_span[0][0], t_span[0][1], num_points).reshape(-1, 1)
        elif method == "lhs":
            t = generate_lhs_sample(num_points, t_span)
        elif method == "random":
            t = generate_random_sample(num_points, t_span)
        else:
            raise ValueError("Method must be 'grid', 'lhs', or 'random'.")

        r_pile = torch.full_like(t, self.spans[0][0])
        r_far = torch.full_like(t, self.spans[0][1])

        bc_pile_points = torch.cat((r_pile, t), dim=1)
        bc_far_points = torch.cat((r_far, t), dim=1)

        return bc_pile_points.to(torch.device(device)), bc_far_points.to(torch.device(device))