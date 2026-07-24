from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube


def generate_grid_sample(
    n_samples: int, spans: List[Tuple[float, float]]
) -> torch.Tensor:
    """
    Generate uniform grid samples across given dimensions.

    If n_samples is not a perfect n_dims-th power, the full regular grid
    (nums_each_dim ** n_dims points) is subsampled down to n_samples by
    picking indices evenly spread across the flattened grid. A plain
    prefix slice would instead drop only the tail of the flattened grid,
    which cuts a whole edge of the domain rather than thinning it evenly.

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

    if grid_points.shape[0] == n_samples:
        return grid_points

    indices = torch.linspace(0, grid_points.shape[0] - 1, n_samples).round().long()
    return grid_points[indices]


def generate_lhs_sample(
    n_samples: int, spans: List[Tuple[float, float]], seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate Latin Hypercube samples across given dimensions.

    Args:
        n_samples: Total number of desired points.
        spans: A list of (min, max) bounds for each dimension.
        seed: Seed for scipy's LHS engine. `scipy.stats.qmc.LatinHypercube`
            keeps its own random state, independent of `torch.manual_seed`
            -- without passing a seed here, LHS-sampled points differ
            between runs even with the torch RNG seeded.

    Returns:
        A PyTorch tensor containing the LHS points, shape (n_samples, len(spans)).
    """
    lhs = LatinHypercube(d=len(spans), seed=seed)
    lhs_samples = lhs.random(n=n_samples)

    # Scale each column to the corresponding physical span
    for i, span in enumerate(spans):
        lhs_samples[:, i] = span[0] + (span[1] - span[0]) * lhs_samples[:, i]

    return torch.tensor(lhs_samples, dtype=torch.float32)


def generate_random_sample(
    n_samples: int, spans: List[Tuple[float, float]], seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate uniform random samples across given dimensions.

    Args:
        n_samples: Total number of desired points.
        spans: A list of (min, max) bounds for each dimension.
        seed: If given, seeds torch's global RNG immediately before
            sampling, for reproducible output.

    Returns:
        A PyTorch tensor containing the random points.
    """
    if seed is not None:
        torch.manual_seed(seed)
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

    @staticmethod
    def _sample_points(
        num_points: int,
        spans: List[Tuple[float, float]],
        method: str = "lhs",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate sample points using the selected sampling strategy.

        Args:
            num_points: Number of points to generate.
            spans: Sampling bounds for each dimension.
            method: Sampling strategy ('grid', 'lhs', or 'random').
            seed: Seed for reproducible 'lhs'/'random' sampling. Ignored
                for 'grid', which is already deterministic.

        Returns:
            Sampled points.
        """
        if method == "grid":
            return generate_grid_sample(num_points, spans)

        if method == "lhs":
            return generate_lhs_sample(num_points, spans, seed=seed)

        if method == "random":
            return generate_random_sample(num_points, spans, seed=seed)

        raise ValueError("Method must be 'grid', 'lhs', or 'random'.")

    def sample(
        self,
        num_dom: int,
        num_ic: int,
        num_bc: int,
        methods: Tuple[str, str, str] = ("lhs", "grid", "grid"),
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate all required collocation points for the PINN.

        Args:
            num_dom: Number of interior domain points.
            num_ic: Number of initial condition points.
            num_bc: Number of boundary points (per boundary).
            methods: Sampling methods for [domain, IC, BCs]. Defaults to ("lhs", "grid", "grid").
            device: Device to store the tensors on.
            seed: If given, makes 'lhs'/'random' sampling reproducible.
                Domain/IC/BC each get a distinct derived seed so they
                don't draw identical sequences from each other.

        Returns:
            Tuple containing (domain_points, ic_points, bc_pile_points, bc_far_points).
        """
        dom_seed = None if seed is None else seed
        ic_seed = None if seed is None else seed + 1
        bc_seed = None if seed is None else seed + 2

        dom_points = self.sample_domain_points(
            num_dom, method=methods[0], device=device, seed=dom_seed
        )
        ic_points = self.sample_ic_points(
            num_ic, method=methods[1], device=device, seed=ic_seed
        )
        bc_pile_points, bc_far_points = self.sample_bc_points(
            num_bc, method=methods[2], device=device, seed=bc_seed
        )

        return dom_points, ic_points, bc_pile_points, bc_far_points

    def sample_domain_points(
        self,
        num_points: int,
        method: str = "lhs",
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample interior domain points.

        Points are slightly offset from boundaries to avoid singularities during
        PDE residual calculations (e.g., dividing by zero at r=0).

        Args:
            num_points: Number of domain points to generate.
            method: Sampling method ('grid', 'lhs', or 'random').
            device: Target device for the tensor.
            seed: If given, makes 'lhs'/'random' sampling reproducible.

        Returns:
            PyTorch tensor of domain points, shape (num_points, 2).
        """
        # Offset to prevent dividing by zero or strictly overlapping with BCs/ICs
        safe_spans = [(span[0] + 1e-5, span[1] - 1e-5) for span in self.spans]

        points = self._sample_points(num_points, safe_spans, method, seed=seed)

        return points.to(torch.device(device))

    def sample_ic_points(
        self,
        num_points: int,
        method: str = "grid",
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample initial condition points (t* = 0).

        Args:
            num_points: Number of IC points to generate.
            method: Sampling method ('grid', 'lhs', or 'random').
            device: Target device for the tensor.
            seed: If given, makes 'lhs'/'random' sampling reproducible.

        Returns:
            PyTorch tensor of IC points, shape (num_points, 2).
        """
        r_span = [(self.spans[0][0], self.spans[0][1])]

        if method == "grid":
            r = torch.linspace(r_span[0][0], r_span[0][1], num_points).reshape(-1, 1)
        else:
            r = self._sample_points(num_points, r_span, method, seed=seed)

        t = torch.zeros_like(r)

        return torch.cat((r, t), dim=1).to(torch.device(device))

    def sample_bc_points(
        self,
        num_points: int,
        method: str = "grid",
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample boundary condition points at the inner pile (r_min) and far-field (r_max).

        Args:
            num_points: Number of points to generate per boundary.
            method: Sampling method ('grid', 'lhs', or 'random').
            device: Target device for the tensor.
            seed: If given, makes 'lhs'/'random' sampling reproducible.

        Returns:
            Tuple containing (pile_boundary_points, far_field_boundary_points).
        """
        t_span = [(self.spans[1][0], self.spans[1][1])]

        if method == "grid":
            t = torch.linspace(t_span[0][0], t_span[0][1], num_points).reshape(-1, 1)
        else:
            t = self._sample_points(num_points, t_span, method, seed=seed)

        r_pile = torch.full_like(t, self.spans[0][0])
        r_far = torch.full_like(t, self.spans[0][1])

        bc_pile_points = torch.cat((r_pile, t), dim=1)
        bc_far_points = torch.cat((r_far, t), dim=1)

        return bc_pile_points.to(torch.device(device)), bc_far_points.to(
            torch.device(device)
        )


if __name__ == "__main__":
    spans = [
        (0.0, 1.0),  # r*
        (0.0, 2.0),  # t*
    ]

    sampler = GeothermalSampler(spans)

    methods_to_test = ["grid", "lhs", "random"]

    for method in methods_to_test:
        print(f"\n{'=' * 20}")
        print(f"Testing '{method}' sampling")
        print(f"{'=' * 20}")

        dom_points, ic_points, bc_pile_points, bc_far_points = sampler.sample(
            num_dom=20,
            num_ic=10,
            num_bc=10,
            methods=(method, method, method),
        )

        print(f"Domain points shape:  {dom_points.shape}")
        print(f"IC points shape:      {ic_points.shape}")
        print(f"Pile BC shape:        {bc_pile_points.shape}")
        print(f"Far BC shape:         {bc_far_points.shape}")

        print("\nFirst 5 domain points:")
        print(dom_points[:5])

        print("\nFirst 5 IC points:")
        print(ic_points[:5])

        print("\nFirst 5 pile BC points:")
        print(bc_pile_points[:5])

        print("\nFirst 5 far BC points:")
        print(bc_far_points[:5])

        print("\nTesting non-perfect grid (95 points)...")
        grid = generate_grid_sample(95, spans)
        print(f"Generated shape: {grid.shape}")
        print(f"Unique r values: {len(torch.unique(grid[:, 0]))}")
        print(f"Unique t values: {len(torch.unique(grid[:, 1]))}")
