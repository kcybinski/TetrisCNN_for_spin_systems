"""Synthetic spin configurations: idealized paramagnetic, ferromagnetic and
antiferromagnetic states.

These are the out-of-distribution probes the symbolic-regression pipeline of App. I
tests its fitted equations on, and the same probes Figure6.ipynb pushes through a
trained network to see how its decision boundary behaves away from the experimental
data. They live here, rather than in tetriscnn/symbolic_regression.py where they were
first written, because nothing about them needs PySR: they are numpy draws wrapped in
a torch tensor. Keeping them in the SR module made importing them require the optional
PySR install, which is why Figure6.ipynb could no longer run on a plain installation.

They are idealized zero-temperature-like analogs of the three phases, not draws from
the true experimental distribution: real Rydberg snapshots carry projective-measurement
noise and finite-time dynamics these do not reproduce. Metrics computed on them bound
how far a reading extends, rather than resampling the data faithfully.
"""

from typing import Optional, Tuple

import numpy as np
import torch


def generate_paramagnetic_configs(
    grid_shape: Tuple[int, int],
    n_samples: int,
    device: torch.device,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Generate paramagnetic spin configurations (random ±1).

    Each spin independently +1 or -1 with 50% probability.

    Args:
        grid_shape: (height, width) of spin lattice
        n_samples: Number of configurations to generate
        device: PyTorch device
        rng: NumPy generator; a fresh unseeded one is used when None

    Returns:
        Tensor of shape (n_samples, 1, height, width) with ±1 spins
    """
    rng = np.random.default_rng() if rng is None else rng
    return torch.tensor(
        rng.choice([-1, 1], size=(n_samples, 1, *grid_shape)),
        dtype=torch.float32,
        device=device
    )


def generate_ferromagnetic_configs(
    grid_shape: Tuple[int, int],
    n_samples: int,
    num_flips: int = 4,
    device: torch.device = None,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Generate ferromagnetic spin configurations.

    Starts from uniform states (all +1 or all -1) with small perturbations.
    Half from all +1 with flips to -1, half from all -1 with flips to +1.

    Args:
        grid_shape: (height, width) of spin lattice
        n_samples: Number of configurations to generate
        num_flips: Number of spins to flip (3-4 typical)
        device: PyTorch device
        rng: NumPy generator; a fresh unseeded one is used when None

    Returns:
        Tensor of shape (n_samples, 1, height, width) with mostly aligned spins
    """
    rng = np.random.default_rng() if rng is None else rng
    height, width = grid_shape
    configs = []

    # Half from all +1 state
    for _ in range(n_samples // 2):
        config = np.ones((1, height, width), dtype=np.float32)
        flip_positions = rng.choice(height * width, size=num_flips, replace=False)
        for pos in flip_positions:
            i, j = divmod(pos, width)
            config[0, i, j] = -1
        configs.append(config)

    # Half from all -1 state
    for _ in range(n_samples - n_samples // 2):
        config = -np.ones((1, height, width), dtype=np.float32)
        flip_positions = rng.choice(height * width, size=num_flips, replace=False)
        for pos in flip_positions:
            i, j = divmod(pos, width)
            config[0, i, j] = +1
        configs.append(config)

    return torch.tensor(np.array(configs), dtype=torch.float32, device=device)


def generate_antiferromagnetic_configs(
    grid_shape: Tuple[int, int],
    n_samples: int,
    num_flips: int = 4,
    device: torch.device = None,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Generate antiferromagnetic spin configurations.

    Starts from checkerboard pattern with small perturbations.
    Two-fold degeneracy: 50% from (0,0)=+1 sublattice, 50% from (0,0)=-1 sublattice.

    Args:
        grid_shape: (height, width) of spin lattice
        n_samples: Number of configurations to generate
        num_flips: Number of spins to flip
        device: PyTorch device
        rng: NumPy generator; a fresh unseeded one is used when None

    Returns:
        Tensor of shape (n_samples, 1, height, width) with checkerboard pattern
    """
    rng = np.random.default_rng() if rng is None else rng
    height, width = grid_shape
    configs = []

    # Half from sublattice A: (0,0) = +1
    for _ in range(n_samples // 2):
        config = np.zeros((1, height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                config[0, i, j] = +1 if (i + j) % 2 == 0 else -1

        flip_positions = rng.choice(height * width, size=num_flips, replace=False)
        for pos in flip_positions:
            i, j = divmod(pos, width)
            config[0, i, j] *= -1

        configs.append(config)

    # Half from sublattice B: (0,0) = -1
    for _ in range(n_samples - n_samples // 2):
        config = np.zeros((1, height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                config[0, i, j] = -1 if (i + j) % 2 == 0 else +1

        flip_positions = rng.choice(height * width, size=num_flips, replace=False)
        for pos in flip_positions:
            i, j = divmod(pos, width)
            config[0, i, j] *= -1

        configs.append(config)

    return torch.tensor(np.array(configs), dtype=torch.float32, device=device)
