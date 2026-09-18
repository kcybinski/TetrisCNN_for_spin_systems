import itertools
import numpy as np


def generate_patterns(shape, order, rotations=True, reflections=True, translations=True):
    """
    Generate all unique patterns of a given order on a grid of given shape,
    accounting for rotations, reflections, and translations.
    Input: shape: (rows, cols), e.g. (5, 5)
           order: number of sites in the pattern, e.g. 3
    Output: yields unique patterns as lists
    """
    assert order <= shape[0] * shape[1], "Order must be less than or equal to total number of sites."
    assert shape[0] == shape[1], "Currently only square grids are supported."
    
    all_sites = [(i, j) for i in range(shape[0]) for j in range(shape[1])] 
    unique_candidates = {}  # Use a dict instead of set for faster lookups

    for pattern in itertools.combinations(all_sites, order):
        pattern_set = frozenset(pattern)  
        equivalent_patterns = [pattern_set]

        if rotations:
            rotated_patterns = [pattern_set]
            for _ in range(3):  # Generate rotations
                pattern_set = rotate_pattern(pattern_set, shape[1], num_rotations=1)
                rotated_patterns.append(pattern_set)
            equivalent_patterns += rotated_patterns

        if reflections:
            # reflected_patterns = [reflect_pattern(temp_pattern, shape[1]) for temp_pattern in rotated_patterns]
            reflected_patterns = [reflect_pattern(temp_pattern, shape[1]) for temp_pattern in equivalent_patterns]
            equivalent_patterns += reflected_patterns

        # equivalent_patterns = rotated_patterns + reflected_patterns
        if translations:
            equivalent_patterns = [normalize_translation(p) for p in equivalent_patterns]

        # Check for all equivalent states in unique_candidates dictionary
        if not any(level in unique_candidates for level in equivalent_patterns):
            # Store only one of the equivalent patterns
            chosen_pattern = equivalent_patterns[0]
            unique_candidates[chosen_pattern] = True  # Mark presence
            temp = list(chosen_pattern)
            yield temp  # Yield each unique pattern



def rotate_pattern(pattern, width, num_rotations):
    """
    Input: e.g. frozenset({(0, 1), (1, 0), (1, 1)}), 5, num_rotations=1
    Output: e.g. frozenset({(1, 4), (0, 1), (1, 3)})
    """
    pattern_set = set(pattern)
    for _ in range(num_rotations % 4):
        pattern_set = {rotate_90(site, width) for site in pattern_set}
    return frozenset(pattern_set)

def reflect_pattern(pattern, width):
    """"
    Input: e.g. frozenset({(0, 1), (1, 0), (1, 1)})
    Output: e.g.  frozenset({(0, 3), (1, 4), (1, 3)})
    """
    pattern_set = set()
    for site in pattern:
        pattern_set.add((site[0], width - 1 - site[1]))
    return frozenset(pattern_set)

def rotate_90(site, width):
    """
    Input: e.g. (0, 0)
    Output: e.g. (0, 4)
    """
    rotated_site = (site[1], width - 1 - site[0])
    return rotated_site    


def normalize_translation(pattern):
    """
    Shift a pattern so that its minimum row and column start at (0,0).
    Input: e.g. frozenset({(1,1), (2,2)})
    Output: e.g. frozenset({(0,0), (1,1)})
    """
    min_x = min(x for x, _ in pattern)
    min_y = min(y for _, y in pattern)
    return frozenset({(x - min_x, y - min_y) for (x, y) in pattern})



def draw_pattern(pattern, width=5, verbose=True):
    """Draws the grid for the given state.
    Input: e.g. [(x1, y1), (x2, y2), ...] or
    frozenset({(x2,y2), ...})
    """
    if verbose:
        print(pattern)
    grid = np.zeros((width, width), dtype=int)

    for site in pattern:
        grid[site] = 1              # helper actors

    for row in grid:
        for cell in row:
            if cell == 0:
                print(".", end=" ")
            elif cell == 1:
                print("0", end=" ")
            else:
                print(cell, end=" ")
        print()









def pattern_correlator(data, pattern, window_shape=None):
    """
    Compute the correlation function of a 2D pattern on a batch of 2D grids.

    Parameters
    ----------
    data : 3D numpy array, shape (B, H, W)
        Batch of grids.
    pattern : list of (row, col) tuples 
        Define the pattern.
    window_shape : tuple of (height, width), optional
        If provided, use this as the bounding box instead of computing from pattern.
        Useful for enforcing a fixed window size (e.g., kernel shape) for consistent
        translation deduplication.
    Returns
    -------
    1D numpy array of shape (B,)
        The average product for each batch.
    """
    if data.ndim == 2:
        data = data[None, None, ...]     # (1, 1, H, W)
    elif data.ndim == 3:
        data = data[:, None, ...]        # (B, 1, H, W)
    elif data.ndim == 4:
        pass                             # already (B, C, H, W)
    else:
        raise ValueError("Data must be 2D, 3D, or 4D numpy array.")

    B, C, nrows, ncols = data.shape

    if window_shape is not None:
        max_row = window_shape[0] - 1
        max_col = window_shape[1] - 1
    else:
        max_row = max(r for r, _ in pattern)
        max_col = max(c for _, c in pattern)

    stacked = [
        data[..., dr:nrows - max_row + dr, dc:ncols - max_col + dc]
        for (dr, dc) in pattern
    ]

    stacked = np.stack(stacked, axis=0)   # (P, B, C, H', W')
    product = np.prod(stacked, axis=0)    # (B, C, H', W')
    mean = np.mean(product, axis=(-2, -1))  # (B, C)
    return mean.squeeze()  # -> (B,) or scalar if originally 2D
    # # Collect all shifted slices into a stack
    # stacked = []
    # for (dr, dc) in pattern: 
    #     stacked.append(data[..., dr:nrows - max_row + dr, dc:ncols - max_col + dc])

    # # Multiply along pattern dimension
    # stacked = np.stack(stacked, axis=0)  # shape (P, B, H', W')
    # product = np.prod(stacked, axis=0)   # shape (B, H', W')

    # # Average over spatial positions for each batch
    # return np.mean(product, axis=(1, 2)) # shape (B,)



def pattern_to_mask(pattern, width, height=None):
    """
    Convert a pattern into a binary mask (2D numpy array).
    """
    if height is None:
        height = width

    mask = np.zeros((height, width), dtype=int)

    for (r, c) in pattern:
        mask[r, c] = 1

    return mask


if __name__ == "__main__":
    shape = (3, 3)
    order = 5
    patterns = list(generate_patterns(shape, order, rotations=True, reflections=True, translations=True))

    print(f"Generated {len(patterns)} unique patterns of order {order} on a {shape[0]}x{shape[1]} grid.")
    for pattern in patterns:
        draw_pattern(pattern, width=shape[0], verbose=True)
        print()
