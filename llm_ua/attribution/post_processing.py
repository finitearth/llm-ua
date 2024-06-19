import numpy as np


def normalize_attr_map(
    attr_map: np.ndarray,
    absolute: bool = True,
    min_max: bool = True,
    to_image: bool = False,
) -> np.ndarray:
    assert attr_map.ndim == 3 and attr_map.shape[0] == 1, \
        f'Attribution map of a single sample should have shape (1, seq, hidden_dim), but got: {attr_map.shape}'
    if absolute:
        attr_map = np.abs(attr_map)
    # avg across hidden dimension, and remove the batch dimension
    attr_map = attr_map.mean(-1)[0]

    if min_max:
        min_val = attr_map.min()
        max_val = attr_map.max()
        attr_map = (attr_map - min_val) / (max_val - min_val + 1e-8)
        if to_image:
            # to single-channel image
            attr_map = np.clip(attr_map * 255, a_min=0, a_max=255).astype(np.uint8)
    else:
        if to_image:
            raise ValueError(
                'min_max is False, but to_image is True. The attribution map needs to be normalized to '
                '[0, 1] before being converted to image.')
    return attr_map
