def compute_strides(shape):
    strides = [1]
    if (len(shape) == 1):
        return strides

    for dim in reversed(shape[1:]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))

def compute_shape_size(shape):
    size = 1
    for dim in shape:
        size *= dim
    return size

def compute_linear_index(indices, strides):
    if len(indices) != len(strides):
        raise ValueError(f"Invalid indices {indices} for strides {strides}")
 
    return sum(index * stride for index, stride in zip(indices, strides))

def reshape_linear_indices(index, shape):
    shape_size = compute_shape_size(shape)
    if (index >= shape_size):
        raise ValueError(f"index out of range for shape_size {shape_size}")
    
    new_indices = []

    remainder = index
    for dim in reversed(shape[1:]):
        new_indices.insert(0, remainder % dim)
        remainder = remainder // dim
    new_indices.insert(0, remainder)
    
    return new_indices

def expand_linear_indices(index, shape, targe_shape_dim):
    if (targe_shape_dim > len(shape) or targe_shape_dim <= 0):
        raise ValueError(f"Invalid targe_shape_dim {targe_shape_dim}")
    
    return reshape_linear_indices(index, shape[-targe_shape_dim:])

def compute_permuted_linear_index(original_linear_index, original_shape, perm, strides):
    original_indices = reshape_linear_indices(original_linear_index, original_shape)
    new_indices = [original_indices[p] for p in perm]
    return compute_linear_index(new_indices, strides)

def compute_inverse_perm(perm):
    inv_dims = [0] * len(perm)
    for idx, pos in enumerate(perm):
        inv_dims[pos] = idx
    return tuple(inv_dims)