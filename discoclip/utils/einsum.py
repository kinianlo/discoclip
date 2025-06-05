from itertools import count

import opt_einsum as oe
from lambeq.backend.tensor import Diagram, Swap, Cup, Cap, Spider

def union_find(merges: list[tuple[int, int]]):
    """
    Given a list of merges, return a dictionary which maps
    an edge to its representative.
    """
    parent = {i: i for merge in merges for i in merge}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for x, y in merges:
        union(x, y)

    return {i: find(i) for i in parent}

def tn_to_einsum(diag: Diagram, interleaved: bool = False):
    """Convert a diagram to an einsum string expression,
    and a list of tensors to contract.
    arguments:
        interleaved: if True, return the data for the
        interleaved mode of einsum, i.e. a list of interleaved
        tensors and their indices, and finally the dangling
        indices. E.g. for the einsum string 'abc,cd->abd', the
        interleaved data would be:
        [tensor_1, [0, 1, 2], tensor_2, [2, 3], [0, 1, 3]]
    """
    idx_gen = count(0)

    merges = []
    tensors = []
    tensor_edges = [] 
    size_dict = {}

    def get_new_index(size):
        """Generate a new index and record its size"""
        new_index = next(idx_gen)
        size_dict[new_index] = size
        return new_index

    inputs = [get_new_index(size) for size in diag.dom.dim]
    scan = inputs[:]

    for layer in diag.layers:
        l, box, _ = layer.unpack()

        if isinstance(box, Swap):
            scan[len(l)], scan[len(l) + 1] = scan[len(l) + 1], scan[len(l)]

        elif isinstance(box, Cup):
            merges.append((scan[len(l)], scan[len(l) + 1]))
            scan = scan[:len(l)] + scan[len(l) + 2:]

        elif isinstance(box, Cap):
            new_edge = get_new_index(box.left.dim[0])
            scan = scan[:len(l)] + [new_edge, new_edge] + scan[len(l):]

        elif isinstance(box, Spider):
            new_edge = get_new_index(box.type.dim[0])
            merges.extend((scan[len(l) + i], new_edge) for i in range(len(box.dom)))
            output_edges = [new_edge for _ in range(len(box.cod))]
            scan = scan[:len(l)] + output_edges + scan[len(l) + len(box.dom):]
        else:
            input_edges = scan[len(l):len(l) + len(box.dom)]
            output_edges = [get_new_index(size) for size in box.cod.dim]
            tensors.append((box.data, box.dom.dim + box.cod.dim))
            tensor_edges.append(input_edges + output_edges)
            scan = scan[:len(l)] + output_edges + scan[len(l) + len(box.dom):]
    outputs = scan

    # Merge edges 
    repr = union_find(merges) 
    tensor_edges = [[repr.get(edge, edge) for edge in edges] for edges in tensor_edges]
    inputs = [repr.get(edge, edge) for edge in inputs]
    outputs = [repr.get(edge, edge) for edge in outputs]
    size_dict = {repr.get(edge, edge): size for edge, size in size_dict.items()}

    dangling = inputs + outputs
    if len(set(dangling)) != len(dangling):
        raise ValueError("Duplicate dangling indices found in the diagram. "
                         "This is not supported by the current implementation.")
  
    if not interleaved:
        subs = [''.join(oe.get_symbol(i) for i in indices) for indices in tensor_edges]
        output_subs = ''.join(oe.get_symbol(i) for i in dangling)
        einsum_string = ','.join(subs) + '->' + output_subs
        return einsum_string, tensors
    else:
        data = []
        for tensor, indices in zip(tensors, tensor_edges):
            data.append(tensor)
            data.append(indices)
        data.append(dangling)
        return data
        
def to_batched_einsum(einsum_string):
    return '...' + einsum_string.replace(',', ',...').replace('->', '->...')