from .matching import NeuralCollaborativeFiltering


def matching_fn_builder(**kwargs):
    cls = NeuralCollaborativeFiltering
    return cls(**kwargs)