from src.tensor import Tensor  # Assuming your Tensor class is here


class Parameter(Tensor):
    """
    A subclass of Tensor used to indicate that this tensor is a trainable parameter.
    """
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)
        self.is_parameter = True  # mark this as a parameter
