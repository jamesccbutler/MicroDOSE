from collections import OrderedDict
from dataclasses import fields, is_dataclass
from typing import Any, Tuple

import numpy as np

# Check if torch is available
try:
    import torch
    _TORCH_AVAILABLE = True
    _TORCH_VERSION = torch.__version__
except ImportError:
    _TORCH_AVAILABLE = False
    _TORCH_VERSION = None


def is_tensor(x) -> bool:
    """
    Tests if `x` is a `torch.Tensor` or `np.ndarray`.
    """
    return isinstance(x, np.ndarray) or (_TORCH_AVAILABLE and isinstance(x, torch.Tensor))


class BaseOutput(OrderedDict):
    """
    Base class for all model outputs as a dataclass. Allows indexing like a dictionary or tuple.
    """

    def __init_subclass__(cls) -> None:
        """Register subclasses as PyTorch pytree nodes."""
        if _TORCH_AVAILABLE:
            import torch.utils._pytree

            if _TORCH_VERSION < "2.2":
                torch.utils._pytree._register_pytree_node(
                    cls,
                    torch.utils._pytree._dict_flatten,
                    lambda values, context: cls(**torch.utils._pytree._dict_unflatten(values, context)),
                )
            else:
                torch.utils._pytree.register_pytree_node(
                    cls,
                    torch.utils._pytree._dict_flatten,
                    lambda values, context: cls(**torch.utils._pytree._dict_unflatten(values, context)),
                )

    def __post_init__(self) -> None:
        class_fields = fields(self)

        # Ensure the class has fields
        if not class_fields:
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and isinstance(first_field, dict):
            for key, value in first_field.items():
                self[key] = value
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __getitem__(self, k: Any) -> Any:
        if isinstance(k, str):
            return dict(self.items())[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name: Any, value: Any) -> None:
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any, ...]:
        """Convert self to a tuple containing all non-None attributes."""
        return tuple(self[k] for k in self.keys())
