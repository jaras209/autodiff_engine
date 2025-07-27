"""Operation classes for autodiff computation graph."""
from __future__ import annotations
from typing import Tuple
import abc
import math

class Operation(abc.ABC):
    """Abstract base class for all operations in the computation graph."""
    @abc.abstractmethod
    def forward(self, *inputs: float) -> float:
        pass

    @abc.abstractmethod
    def backward(self, output_grad: float, *inputs: float) -> Tuple[float, ...]:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

class Add(Operation):
    def forward(self, a: float, b: float) -> float:
        return a + b

    def backward(self, output_grad: float, a: float, b: float) -> Tuple[float, float]:
        return output_grad, output_grad

    def __str__(self) -> str:
        return '+'

class Mul(Operation):
    def forward(self, a: float, b: float) -> float:
        return a * b

    def backward(self, output_grad: float, a: float, b: float) -> Tuple[float, float]:
        return output_grad * b, output_grad * a

    def __str__(self) -> str:
        return '*'

class Neg(Operation):
    def forward(self, a: float) -> float:
        return -a

    def backward(self, output_grad: float, a: float) -> Tuple[float]:
        return (-output_grad,)

    def __str__(self) -> str:
        return 'neg'

class Sub(Operation):
    def forward(self, a: float, b: float) -> float:
        return a - b

    def backward(self, output_grad: float, a: float, b: float) -> Tuple[float, float]:
        return output_grad, -output_grad

    def __str__(self) -> str:
        return '-'

class Div(Operation):
    def forward(self, a: float, b: float) -> float:
        return a / b

    def backward(self, output_grad: float, a: float, b: float) -> Tuple[float, float]:
        return output_grad / b, -output_grad * a / (b * b)

    def __str__(self) -> str:
        return '/'

class Pow(Operation):
    def forward(self, a: float, b: float) -> float:
        return a ** b

    def backward(self, output_grad: float, a: float, b: float) -> Tuple[float, float]:
        da = output_grad * b * (a ** (b - 1))
        db = output_grad * (a ** b) * math.log(a) if a > 0 else 0.0
        return da, db

    def __str__(self) -> str:
        return '**'

class Exp(Operation):
    def forward(self, x: float) -> float:
        return math.exp(x)

    def backward(self, output_grad: float, x: float) -> Tuple[float]:
        return (output_grad * math.exp(x),)

    def __str__(self) -> str:
        return 'exp'

class Log(Operation):
    def forward(self, x: float) -> float:
        return math.log(x)

    def backward(self, output_grad: float, x: float) -> Tuple[float]:
        return (output_grad / x,)

    def __str__(self) -> str:
        return 'log'

class Sin(Operation):
    def forward(self, x: float) -> float:
        return math.sin(x)

    def backward(self, output_grad: float, x: float) -> Tuple[float]:
        return (output_grad * math.cos(x),)

    def __str__(self) -> str:
        return 'sin'

class Cos(Operation):
    def forward(self, x: float) -> float:
        return math.cos(x)

    def backward(self, output_grad: float, x: float) -> Tuple[float]:
        return (-output_grad * math.sin(x),)

    def __str__(self) -> str:
        return 'cos'

class Tan(Operation):
    def forward(self, x: float) -> float:
        return math.tan(x)

    def backward(self, output_grad: float, x: float) -> Tuple[float]:
        return (output_grad / (math.cos(x) ** 2),)

    def __str__(self) -> str:
        return 'tan'

class Cot(Operation):
    def forward(self, x: float) -> float:
        return 1.0 / math.tan(x)

    def backward(self, output_grad: float, x: float) -> Tuple[float]:
        return (-output_grad / (math.sin(x) ** 2),)

    def __str__(self) -> str:
        return 'cot'

class Sinh(Operation):
    def forward(self, x: float) -> float:
        return math.sinh(x)

    def backward(self, output_grad: float, x: float) -> Tuple[float]:
        return (output_grad * math.cosh(x),)

    def __str__(self) -> str:
        return 'sinh'

class Cosh(Operation):
    def forward(self, x: float) -> float:
        return math.cosh(x)

    def backward(self, output_grad: float, x: float) -> Tuple[float]:
        return (output_grad * math.sinh(x),)

    def __str__(self) -> str:
        return 'cosh'

class Tanh(Operation):
    def forward(self, x: float) -> float:
        return math.tanh(x)

    def backward(self, output_grad: float, x: float) -> Tuple[float]:
        return (output_grad * (1.0 - math.tanh(x) ** 2),)

    def __str__(self) -> str:
        return 'tanh'

class Coth(Operation):
    def forward(self, x: float) -> float:
        return 1.0 / math.tanh(x)

    def backward(self, output_grad: float, x: float) -> Tuple[float]:
        return (-output_grad / (math.sinh(x) ** 2),)

    def __str__(self) -> str:
        return 'coth' 