from __future__ import annotations
from typing import Callable, Optional, Tuple, Any, Set, List
from operations import (
    Operation, Add, Mul, Neg, Sub, Div, Pow, 
    Exp, Log, Sin, Cos, Tan, Cot, Sinh, Cosh, Tanh, Coth
)
import graphviz

class Value:
    """A class for automatic differentiation, representing a node in a computation graph."""
    def __init__(
        self,
        value: float,
        prev: Tuple[Value, ...] = (),
        op: Optional[Operation] = None,
        backward: Optional[Callable[[], None]] = None,
        label: Optional[str] = None,
    ) -> None:
        self.value: float = value
        self.grad: float = 0.0
        self.op: Optional[Operation] = op
        self.prev: Tuple[Value, ...] = prev
        self._backward: Callable[[], None] = backward if backward is not None else lambda: None
        self.label: Optional[str] = label

    def __repr__(self) -> str:
        label_str = f", label='{self.label}'" if self.label is not None else ""
        return f"Value(value={self.value}, grad={self.grad}, op={self.op}{label_str})"

    def __add__(self, other: Any) -> Value:
        if not isinstance(other, Value):
            other = Value(float(other))
        op = Add()
        value = op.forward(self.value, other.value)
        return Value(value, prev=(self, other), op=op)

    def __radd__(self, other: Any) -> Value:
        return self.__add__(other)

    def __mul__(self, other: Any) -> Value:
        if not isinstance(other, Value):
            other = Value(float(other))
        op = Mul()
        value = op.forward(self.value, other.value)
        return Value(value, prev=(self, other), op=op)

    def __rmul__(self, other: Any) -> Value:
        return self.__mul__(other)

    def __neg__(self) -> Value:
        op = Neg()
        value = op.forward(self.value)
        return Value(value, prev=(self,), op=op)

    def __sub__(self, other: Any) -> Value:
        if not isinstance(other, Value):
            other = Value(float(other))
        op = Sub()
        value = op.forward(self.value, other.value)
        return Value(value, prev=(self, other), op=op)

    def __rsub__(self, other: Any) -> Value:
        if not isinstance(other, Value):
            other = Value(float(other))
        op = Sub()
        value = op.forward(other.value, self.value)
        return Value(value, prev=(other, self), op=op)

    def __truediv__(self, other: Any) -> Value:
        if not isinstance(other, Value):
            other = Value(float(other))
        op = Div()
        value = op.forward(self.value, other.value)
        return Value(value, prev=(self, other), op=op)

    def __rtruediv__(self, other: Any) -> Value:
        if not isinstance(other, Value):
            other = Value(float(other))
        op = Div()
        value = op.forward(other.value, self.value)
        return Value(value, prev=(other, self), op=op)

    def __pow__(self, other: Any) -> Value:
        if not isinstance(other, Value):
            other = Value(float(other))
        op = Pow()
        value = op.forward(self.value, other.value)
        return Value(value, prev=(self, other), op=op)

    def __rpow__(self, other: Any) -> Value:
        if not isinstance(other, Value):
            other = Value(float(other))
        op = Pow()
        value = op.forward(other.value, self.value)
        return Value(value, prev=(other, self), op=op)

    def exp(self) -> Value:
        """Compute e^x."""
        op = Exp()
        value = op.forward(self.value)
        return Value(value, prev=(self,), op=op)

    def log(self) -> Value:
        """Compute natural logarithm ln(x)."""
        op = Log()
        value = op.forward(self.value)
        return Value(value, prev=(self,), op=op)

    def sin(self) -> Value:
        """Compute sine of x."""
        op = Sin()
        value = op.forward(self.value)
        return Value(value, prev=(self,), op=op)

    def cos(self) -> Value:
        """Compute cosine of x."""
        op = Cos()
        value = op.forward(self.value)
        return Value(value, prev=(self,), op=op)

    def tan(self) -> Value:
        """Compute tangent of x."""
        op = Tan()
        value = op.forward(self.value)
        return Value(value, prev=(self,), op=op)

    def cot(self) -> Value:
        """Compute cotangent of x."""
        op = Cot()
        value = op.forward(self.value)
        return Value(value, prev=(self,), op=op)

    def sinh(self) -> Value:
        """Compute hyperbolic sine of x."""
        op = Sinh()
        value = op.forward(self.value)
        return Value(value, prev=(self,), op=op)

    def cosh(self) -> Value:
        """Compute hyperbolic cosine of x."""
        op = Cosh()
        value = op.forward(self.value)
        return Value(value, prev=(self,), op=op)

    def tanh(self) -> Value:
        """Compute hyperbolic tangent of x."""
        op = Tanh()
        value = op.forward(self.value)
        return Value(value, prev=(self,), op=op)

    def coth(self) -> Value:
        """Compute hyperbolic cotangent of x."""
        op = Coth()
        value = op.forward(self.value)
        return Value(value, prev=(self,), op=op)

    def visualize(self, filename: str = 'graph', format: str = 'svg') -> None:
        """
        Visualize the computation graph using Graphviz and save to a file.
        Args:
            filename: Output file name (without extension).
            format: Output format ('svg', 'png', 'pdf', etc.).
        """
        dot = graphviz.Digraph(format=format)
        dot.attr(bgcolor='white')
        dot.attr('node', shape='record', style='filled', fontname='Helvetica', fontsize='12')

        # Color themes
        value_color = '#8ecae6'
        op_color = '#ffb703'
        grad_color = '#219ebc'
        leaf_color = '#d9ed92'

        # Track visited nodes
        visited = set()

        def add_nodes(v: Value) -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            
            # Use custom label if available, otherwise show value and grad
            if v.label is not None:
                node_label = f"<v> {v.label}|<g> grad={v.grad:.4g}"
            else:
                node_label = f"<v> value={v.value:.4g}|<g> grad={v.grad:.4g}"
            
            color = leaf_color if not v.prev else value_color
            dot.node(str(id(v)), label=node_label, fillcolor=color)
            if v.op is not None:
                op_id = f"op_{id(v)}"
                dot.node(op_id, label=str(v.op), shape='circle', fillcolor=op_color, fontsize='16', fontcolor='black')
                dot.edge(op_id, str(id(v)), color=op_color, penwidth='2')
                for i, parent in enumerate(v.prev):
                    add_nodes(parent)
                    dot.edge(str(id(parent)), op_id, color=grad_color, penwidth='2')
            else:
                for parent in v.prev:
                    add_nodes(parent)

        add_nodes(self)
        dot.render(filename, view=False, cleanup=True)

    def backward(self) -> None:
        """
        Perform reverse-mode autodiff to compute gradients for all nodes in the computation graph.
        Sets self.grad = 1.0 and propagates gradients backward using each operation's backward method.
        """
        # Topological order of nodes (post-order DFS)
        topo: List[Value] = []
        visited: Set[int] = set()

        def build_topo(v: Value) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for parent in v.prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Initialize all gradients to 0
        for node in topo:
            node.grad = 0.0
        self.grad = 1.0  # Seed output gradient

        # Traverse in reverse topological order
        for v in reversed(topo):
            if v.op is not None and v.prev:
                grads = v.op.backward(v.grad, *(parent.value for parent in v.prev))
                for parent, grad in zip(v.prev, grads):
                    parent.grad += grad
            v._backward()  # In case custom backward logic is set 