"""Calculator tool implementation.

This tool evaluates a constrained math expression with an AST whitelist rather
than using Python's unrestricted `eval`.
"""

from __future__ import annotations

import ast
import math

from tools.base import ToolResult


class SafeMathEvaluator(ast.NodeVisitor):
    """Very small AST-based evaluator for calculator expressions."""

    ALLOWED_FUNCTIONS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "fabs": math.fabs,
        "floor": math.floor,
        "ceil": math.ceil,
        "round": round,
        "abs": abs,
    }
    ALLOWED_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }
    ALLOWED_BINOPS = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.FloorDiv: lambda left, right: left // right,
        ast.Mod: lambda left, right: left % right,
        ast.Pow: lambda left, right: left**right,
    }
    ALLOWED_UNARYOPS = {
        ast.UAdd: lambda value: +value,
        ast.USub: lambda value: -value,
    }

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed.")

    def visit_Name(self, node: ast.Name) -> float:
        if node.id not in self.ALLOWED_CONSTANTS:
            raise ValueError(f"Unknown constant: {node.id}")
        return float(self.ALLOWED_CONSTANTS[node.id])

    def visit_BinOp(self, node: ast.BinOp) -> float:
        operator_type = type(node.op)
        if operator_type not in self.ALLOWED_BINOPS:
            raise ValueError(f"Unsupported operator: {operator_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return float(self.ALLOWED_BINOPS[operator_type](left, right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operator_type = type(node.op)
        if operator_type not in self.ALLOWED_UNARYOPS:
            raise ValueError(f"Unsupported unary operator: {operator_type.__name__}")
        operand = self.visit(node.operand)
        return float(self.ALLOWED_UNARYOPS[operator_type](operand))

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed.")
        function_name = node.func.id
        if function_name not in self.ALLOWED_FUNCTIONS:
            raise ValueError(f"Unsupported function: {function_name}")
        args = [self.visit(arg) for arg in node.args]
        return float(self.ALLOWED_FUNCTIONS[function_name](*args))

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def run_calculator(expression: str) -> ToolResult:
    """Evaluate one arithmetic expression with a strict AST whitelist."""

    try:
        tree = ast.parse(expression, mode="eval")
        value = SafeMathEvaluator().visit(tree)
        return ToolResult(
            tool_name="calculator",
            tool_input=expression,
            success=True,
            output_text=str(value),
        )
    except Exception as exc:
        return ToolResult(
            tool_name="calculator",
            tool_input=expression,
            success=False,
            output_text="",
            error_text=f"{type(exc).__name__}: {exc}",
        )
