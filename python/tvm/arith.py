"""Arithmetic data structure and utility"""
from __future__ import absolute_import as _abs

from ._ffi.node import NodeBase, register_node
from ._ffi.function import _init_api
from . import _api_internal

class IntSet(NodeBase):
    """Represent a set of integer in one dimension."""
    def is_nothing(self):
        """Whether the set represent nothing"""
        return _api_internal._IntSetIsNothing(self)

    def is_everything(self):
        """Whether the set represent everything"""
        return _api_internal._IntSetIsEverything(self)


@register_node
class IntervalSet(IntSet):
    """Represent set of continuous interval"""
    def min(self):
        """get the minimum value"""
        return _api_internal._IntervalSetGetMin(self)

    def max(self):
        """get the maximum value"""
        return _api_internal._IntervalSetGetMax(self)


@register_node
class StrideSet(IntSet):
    """Represent set of strided integers"""


@register_node("arith.ModularSet")
class ModularSet(NodeBase):
    """Represent range of (coeff * x + base) for x in Z """
    def __init__(self, coeff, base):
        self.__init_handle_by_constructor__(
            _make_ModularSet, coeff, base)


@register_node("arith.ConstIntBound")
class ConstIntBound(NodeBase):
    """Represent constant integer bound

    Parameters
    ----------
    min_value : int
        The minimum value of the bound.

    max_value : int
        The maximum value of the bound.
    """
    POS_INF = (1 << 63) - 1
    NEG_INF = -POS_INF

    def __init__(self, min_value, max_value):
        self.__init_handle_by_constructor__(
            _make_ConstIntBound, min_value, max_value)


class ConstraintScope:
    """Constraint scope.

    Parameters
    ----------
    fenter : function
        A function that will be called to create an enter context.

    Note
    ----
    Do not create object directly, use Analyzer.constraint_scope
    """
    def __init__(self, fenter):
        self._fenter = fenter
        self._fexit = None

    def __enter__(self):
        self._fexit = self._fenter()

    def __exit__(self, ptype, value, trace):
        self._fexit()


class Analyzer:
    """Integer arithmetic analyzer

    This is a stateful analyzer class that can
    be used to perform various symbolic integer analysis.
    """
    def __init__(self):
        _mod = _CreateAnalyzer()
        self._const_int_bound = _mod("const_int_bound")
        self._const_int_bound_update = _mod("const_int_bound_update")
        self._bind = _mod("bind")
        self._modular_set = _mod("modular_set")
        self._rewrite_simplify = _mod("rewrite_simplify")
        self._canonical_simplify = _mod("canonical_simplify")
        self._enter_constraint_context = _mod("enter_constraint_context")

    def const_int_bound(self, expr):
        """Find constant integer bound for expr.

        Parameters
        ----------
        expr : tvm.Expr
            The expression.

        Returns
        -------
        bound : ConstIntBound
            The result bound
        """
        return self._const_int_bound(expr)

    def modular_set(self, expr):
        """Find a modular set that expr belongs to.

        Parameters
        ----------
        expr : tvm.Expr
            The expression.

        Returns
        -------
        result : ModularSet
            The result.
        """
        return self._modular_set(expr)

    def rewrite_simplify(self, expr):
        """Simplify expression via rewriting rules.

        Parameters
        ----------
        expr : tvm.Expr
            The expression.

        Returns
        -------
        result : Expr
            The result.
        """
        return self._rewrite_simplify(expr)

    def canonical_simplify(self, expr):
        """Simplify expression via canonicalization.

        Parameters
        ----------
        expr : tvm.Expr
            The expression.

        Returns
        -------
        result : Expr
            The result.
        """
        return self._canonical_simplify(expr)

    def bind(self, var, expr):
        """Bind a variable to the expression.

        Parameters
        ----------
        var : tvm.Var
            The variable.

        expr : tvm.Expr
            The expression.
        """
        return self._bind(var, expr)

    def constraint_scope(self, constraint):
        """Create a constraint scope.

        Parameters
        ----------
        constraint : tvm.Expr
            The constraint expression.

        returns
        -------
        scope : ConstraintScope
            The constraint scope

        Examples
        --------
        .. code-block:: python

          x = tvm.var("x")
          analyzer = tvm.arith.Analyzer()
          with analzyer.constraint_scope(x % 3 == 0):
              # constraint in effect
              assert analyzer.modular_set(x).coeff == 3
          # constraint no longer in effect
          assert analyzer.modular_set(x).coeff != 3
        """
        def _fenter():
            return self._enter_constraint_context(constraint)
        return ConstraintScope(_fenter)

    def update(self, var, info, override=False):
        """Update infomation about var

        Parameters
        ----------
        var : tvm.Var
            The variable.

        info : tvm.NodeBase
            Related information.

        override : bool
            Whether allow override.
        """
        if isinstance(info, ConstIntBound):
            self._const_int_bound_update(var, info, override)
        else:
            raise TypeError(
                "Do not know how to handle type {}".format(type(info)))


_init_api("tvm.arith")
