from collections.abc import Generator
import pulp


class LpProblem:
    """Outer class for pulp.LpProblem, with syntactical support for constraint addition with `LpArray` \
    and `LpMatrix` objects.
    """

    def __init__(self, name="NoName", sense=pulp.const.LpMaximize):
        self.model = pulp.LpProblem(name, sense)

    def __iadd__(self, other: Generator | pulp.pulp.LpConstraint |
                 pulp.pulp.LpAffineExpression | pulp.pulp.LpVariable = None) -> None:
        """pulp.LpProblem-style constraint & objective addition
        Args:
            other (Generator | pulp.pulp.LpConstraint | pulp.pulp.LpAffineExpression | pulp.pulp.LpVariable,\
                optional): LP object or generator of LP objects. Defaults to None.
        Raises:
            TypeError: Type not appropriate for constraint or objective.
        """
        match other:
            case Generator() as constraints:
                for const in constraints:
                    self.model.addConstraint(const)

            case pulp.pulp.LpConstraint() | pulp.pulp.LpAffineExpression() | pulp.pulp.LpVariable() as const:
                self.model.addConstraint(const)

            case _:
                raise TypeError(
                    f"Type {type(other).__name__} cannot be applied to LP problem")

        return self

    def solve(self, time_limit=10000, optimizer=pulp.GUROBI, message=True) -> None:
        """Solve LpModel"""
        return self.model.solve(optimizer(timeLimit=time_limit, msg=message))

    def __str__(self) -> str:
        return str(self.model)
