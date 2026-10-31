"""Shared ratio values and aggregation helpers for evaluation metrics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Ratio:
    """A metric reported as numerator / denominator.

    The project note requires reporting both parts, and a zero denominator means
    "not applicable", never a perfect score.

    Attributes:
        num: Metric numerator, such as correctly handled cases.
        den: Metric denominator, such as applicable cases.
    """

    num: float
    den: float

    @property
    def value(self) -> float | None:
        """Return the quotient, or ``None`` when the denominator is zero.

        Returns:
            Floating-point ratio, or ``None`` to represent a not-applicable
            metric rather than a perfect score.
        """
        return None if self.den == 0 else self.num / self.den

    def fmt(self, digits: int = 3) -> str:
        """Format the ratio with its numerator and denominator.

        Args:
            digits: Decimal places for the quotient and any non-integral
                numerator or denominator.

        Returns:
            ``value (num/den)`` text, or ``N/A (0/0)`` when undefined.
        """
        if self.value is None:
            return "N/A (0/0)"
        num = int(self.num) if float(self.num).is_integer() else round(self.num, digits)
        den = int(self.den) if float(self.den).is_integer() else round(self.den, digits)
        return f"{self.value:.{digits}f} ({num}/{den})"

    def to_dict(self) -> dict:
        """Return the machine-readable ratio components.

        Returns:
            Mapping with ``value`` (possibly ``None``), ``num``, and ``den``.
        """
        return {"value": self.value, "num": self.num, "den": self.den}


def mean_ratio(values: list[float]) -> Ratio:
    """Represent the arithmetic mean of per-query values as a ratio.

    Args:
        values: One score per applicable query.

    Returns:
        A ``Ratio`` with the score sum as numerator and count as denominator.
        An empty list produces a zero-denominator ratio.
    """
    return Ratio(sum(values), len(values))
