from dataclasses import dataclass


@dataclass(frozen=True)
class Ratio:
    """A metric reported as numerator / denominator.

    The project note requires reporting both parts, and a zero denominator means
    "not applicable", never a perfect score.
    """

    num: float
    den: float

    @property
    def value(self) -> float | None:
        return None if self.den == 0 else self.num / self.den


    def fmt(self, digits: int = 3) -> str:
        if self.value is None:
            return "N/A (0/0)"
        num = int(self.num) if float(self.num).is_integer() else round(self.num, digits)
        den = int(self.den) if float(self.den).is_integer() else round(self.den, digits)
        return f"{self.value:.{digits}f} ({num}/{den})"

    def to_dict(self) -> dict:
        return {"value": self.value, "num": self.num, "den": self.den}


def mean_ratio(values: list[float]) -> Ratio:
    """Mean of per-query scores, kept as sum / count so N/A survives aggregation."""
    return Ratio(sum(values), len(values))


