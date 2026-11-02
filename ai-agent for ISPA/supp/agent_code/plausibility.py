from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Range:
    field: str
    label: str
    unit: str
    low: Optional[float]
    high: Optional[float]
    reason: str

    def outside(self, v: float) -> bool:
        return (self.low is not None and v < self.low) or \
               (self.high is not None and v > self.high)

    def describe(self) -> str:
        if self.low is not None and self.high is not None:
            return f"{self.low:g}-{self.high:g} {self.unit}".strip()
        if self.high is not None:
            return f"up to {self.high:g} {self.unit}".strip()
        return f"at least {self.low:g} {self.unit}".strip()


RANGES = [
    Range("exposure.pH", "formulation pH", "", 3.0, 9.0,
          "outside pH 3-9 the barrier itself is altered by the vehicle, and the "
          "Henderson-Hasselbalch treatment of a single ionisation site no longer "
          "describes the permeant reliably"),
    Range("exposure.temperature_c", "temperature", "°C", 20.0, 40.0,
          "the diffusivity correlations and the evaporation sub-module are fitted "
          "around skin-surface temperature (32 °C in OECD TG 428); outside 20-40 °C "
          "they extrapolate"),
    Range("exposure.dose_ml_cm2", "applied dose", "mL/cm²", 0.001, 0.5,
          "below 0.001 mL/cm² the applied film is thinner than the surface roughness "
          "the model assumes; above 0.5 mL/cm² it is well beyond the near-infinite "
          "dose (0.234 mL/cm²) used in the validation cases"),
    Range("exposure.duration_h", "exposure duration", "h", None, 72.0,
          "beyond 72 h neither skin viability in an in vitro test nor the assumption "
          "of an intact barrier is maintained"),
]

FRACTION_SUM_TOLERANCE = 0.001


@dataclass
class Note:
    field: str
    value: float
    message: str
    question: str


def check(case) -> List[Note]:
    notes: List[Note] = []
    for r in RANGES:
        v = _get(case, r.field)
        if v is None or not r.outside(float(v)):
            continue
        notes.append(Note(
            field=r.field, value=float(v),
            message=(f"{r.label} {v:g} {r.unit}".rstrip() +
                     f" is outside the range the model has been exercised over "
                     f"({r.describe()}): {r.reason}"),
            question=(f"the {r.label} is {v:g} {r.unit}".rstrip() +
                      f", outside the usual {r.describe()} - is that intended?")))

    f = case.formulation
    vals = [getattr(f, k, None) for k in f.FRACTION_FIELDS]
    if all(v is not None for v in vals):
        tot = sum(vals)
        if abs(tot - 1.0) > FRACTION_SUM_TOLERANCE:
            notes.append(Note(
                field="formulation.fractions", value=tot,
                message=(f"the formulation mass fractions sum to {tot:.4f} rather than 1; "
                         f"the composition is most likely missing a component, and every "
                         f"phase fraction in Eq. 2 is computed from these values"),
                question=(f"the formulation mass fractions sum to {tot:.4f} rather than 1 "
                          f"- is a component missing?")))
    return notes


def _get(case, path: str):
    obj = case
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj
