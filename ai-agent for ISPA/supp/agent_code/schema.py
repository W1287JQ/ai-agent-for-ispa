from __future__ import annotations

from enum import Enum
from typing import ClassVar, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .step1_properties import parse_smiles


class ApplicationMode(str, Enum):
    leave_on = "leave_on"
    wash_off = "wash_off"


class TestType(str, Enum):
    in_vitro = "in_vitro"
    in_vivo = "in_vivo"


class Occlusion(str, Enum):
    occluded = "occluded"
    unoccluded = "unoccluded"


class DoseRegime(str, Enum):
    finite = "finite"
    infinite = "infinite"


class Intent(str, Enum):
    new_project = "new_project"
    retrieve_project = "retrieve_project"
    modify_case = "modify_case"
    query_report = "query_report"
    other = "other"


class Permeant(BaseModel):
    name: Optional[str] = None
    smiles: Optional[str] = None
    vapour_pressure_pa: Optional[float] = Field(None, ge=0)
    density_kg_m3: Optional[float] = Field(None, gt=0)
    frac_non_ion: Optional[float] = Field(None, ge=0, le=1)
    frac_unbound: Optional[float] = Field(None, gt=0, le=1)

    @field_validator("smiles")
    @classmethod
    def _smiles_parses(cls, v):
        if v is None or not str(v).strip():
            return None
        if parse_smiles(v) is None:
            raise ValueError(f"RDKit cannot parse the SMILES string {v!r}")
        return v.strip()


class Formulation(BaseModel):
    name: Optional[str] = None
    cosolvent: Optional[str] = "None"
    cosolvent_fraction: Optional[float] = Field(None, ge=0, le=1)
    water_fraction: Optional[float] = Field(None, ge=0, le=1)
    oil_fraction: Optional[float] = Field(None, ge=0, le=1)
    surfactant_fraction: Optional[float] = Field(None, ge=0, le=1)
    insoluble_fraction: Optional[float] = Field(None, ge=0, le=1)
    active_fraction: Optional[float] = Field(None, ge=0, le=1)
    pH: Optional[float] = Field(None, ge=0, le=14)
    volatile_solvent: Optional[str] = "Water"

    FRACTION_FIELDS: ClassVar[tuple] = ("cosolvent_fraction", "water_fraction", "oil_fraction",
                       "surfactant_fraction", "insoluble_fraction", "active_fraction")

    def complete(self) -> bool:
        return all(getattr(self, f) is not None for f in self.FRACTION_FIELDS)

    def missing_fractions(self) -> List[str]:
        return [f for f in self.FRACTION_FIELDS if getattr(self, f) is None]

    @model_validator(mode="after")
    def _fractions_sum(self):
        if self.complete():
            tot = sum(getattr(self, f) for f in self.FRACTION_FIELDS)
            if abs(tot - 1.0) > 0.011:
                raise ValueError(
                    f"formulation mass fractions sum to {tot:.3f}; they must sum to 1")
        return self


class Exposure(BaseModel):
    pH: Optional[float] = Field(None, ge=0, le=14)
    dose_ml_cm2: Optional[float] = Field(None, ge=0)
    temperature_c: Optional[float] = Field(None, ge=-273.15)
    duration_h: Optional[float] = Field(None, ge=0)
    removal_time_h: Optional[float] = Field(None, ge=0)
    area_cm2: float = Field(1.0, gt=0)
    relative_humidity: float = Field(33.0, ge=0, le=100)


class Scenario(BaseModel):
    application: Optional[ApplicationMode] = None
    test_type: Optional[TestType] = None
    occlusion: Optional[Occlusion] = None
    dose_regime: Optional[DoseRegime] = None
    evidence: dict = Field(default_factory=dict)


class SimulationCase(BaseModel):
    project_name: Optional[str] = None
    permeant: Permeant = Field(default_factory=Permeant)
    formulation: Formulation = Field(default_factory=Formulation)
    exposure: Exposure = Field(default_factory=Exposure)
    scenario: Scenario = Field(default_factory=Scenario)
    notes: List[str] = Field(default_factory=list)

    REQUIRED: ClassVar[list] = [
        ("permeant.smiles", "the SMILES string of the permeant"),
        ("formulation.name", "a name for the formulation"),
        ("exposure.pH", "the formulation pH"),
        ("exposure.dose_ml_cm2", "the applied dose in mL/cm²"),
        ("exposure.temperature_c", "the temperature in °C"),
        ("exposure.duration_h", "the exposure duration in hours"),
        ("scenario.application", "the application mode (leave-on or wash-off)"),
        ("scenario.test_type", "the test type (in vitro or in vivo)"),
        ("scenario.occlusion", "the occlusion status (occluded or unoccluded)"),
    ]

    def _get(self, path: str):
        obj = self
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def missing(self) -> List[tuple[str, str]]:
        out = [(p, d) for p, d in self.REQUIRED if self._get(p) is None]
        if not self.formulation.complete():
            for f in self.formulation.missing_fractions():
                out.append((f"formulation.{f}", f"the {f.replace('_', ' ')} of the formulation"))
        if self.scenario.application == ApplicationMode.wash_off and self.exposure.removal_time_h is None:
            out.append(("exposure.removal_time_h", "the wash-off (removal) time in hours"))
        return out

    def set_path(self, path: str, value):
        obj = self
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def _restate_rows(self):
        f, e, s = self.formulation, self.exposure, self.scenario
        rows = [
            ("Project", None, self.project_name or "(unnamed)"),
            ("Permeant", "permeant.name", self.permeant.name or "(unnamed)"),
            ("  SMILES", "permeant.smiles", self.permeant.smiles),
            ("Formulation", "formulation.name", f.name),
            ("  cosolvent", "formulation.cosolvent", f"{f.cosolvent} ({f.cosolvent_fraction})"),
            ("  water", "formulation.water_fraction", f.water_fraction),
            ("  oil", "formulation.oil_fraction", f.oil_fraction),
            ("  surfactant", "formulation.surfactant_fraction", f.surfactant_fraction),
            ("  insoluble", "formulation.insoluble_fraction", f.insoluble_fraction),
            ("  active", "formulation.active_fraction", f.active_fraction),
            ("Exposure pH", "exposure.pH", e.pH),
            ("Dose", "exposure.dose_ml_cm2", f"{e.dose_ml_cm2} mL/cm²"),
            ("Temperature", "exposure.temperature_c", f"{e.temperature_c} °C"),
            ("Duration", "exposure.duration_h", f"{e.duration_h} h"),
            ("Application mode", "scenario.application", getattr(s.application, "value", None)),
        ]
        if s.application == ApplicationMode.wash_off:
            rows.append(("  removed at", "exposure.removal_time_h", f"{e.removal_time_h} h"))
        rows += [
            ("Test type", "scenario.test_type", getattr(s.test_type, "value", None)),
            ("Occlusion", "scenario.occlusion", getattr(s.occlusion, "value", None)),
            ("Dose regime", "scenario.dose_regime", getattr(s.dose_regime, "value", None)),
        ]
        return rows

    def restate(self, provenance=None, disagreements=None) -> str:
        dis = {d["field"]: d for d in (disagreements or [])}
        rows = self._restate_rows()
        w = max(len(r[0]) for r in rows)
        out = []
        for label, path, value in rows:
            line = f"{label.ljust(w)} : {value}"
            note = ""
            if provenance is not None and path:
                vs = provenance.get(path)
                if vs is not None:
                    if vs.origin in ("user", "clarified") and vs.snippet:
                        note = f'  ← from your words: "{vs.snippet}"'
                    elif vs.origin == "database":
                        note = f"  ← {vs.detail}"
                    elif vs.origin == "computed":
                        note = f"  ← {vs.detail}"
            if path in dis:
                d = dis[path]
                note += f"   [!] second extraction read {d['secondary']!r}"
            out.append(line + note)
        if dis:
            out.append("")
            out.append(f"[!] {len(dis)} field(s) where two independent extractions disagreed — "
                       f"please check these before confirming.")
        return "\n".join(out)


FINITE_DOSE_ML_CM2 = 0.003
INFINITE_DOSE_ML_CM2 = 0.234
INFINITE_DOSE_THRESHOLD = 0.05


def infer_dose_regime(dose_ml_cm2: Optional[float]) -> Optional[DoseRegime]:
    if dose_ml_cm2 is None:
        return None
    return DoseRegime.infinite if dose_ml_cm2 >= INFINITE_DOSE_THRESHOLD else DoseRegime.finite
