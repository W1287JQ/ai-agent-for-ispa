from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import applicability
from . import plausibility
from . import step1_properties as step1
from . import step2_qspr as step2
from . import step3_pbpk as step3
from .database import Database
from .llm import RuleBackend, get_backend
from .provenance import (Provenance, USER, CLARIFIED, DATABASE,
                         LITERATURE, TRACED, COMPUTED, DEFAULT, FITTED)
from .schema import (INFINITE_DOSE_THRESHOLD, ApplicationMode, DoseRegime, Exposure,
                     Formulation, Intent, Occlusion,
                     Permeant, Scenario, SimulationCase, TestType, infer_dose_regime,
                     FINITE_DOSE_ML_CM2, INFINITE_DOSE_ML_CM2)


CROSS_CHECK_FIELDS = {
    "permeant_name": "permeant.name", "formulation_name": "formulation.name",
    "pH": "exposure.pH", "dose_ml_cm2": "exposure.dose_ml_cm2",
    "temperature_c": "exposure.temperature_c", "duration_h": "exposure.duration_h",
    "removal_time_h": "exposure.removal_time_h", "application": "scenario.application",
    "test_type": "scenario.test_type", "occlusion": "scenario.occlusion",
}


def _norm_name(x: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", " ", str(x).lower())
    t = re.sub(r"^\s*(a|an|the)\s+", "", t)
    return re.sub(r"\s+", " ", t).strip()


_TIME_STATED = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:-\s*)?(?:h|hr|hrs|hour|hours|min|mins|minute|minutes|"
    r"s|sec|secs|second|seconds|d|day|days)\b", re.I)

_REMOVAL_STATED = re.compile(
    r"\b(?:wash(?:ed|ing|es)?\s+(?:off|out|away|with|the)|washed|rinse[sd]?|rinsing|"
    r"wipe[sd]?|wiping|remove[sd]?|removal|showered?|bathe[sd]?|towel(?:led|ed)?)\b",
    re.I)

_RETENTION_STATED = re.compile(
    r"\b(?:leave[- ]on|left\s+(?:\w+\s+){0,2}on\b|left\s+in\s+place|stays?\s+in\s+place|"
    r"not\s+(?:remove[sd]?|rinse[sd]?|wash(?:ed|es)?|wipe[sd]?|topped)|"
    r"without\s+(?:being\s+)?"
    r"(?:removing|removal|rinsing|washing|wiping)|no\s+(?:rinsing|washing|wash|rinse)\b|"
    r"allowed\s+to\s+remain|remains?\s+on\s+the\s+skin|remained\s+on\s+the\s+skin|"
    r"worn\s+for|kept\s+on|retained\s+on)\b", re.I)


def application_needs_confirmation(text: str) -> bool:
    if not text or not _TIME_STATED.search(text):
        return False
    if _REMOVAL_STATED.search(text) or _RETENTION_STATED.search(text):
        return False
    return True


APPLICATION_QUESTION = (
    "the application mode: the request gives a contact time but does not say whether "
    "the product stays on the skin or is removed - please state either leave-on, or "
    "wash-off together with the removal time")


def _values_differ(a, b, rel: float = 0.01) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a == 0 or b == 0:
            return abs(a - b) > 1e-9
        return abs(a - b) / max(abs(a), abs(b)) > rel
    return _norm_name(a) != _norm_name(b)


@dataclass
class CaseResult:
    case: SimulationCase
    properties: Dict[str, Any]
    qspr: Dict[str, Any]
    pbpk: step3.PBPKResult
    log: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    extraction_disagreements: List[dict] = field(default_factory=list)
    cross_check_status: str = "not run"
    domain: Optional[applicability.DomainReport] = None

    @property
    def summary(self):
        return self.pbpk.summary


class ClarificationNeeded(Exception):
    def __init__(self, question: str, missing):
        super().__init__(question)
        self.question = question
        self.missing = missing


class Agent:
    KOW_CONVENTIONS = ("logd8", "dow_ph", "dow_74", "logp_neutral")

    def __init__(self, db: Optional[Database] = None, backend: str = "auto",
                 ask: Optional[Callable[[str], str]] = None, verbose: bool = False,
                 kow_convention: str = "logd8", skin_preset=None,
                 cross_check: bool = True, domain_check: str = "ask",
                 n_exponent: float = step2.N_EXPONENT):
        if kow_convention not in self.KOW_CONVENTIONS:
            raise ValueError(f"kow_convention must be one of {self.KOW_CONVENTIONS}")
        self.kow_convention = kow_convention
        self.n_exponent = float(n_exponent)
        self.db = db or Database()
        self.llm = get_backend(backend)
        self.ask = ask
        self.verbose = verbose
        self.log: List[str] = []
        self.prov = Provenance()
        self.skin_preset = skin_preset
        self.cross_check = cross_check
        self.parse_warnings: List[str] = []
        self.application_unconfirmed = False
        self.application_span = ""
        self.plausibility_notes: List = []
        self.disagreements: List[dict] = []
        self.cross_check_status = "not run"
        self._second = self._independent_backend() if cross_check else None
        if domain_check not in ("ask", "warn", "off"):
            raise ValueError('domain_check must be "ask", "warn" or "off"')
        self.domain_check = domain_check

    def _independent_backend(self):
        if isinstance(self.llm, RuleBackend):
            self.cross_check_status = ("unavailable - the only parser present is the rule backend, "
                                       "so a second pass would not be independent")
            return None
        self.cross_check_status = "LLM vs rule-based extraction"
        return RuleBackend()

    def _cross_check(self, text: str, raw: dict) -> List[dict]:
        if self._second is None:
            return []
        if type(self.llm) is type(self._second):
            self.cross_check_status = ("unavailable - both readers would be the same parser")
            return []
        try:
            other = self._second.extract(text)
        except Exception as exc:
            self.cross_check_status = f"failed: {exc}"
            return []
        out = []
        for key, path in CROSS_CHECK_FIELDS.items():
            a, b = raw.get(key), other.get(key)
            if _values_differ(a, b) or (a is None and b is not None):
                out.append({"field": path, "key": key, "primary": a, "secondary": b})
        return out

    def missing_with_reasons(self, case) -> List[tuple]:
        missing = case.missing()
        if not self.application_unconfirmed:
            return missing
        q = APPLICATION_QUESTION
        if self.application_span:
            q += (f' (the phrase "{self.application_span}" was read as describing '
                  f'this, but it is not an explicit statement about removal)')
        return [(p, q if p == "scenario.application" else d) for p, d in missing]

    def _say(self, msg):
        self.log.append(msg)
        if self.verbose:
            print(msg, flush=True)

    def parse(self, text: str, case: Optional[SimulationCase] = None) -> SimulationCase:
        raw = self.llm.extract(text)
        self._say(f"[intent parser] intent={raw.get('intent')}")
        follow_up = case is not None
        case = case or SimulationCase()
        if not follow_up:
            self.prov = Provenance()
            self.parse_warnings = []
            self.application_unconfirmed = False
            self.application_span = ""
        for problem in raw.get("unit_problems") or []:
            self.parse_warnings.append(f"unit conversion: {problem}")
            self._say(f"[units] {problem}")
        new_dis = self._cross_check(text, raw) if self.cross_check else []
        if not follow_up:
            self.disagreements = []
        resupplied = {k for k, v in raw.items() if v is not None}
        self.disagreements = [d for d in self.disagreements
                              if d["key"] not in resupplied] + new_dis
        if new_dis:
            self._say("[cross-check] independent second extraction disagrees on: "
                      + ", ".join(f"{d['field']} ({d['primary']!r} vs {d['secondary']!r})"
                                  for d in new_dis))
        ev = raw.get("evidence") or {}
        origin = CLARIFIED if follow_up else USER
        g = lambda k: raw.get(k)

        _EV_BASE = {"dose_ml_cm2": "dose", "temperature_c": "temperature",
                    "duration_h": "duration", "removal_time_h": "removal_time",
                    "area_cm2": "area"}

        def _p(path, key, value):
            self.prov.record(path, value, origin,
                             "answered in the clarification loop" if follow_up
                             else "stated in the request",
                             snippet=ev.get(key) or ev.get(_EV_BASE.get(key, ""), ""))

        if g("project_name"):
            case.project_name = g("project_name")
        if g("permeant_name"):
            case.permeant.name = g("permeant_name"); _p("permeant.name", "permeant_name", g("permeant_name"))
        if g("permeant_smiles"):
            case.permeant.smiles = g("permeant_smiles"); _p("permeant.smiles", "permeant_smiles", g("permeant_smiles"))
        if g("formulation_name"):
            case.formulation.name = g("formulation_name")
            _p("formulation.name", "formulation_name", g("formulation_name"))
        for k in ("cosolvent", "cosolvent_fraction", "water_fraction", "oil_fraction",
                  "surfactant_fraction", "insoluble_fraction", "active_fraction"):
            if g(k) is not None:
                setattr(case.formulation, k, g(k)); _p(f"formulation.{k}", k, g(k))
        for k in ("pH", "dose_ml_cm2", "temperature_c", "duration_h", "removal_time_h", "area_cm2"):
            if g(k) is not None:
                setattr(case.exposure, k, g(k)); _p(f"exposure.{k}", k, g(k))
        if g("application"):
            case.scenario.application = ApplicationMode(g("application"))
            _p("scenario.application", "application", g("application"))
        if g("test_type"):
            case.scenario.test_type = TestType(g("test_type"))
            _p("scenario.test_type", "test_type", g("test_type"))
        if g("occlusion"):
            case.scenario.occlusion = Occlusion(g("occlusion"))
            _p("scenario.occlusion", "occlusion", g("occlusion"))
        if application_needs_confirmation(text):
            guessed = getattr(case.scenario.application, "value", None)
            self.application_span = (ev.get("application") or "").strip()
            if guessed is not None:
                self.parse_warnings.append(
                    f"the request times the exposure but does not say whether the product "
                    f"is removed; the extracted application mode ({guessed}) was withdrawn "
                    f"and is being confirmed instead")
                self._say(f"[guardrail] application={guessed} withdrawn: removal not stated")
            case.scenario.application = None
            case.scenario.evidence.pop("application", None)
            self.application_unconfirmed = True

        if g("dose_regime"):
            case.scenario.dose_regime = DoseRegime(g("dose_regime"))
            _p("scenario.dose_regime", "dose_regime", g("dose_regime"))
        if g("evidence"):
            case.scenario.evidence.update(g("evidence"))

        self._resolve_from_database(case)
        self._settle_dose_regime(case)
        return case

    def _settle_dose_regime(self, case: SimulationCase):
        dose = case.exposure.dose_ml_cm2
        implied = infer_dose_regime(dose) if dose is not None else None
        stated = case.scenario.dose_regime

        if stated is not None:
            if implied is not None and implied != stated:
                self.parse_warnings.append(
                    f"the text describes {'an' if stated.value == 'infinite' else 'a'} "
                    f"{stated.value} dose but the applied dose "
                    f"({dose:g} mL/cm²) is on the {implied.value} side of the "
                    f"{INFINITE_DOSE_THRESHOLD} mL/cm² threshold; the stated regime was used")
            return

        if implied is not None:
            case.scenario.dose_regime = implied
            self.prov.record("scenario.dose_regime", implied.value, COMPUTED,
                             "derived from the applied dose",
                             snippet=f"dose {dose} mL/cm² "
                                     f"{'>=' if implied.value == 'infinite' else '<'} "
                                     f"{INFINITE_DOSE_THRESHOLD} mL/cm² threshold")

    def _resolve_from_database(self, case: SimulationCase):
        f = case.formulation
        if f.name:
            rec = self.db.get_formulation(f.name)
            if rec:
                self._say(f"[database] formulation '{rec['name']}' retrieved")
                f.name = rec["name"]
                for src, dst in [("cosolvent", "cosolvent"), ("cosolvent_fraction", "cosolvent_fraction"),
                                 ("water_fraction", "water_fraction"), ("oil_fraction", "oil_fraction"),
                                 ("surfactant_fraction", "surfactant_fraction"),
                                 ("insoluble_fraction", "insoluble_fraction"),
                                 ("active_fraction", "active_fraction"),
                                 ("volatile_solvent", "volatile_solvent")]:
                    if getattr(f, dst) in (None, "None") and rec.get(src) is not None:
                        setattr(f, dst, rec[src])
                        self.prov.record(f"formulation.{dst}", rec[src], DATABASE,
                                         f"formulation record '{rec['name']}'",
                                         snippet=f"{src} = {rec[src]}", reference=rec.get("source", ""))
                if case.exposure.pH is None and rec.get("pH") is not None:
                    case.exposure.pH = rec["pH"]
                    self.prov.record("exposure.pH", rec["pH"], DATABASE,
                                     f"formulation record '{rec['name']}'",
                                     snippet=f"pH = {rec['pH']}", reference=rec.get("source", ""))
        p = case.permeant
        if p.name and not p.smiles:
            rec = self.db.get_permeant(p.name)
            if rec:
                self._say(f"[database] permeant '{rec['name']}' retrieved")
                p.name = rec["name"]; p.smiles = rec["smiles"]
                self.prov.record("permeant.smiles", rec["smiles"], DATABASE,
                                 f"permeant record '{rec['name']}'",
                                 snippet=rec["smiles"], reference=rec.get("source", ""))
                for attr, col in (("vapour_pressure_pa", "vapour_pressure_pa"),
                                  ("density_kg_m3", "density_kg_m3"),
                                  ("frac_non_ion", "frac_non_ion"), ("frac_unbound", "frac_unbound")):
                    if getattr(p, attr) is None and rec.get(col) is not None:
                        setattr(p, attr, rec[col])
                        self.prov.record(f"permeant.{attr}", rec[col], DATABASE,
                                         f"permeant record '{rec['name']}'",
                                         snippet=f"{col} = {rec[col]}", reference=rec.get("source", ""))
        elif p.smiles and not p.name:
            rec = self.db.find_by_smiles(p.smiles)
            if rec:
                p.name = rec["name"]

    def clarify(self, case: SimulationCase, max_rounds: int = 6) -> SimulationCase:
        for _ in range(max_rounds):
            missing = case.missing()
            if not missing:
                return case
            missing = self.missing_with_reasons(case)
            sugg = {}
            for path, _desc in missing:
                if path == "permeant.smiles" and case.permeant.name:
                    s = self.db.suggest("permeant", case.permeant.name)
                    if s:
                        sugg[path] = s
                if path == "formulation.name" and case.formulation.name:
                    s = self.db.suggest("formulation", case.formulation.name)
                    if s:
                        sugg[path] = s
                if path == "exposure.dose_ml_cm2":
                    sugg[path] = [f"a finite film (~{FINITE_DOSE_ML_CM2} mL/cm²)",
                                  f"a near-infinite dose (~{INFINITE_DOSE_ML_CM2} mL/cm²)"]
            question = self.llm.clarify(missing, sugg)
            self._say(f"[clarification] {question}")
            if self.ask is None:
                raise ClarificationNeeded(question, missing)
            reply = self.ask(question)
            self.parse(reply, case)
        raise ClarificationNeeded("Still missing: " + ", ".join(d for _p, d in case.missing()),
                                  case.missing())

    def run_pipeline(self, case: SimulationCase, profile_times_h=None) -> CaseResult:
        missing = case.missing()
        if missing:
            raise ValueError("case is incomplete: " + ", ".join(d for _p, d in missing))
        warnings: List[str] = list(self.parse_warnings)

        self.plausibility_notes = plausibility.check(case)
        for note in self.plausibility_notes:
            warnings.append(note.message)
            self._say(f"[plausibility] {note.message}")
            self.prov.record(f"plausibility.{note.field}", note.value, COMPUTED,
                             "outside the advisory range of Section 2.1.2",
                             snippet=note.question)
        if self.plausibility_notes and self.ask is not None:
            reply = self.ask("Before I run this: "
                             + "; ".join(n.question for n in self.plausibility_notes)
                             + "  (reply 'continue' to run it as stated)")
            warnings.append(f"the user was asked to confirm {len(self.plausibility_notes)} "
                            f"out-of-range value(s) and replied: {reply.strip()!r}")

        rec = self.db.get_permeant(case.permeant.name) if case.permeant.name else None
        props = step1.estimate(case.permeant.smiles,
                               pka_override=(rec or {}).get("pka_override"),
                               logP_override=(rec or {}).get("logp_override"),
                               logS0_override=(rec or {}).get("logs0_override"),
                               property_source=(rec or {}).get("property_source") or "")
        warnings += props.warnings

        domain = None
        if self.domain_check != "off":
            try:
                domain = applicability.check(case.permeant.smiles, props.sites)
            except Exception as exc:
                self._say(f"[domain] check unavailable: {exc}")
            else:
                self._say("[domain] " + domain.render().replace("\n", "\n          "))
                if domain.verdict != applicability.IN_DOMAIN:
                    warnings.append(
                        f"applicability domain: the permeant is {domain.verdict} for "
                        + "; ".join(f"{t.tool} ({t.reason})" for t in domain.flagged()))
                if domain.verdict == applicability.OUT_OF_DOMAIN and self.domain_check == "ask":
                    if self.ask is None:
                        raise ClarificationNeeded(domain.question(),
                                                  [("permeant.smiles", "a permeant inside the "
                                                    "applicability domain, or measured properties")])
                    reply = self.ask(domain.question())
                    if not reply.strip().lower().startswith(("c", "cont", "y", "继续", "是")):
                        raise ClarificationNeeded(
                            "Supply measured logP / intrinsic solubility / pKa for this permeant "
                            "(they can be stored with Database.set_property_override) and re-run.",
                            [("permeant.smiles", "measured properties for an out-of-domain permeant")])
                    warnings.append("the user chose to continue with estimated properties for an "
                                    "out-of-domain permeant")
        pH = case.exposure.pH
        logSw_pH, logSw7, logD_pH = props.logSw(pH), props.logSw(7.0), props.logD(pH)
        pr = self.prov
        pr.record("step1.MW", round(props.mw, 3), COMPUTED, "RDKit molecular weight from the SMILES",
                  snippet=case.permeant.smiles or "")
        _ov = (rec or {}).get("property_source") or ""
        pr.record("step1.logP", round(props.logP, 4),
                  LITERATURE if (rec or {}).get("logp_override") is not None else COMPUTED,
                  "curated measured value" if (rec or {}).get("logp_override") is not None
                  else "RDKit Wildman-Crippen atom contributions",
                  snippet="Crippen MolLogP(mol)" if (rec or {}).get("logp_override") is None else "",
                  reference=_ov or "Wildman & Crippen 1999")
        pr.record("step1.logS0", round(props.logS0, 4),
                  LITERATURE if (rec or {}).get("logs0_override") is not None else COMPUTED,
                  "curated measured value" if (rec or {}).get("logs0_override") is not None
                  else "ESOL regression (Eq. 1)",
                  snippet="" if (rec or {}).get("logs0_override") is not None
                  else "log S0 = 0.16 - 0.63*logP - 0.0062*MW + 0.066*RB - 0.74*AP",
                  reference=_ov or "Delaney 2004")
        for i, site in enumerate(props.sites):
            pr.record(f"step1.pKa[{i}]", round(site.pka, 3),
                      LITERATURE if site.method == "database override" else COMPUTED,
                      f"{site.type} site, {site.label}", snippet=site.method,
                      reference=props.pka_method)
        for nm, val in (("logSw_pH", logSw_pH), ("logSw7", logSw7), ("logD_pH", logD_pH)):
            pr.record(f"step1.{nm}", round(val, 4), COMPUTED,
                      "Henderson-Hasselbalch from logP / logS0 / pKa",
                      snippet=f"ionisation factor at pH {pH if 'pH' in nm else 7.0}")
        self._say(f"[step 1] MW={props.mw:.2f}  logP={props.logP:.3f}  logS0={props.logS0:.3f}  "
                  f"pKa={[round(s.pka,2) for s in props.sites]} ({props.pka_method})")
        self._say(f"[step 1] at pH {pH}: logSw={logSw_pH:.3f}  logD={logD_pH:.3f}  (logSw7={logSw7:.3f})")

        cos = self.db.get_cosolvent(case.formulation.cosolvent)
        if cos is None:
            s = self.db.suggest("cosolvent", case.formulation.cosolvent or "")
            raise ClarificationNeeded(
                f"I do not have cosolvency parameters (M, N) for '{case.formulation.cosolvent}'."
                + (f" Did you mean: {', '.join(s)}?" if s else
                   " Please give the M and N values, or name a cosolvent from the database."),
                [("formulation.cosolvent", "the cosolvent M/N parameters")])
        f = case.formulation
        q = step2.compute(logSw_pH=logSw_pH, logSw7=logSw7, logD_pH=logD_pH,
                          w_cos=f.cosolvent_fraction, w_water=f.water_fraction,
                          w_oil=f.oil_fraction, w_surf=f.surfactant_fraction,
                          M=cos["M"], N=cos["N"], mw_cos=cos.get("MW"),
                          n_exp=self.n_exponent)
        if self.n_exponent != step2.N_EXPONENT:
            pr.record("step2.n_exponent", self.n_exponent, USER,
                      "Eq. 2 oil-term exponent set by the caller",
                      snippet=f"n = {self.n_exponent} (published value "
                              f"{step2.N_EXPONENT})")
        if cos.get("M") or cos.get("N"):
            pr.record("step2.M", cos["M"], DATABASE, f"cosolvent record '{cos['name']}'",
                      snippet=f"M = {cos['M']}", reference=cos.get("source", ""))
            pr.record("step2.N", cos["N"], DATABASE, f"cosolvent record '{cos['name']}'",
                      snippet=f"N = {cos['N']}", reference=cos.get("source", ""))
        pr.record("step2.sigma", round(q.sigma, 4), COMPUTED, "Eq. 4",
                  snippet="sigma = M*logD + N", reference="Zhang et al. 2025")
        pr.record("step2.logSc", round(q.logSc, 4), COMPUTED, "Eq. 3 (Jouyban log-linear)",
                  snippet="log Sc = log Sw(pH) + sigma*f_cosolvent", reference="Zhang et al. 2025")
        pr.record("step2.Pvw", round(q.Pvw, 5), COMPUTED, "Eq. 2",
                  snippet="Pv/w = fc*(Sc/Sw7) + fo*Dow^0.8", reference="Zhang et al. 2025")
        self._say(f"[step 2] fc={q.f_continuous:.3f} fo={q.f_oil:.3f} sigma={q.sigma:.3f}  "
                  f"-> Pv/w={q.Pvw:.4f}  logSf={q.logSf:.3f}")

        Sf_kg_m3 = (10 ** q.logSf) * props.mw
        applied_conc_kg_m3 = f.active_fraction * 1000.0
        if Sf_kg_m3 < applied_conc_kg_m3:
            warnings.append(
                f"the predicted formulation solubility ({Sf_kg_m3:.3g} kg/m³) is below the applied "
                f"active concentration ({applied_conc_kg_m3:.3g} kg/m³): the vehicle is predicted to "
                f"be saturated and a separate active phase is modelled.")

        s = case.scenario
        frac_non_ion = (case.permeant.frac_non_ion if case.permeant.frac_non_ion is not None
                        else props.frac_non_ionised(7.4))
        frac_unbound = case.permeant.frac_unbound
        if frac_unbound is None:
            Kow = 10 ** props.logP
            if props.acid_base() == "A":
                frac_unbound = 1 - (0.7936 * math.exp(math.log10(Kow)) + 0.2239) / (0.7936 * math.exp(math.log10(Kow)) + 1.2239)
            else:
                frac_unbound = 1 - (0.5578 * math.exp(math.log10(Kow)) + 0.0188) / (0.5578 * math.exp(math.log10(Kow)) + 1.0188)
            frac_unbound = min(max(frac_unbound, 1e-4), 1.0)
        vp = case.permeant.vapour_pressure_pa or 0.0
        evap_on = (s.occlusion == Occlusion.unoccluded and s.dose_regime == DoseRegime.finite)
        if evap_on and not vp:
            warnings.append("unoccluded finite dose, but no vapour pressure is stored for this "
                            "permeant: only the solvent evaporates in the simulation.")

        _kow_detail = {
            "logd8": ("the convention the published KNIME workflow uses: "
                      "'compute chem' sets CHEM_KOW = 10**logP and the legacy node defines "
                      "logP as the ChemAxon 'logD at pH=8.0' column",
                      "CHEM_KOW = 10 ** logP ; logP = table['logD at pH=8.0']", TRACED),
            "dow_ph": ("octanol-water distribution coefficient at the formulation pH",
                       "Dow(pH)", COMPUTED),
            "dow_74": ("octanol-water distribution coefficient at pH 7.4", "Dow(7.4)", COMPUTED),
            "logp_neutral": ("neutral-form partition coefficient", "logP", COMPUTED),
        }[self.kow_convention]
        _kow = 10 ** {"logd8": props.logD(8.0), "dow_ph": logD_pH,
                      "dow_74": props.logD(7.4), "logp_neutral": props.logP}[self.kow_convention]
        pr.record("step3.CHEM_KOW", _kow, _kow_detail[2], _kow_detail[0], snippet=_kow_detail[1],
                  reference="E2E KNIME workflow, compute chem (#46)/(#37)"
                            if self.kow_convention == "logd8" else "")
        pr.record("step3.frac_non_ion", round(frac_non_ion, 6),
                  DATABASE if case.permeant.frac_non_ion is not None else COMPUTED,
                  "stored with the permeant record" if case.permeant.frac_non_ion is not None
                  else "Henderson-Hasselbalch at pH 7.4")
        pr.record("step3.frac_unbound", round(frac_unbound, 6),
                  DATABASE if case.permeant.frac_unbound is not None else COMPUTED,
                  "stored with the permeant record" if case.permeant.frac_unbound is not None
                  else "SurreyDPK's Yamazaki/Kanaoka correlation",
                  reference="" if case.permeant.frac_unbound is not None else "core/chemical.py")

        inp = step3.PBPKInputs(
            mw=props.mw,
            kow=_kow,
            frac_non_ion=frac_non_ion, frac_unbound=frac_unbound,
            acid_base=props.acid_base(),
            density=case.permeant.density_kg_m3 or 1000.0,
            vapour_pressure_pa=vp,
            kw_vh=q.Pvw, solubility_kg_m3=Sf_kg_m3,
            active_mass_fraction=f.active_fraction,
            dose_ml_cm2=case.exposure.dose_ml_cm2, duration_h=case.exposure.duration_h,
            temperature_c=case.exposure.temperature_c, area_cm2=case.exposure.area_cm2,
            test_type=s.test_type.value, application=s.application.value,
            removal_time_h=case.exposure.removal_time_h,
            occlusion=s.occlusion.value, dose_regime=s.dose_regime.value,
            relative_humidity=case.exposure.relative_humidity)
        if self.skin_preset is not None:
            sp = self.skin_preset
            for attr, val in (("n_sc_layers", sp.n_sc_layers),
                              ("sc_surface_hydration", sp.sc_surface_hydration),
                              ("receptor_D", sp.receptor_D),
                              ("dermis_thickness_m", sp.dermis_um() * 1e-6),
                              ("sc_D_scale", sp.sc_D_scale)):
                setattr(inp, attr, val)
                org = sp.origins.get(attr, FITTED)
                detail = {FITTED: f"calibrated against published results (preset '{sp.name}')",
                          LITERATURE: "measured in the source experimental paper",
                          DEFAULT: "preset leaves the model default unchanged",
                          }.get(org, f"skin preset '{sp.name}'")
                pr.record(f"step3.{attr}", val, org, detail,
                          snippet=sp.provenance if org == FITTED else "",
                          reference="README 4.3d / 4.3g")
        else:
            for attr in ("n_sc_layers", "sc_surface_hydration", "receptor_D", "dermis_thickness_m"):
                pr.record(f"step3.{attr}", getattr(inp, attr), DEFAULT,
                          "SurreyDPK template default", reference="vendor/SurreyDPK")
        if inp.sc_D_scale != 1.0 and "step3.sc_D_scale" not in pr:
            pr.record("step3.sc_D_scale", inp.sc_D_scale, FITTED,
                      "stratum-corneum diffusivity scaling calibrated against published totals",
                      reference="README 4.3g")
        for nm, val in (("compartments", "V,S,E,D,H" if inp.test_type == "in_vitro" else "V,S,E,D"),
                        ("evaporation_module", inp.evaporation_on()),
                        ("infinite_vehicle", inp.dose_regime == "infinite")):
            pr.record(f"step3.{nm}", val, COMPUTED,
                      "selected by the orchestrator from the scenario classification",
                      snippet=f"{inp.test_type}, {inp.application}, {inp.occlusion}, {inp.dose_regime}")
        self._say(f"[orchestrator] scenario -> {inp.test_type}, {inp.application}, {inp.occlusion}, "
                  f"{inp.dose_regime} dose; evaporation sub-module "
                  f"{'ON' if inp.evaporation_on() else 'off'}")
        try:
            res = step3.run(inp, profile_times_h=profile_times_h)
        except Exception as exc:
            self._say(f"[orchestrator] PBPK failed ({exc}); retrying with a finer time grid")
            inp.steps_per_hour = max(10, inp.steps_per_hour * 2)
            res = step3.run(inp, profile_times_h=profile_times_h)
        bal = res.summary["mass_balance"]
        if abs(bal - 1.0) > 0.02:
            warnings.append(f"mass balance is {bal:.3f} rather than 1.000; treat the result with caution.")
        self._say("[step 3] absorbed into skin {:.4f}; receptor/systemic {:.4f}; evaporated {:.4f}".format(
            res.summary["fraction_in_skin"], res.summary["fraction_receptor_or_systemic"],
            res.summary["fraction_evaporated"]))

        properties = props.to_dict(pH=pH)
        properties["frac_non_ion_7.4"] = frac_non_ion
        properties["frac_unbound"] = frac_unbound
        properties["Sf_kg_m3"] = Sf_kg_m3
        return CaseResult(case=case, properties=properties, qspr=q.to_dict(), pbpk=res,
                          log=list(self.log), warnings=warnings, provenance=self.prov,
                          extraction_disagreements=list(self.disagreements),
                          cross_check_status=self.cross_check_status, domain=domain)

    def simulate(self, text: str, confirm: bool = False, profile_times_h=None) -> CaseResult:
        case = self.parse(text)
        case = self.clarify(case)
        restated = case.restate(self.prov, self.disagreements)
        self._say("[orchestrator] confirmed parameter set:\n" + restated)
        if confirm and self.ask is not None:
            reply = self.ask(restated + "\n\nIs this correct? (yes / what to change)")
            if not reply.strip().lower().startswith(("y", "ok", "correct", "是", "对")):
                self.parse(reply, case)
                case = self.clarify(case)
        if case.formulation.name and not self.db.get_formulation(case.formulation.name) \
                and case.formulation.complete():
            self.db.upsert_formulation(
                case.formulation.name, case.formulation.cosolvent, case.formulation.cosolvent_fraction,
                case.formulation.water_fraction, case.formulation.oil_fraction,
                case.formulation.surfactant_fraction, case.formulation.insoluble_fraction,
                case.formulation.active_fraction, case.exposure.pH, source="saved by agent")
            self._say(f"[database] formulation '{case.formulation.name}' saved for re-use")
        if case.permeant.name and case.permeant.smiles and not self.db.get_permeant(case.permeant.name):
            self.db.upsert_permeant(case.permeant.name, case.permeant.smiles, source="saved by agent")
            self._say(f"[database] permeant '{case.permeant.name}' saved for re-use")
        return self.run_pipeline(case, profile_times_h=profile_times_h)
