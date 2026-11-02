from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .units import normalise

ROOT = Path(__file__).resolve().parent.parent

EXTRACTION_TOOL = {
    "name": "record_simulation_case",
    "description": (
        "Record the percutaneous-absorption simulation case described by the user. "
        "Leave a field null when the user did not state it or imply it unambiguously - "
        "never guess a numerical value. Report every quantity as the number and the unit "
        "exactly as the user wrote them; do NOT convert units, the caller does that."),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {"type": "string",
                       "enum": ["new_project", "retrieve_project", "modify_case", "query_report", "other"]},
            "project_name": {"type": ["string", "null"]},
            "permeant_name": {"type": ["string", "null"]},
            "permeant_smiles": {"type": ["string", "null"]},
            "formulation_name": {"type": ["string", "null"]},
            "cosolvent": {"type": ["string", "null"]},
            "cosolvent_fraction": {"type": ["number", "null"], "description": "mass fraction 0-1"},
            "water_fraction": {"type": ["number", "null"]},
            "oil_fraction": {"type": ["number", "null"]},
            "surfactant_fraction": {"type": ["number", "null"]},
            "insoluble_fraction": {"type": ["number", "null"]},
            "active_fraction": {"type": ["number", "null"]},
            "pH": {"type": ["number", "null"]},

            "dose_value": {"type": ["number", "null"],
                           "description": "the applied-dose number exactly as the user wrote it, "
                                          "with no unit conversion"},
            "dose_unit": {"type": ["string", "null"],
                          "description": "the dose unit exactly as the user wrote it, e.g. "
                                         "'mL/cm2', 'uL/cm2'. If the user gave a bare volume "
                                         "spread over an area, put the volume unit here "
                                         "(e.g. 'uL') and fill area_value/area_unit."},
            "area_value": {"type": ["number", "null"],
                           "description": "application-area number as written, if stated"},
            "area_unit": {"type": ["string", "null"], "description": "e.g. 'cm2'"},
            "temperature_value": {"type": ["number", "null"],
                                  "description": "the temperature number exactly as written, with "
                                                 "no conversion: for 308 K report 308, not 34.85"},
            "temperature_unit": {"type": ["string", "null"],
                                 "description": "'C', 'K' or 'F', as the user wrote it"},
            "duration_value": {"type": ["number", "null"],
                               "description": "exposure-duration number exactly as written"},
            "duration_unit": {"type": ["string", "null"],
                              "description": "'h', 'min' or 'days', as the user wrote it"},
            "removal_time_value": {"type": ["number", "null"],
                                   "description": "wash-off removal-time number exactly as written"},
            "removal_time_unit": {"type": ["string", "null"], "description": "'h' or 'min'"},

            "application": {"type": ["string", "null"], "enum": ["leave_on", "wash_off", None]},
            "test_type": {"type": ["string", "null"], "enum": ["in_vitro", "in_vivo", None]},
            "occlusion": {"type": ["string", "null"], "enum": ["occluded", "unoccluded", None]},
            "dose_regime": {"type": ["string", "null"], "enum": ["finite", "infinite", None],
                            "description": "ONLY when the user states or describes the dose regime "
                                           "in words (e.g. 'infinite dose', 'a finite film', 'the "
                                           "donor was kept replenished'). Do NOT infer it from how "
                                           "large the dose is - the caller applies that threshold."},
            "evidence": {"type": "object",
                         "description": "for EVERY field you set, the verbatim fragment of the "
                                        "user's text that justified it, keyed by the field name "
                                        "(e.g. {\"pH\": \"at pH 3.8\", \"dose_ml_cm2\": "
                                        "\"dose 3.1 uL/cm2\", \"occlusion\": \"under an "
                                        "occlusive cover\"}). Quote the user, do not paraphrase.",
                         "additionalProperties": {"type": "string"}},
        },
        "required": ["intent"],
    },
}

EXTRACTION_SYSTEM = """You are the Parameter Extraction and scenario-classification step of an \
in-silico percutaneous-absorption agent. From the user's free-text description of a permeation \
experiment you fill a typed schema by calling record_simulation_case exactly once.

Rules:
* Never invent a numerical value. If the user does not state a quantity, and it cannot be derived \
from what they did state, set the field to null so the clarification loop can ask for it.
* Do NOT convert units and do NOT do arithmetic. Report each quantity as the number and the unit \
the user wrote, in the matching *_value and *_unit fields: "308 K" is dose_value-style \
temperature_value 308 with temperature_unit "K", never 34.85; "90 minutes" is duration_value 90 \
with duration_unit "min", never 1.5; "15.5 uL applied over 5 cm2" is dose_value 15.5, dose_unit \
"uL", area_value 5, area_unit "cm2". The caller performs every conversion in code.
* For every field you fill, copy the verbatim fragment of the user's text that justified it into \
`evidence` under that field's name, so the report can show where each number came from. For a \
value+unit pair use the base field name as the key (e.g. "temperature", "dose", "duration").
* Classify the three scenario descriptors even when they are only implied by the experimental \
description, and record the justifying phrase in `evidence`:
  - application: "wash_off" if the product is rinsed/washed/wiped off after a contact time \
(record that time as removal_time_h), otherwise "leave_on" for a product left on the skin.
  - test_type: "in_vitro" for diffusion cells, Franz cells, excised/dermatomed skin, receptor \
fluid, porcine/cadaver skin; "in_vivo" for human volunteers, subjects, forearm application, \
clinical exposure, living animals.
  - occlusion: "occluded" for a covered application, occlusive chamber/patch/film, sealed donor \
compartment; "unoccluded" for an open donor compartment, exposed to air, uncovered.
  - dose_regime: set "infinite" only if the text describes an effectively unlimited donor (an \
"infinite dose", a replenished or continuously supplied donor, a reservoir/saturated donor held \
at constant concentration); set "finite" only if the text describes a limited applied film (a \
"finite dose", "a thin film", a single application that is not replenished). If the text says \
nothing about the regime in words, leave it null - the caller derives it from the dose.
  If a descriptor is genuinely ambiguous or contradictory, leave it null rather than guessing.
"""

CLARIFY_SYSTEM = """You are the Clarification Loop of an in-silico percutaneous-absorption agent. \
You are given the list of fields that are missing, ambiguous or failed validation, plus any close \
matches retrieved from the formulation/permeant database. Write ONE short, specific question to \
the user that names the missing field(s) and their units, and offers the database suggestions when \
there are any. Do not ask for anything that is already known. Do not propose a value of your own \
for a numerical field."""

REPORT_SYSTEM = """You are the Report Generator of an in-silico percutaneous-absorption agent. \
You are given the simulation inputs and the numerical outputs of the three deterministic steps \
(property estimation, QSPR, PBPK). Write a concise technical summary for a formulation scientist: \
what was simulated, the estimated properties, the vehicle-water partition coefficient and what it \
implies, where the active ended up, and any applicability-domain caveats you are given. Use only \
the numbers supplied - never invent one. State clearly that the free-text commentary should be \
checked by a domain expert before it informs a formulation decision."""


NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

IN_VITRO_CUES = [
    (r"\bin[- ]?vitro\b", "in vitro"), (r"\bfranz\b", "Franz cell"),
    (r"\bdiffusion cell", "diffusion cell"), (r"\breceptor (?:fluid|compartment|chamber)", "receptor fluid"),
    (r"\bexcised\b", "excised skin"), (r"\bdermatomed\b", "dermatomed skin"),
    (r"\bcadaver\b", "cadaver skin"), (r"\bporcine (?:ear |dorsal )?skin\b", "porcine skin"),
    (r"\bIVPT\b", "IVPT"), (r"\bdonor (?:chamber|compartment)", "donor chamber"),
    (r"\bskin membrane", "skin membrane"), (r"\bstatic cell", "static cell"),
]
IN_VIVO_CUES = [
    (r"\bin[- ]?vivo\b", "in vivo"), (r"\bvolunteer", "volunteers"), (r"\bsubjects?\b", "subjects"),
    (r"\bforearm\b", "forearm"), (r"\bparticipants?\b", "participants"),
    (r"\bhealthy (?:adult|human)", "healthy humans"), (r"\bclinical\b", "clinical study"),
    (r"\bapplied to the (?:skin of|backs? of) (?:\d+ )?(?:healthy )?(?:volunteers|subjects)", "volunteers"),
]
WASH_CUES = [
    (r"\bwash(?:ed|-)?[ -]?off\b", "wash-off"),
    (r"\brinse[sd]?\s+(?:off|from|away|out|with)\b", "rinsed off"),
    (r"\bwash(?:ed|es)?\s+(?:off|from|away|out|with)\b", "washed off"),
    (r"\bwipe[sd]?\s+(?:off|from|away)\b", "wiped off"),
    (r"\bremoved\s+(?:after|from|at)\b", "removed after"),
    (r"\bshampoo\b", "shampoo (wash-off product)"),
    (r"\bcleanser\b", "cleanser (wash-off product)"), (r"\bbody wash\b", "body wash"),
]
LEAVE_CUES = [
    (r"\bleave[- ]?on\b", "leave-on"), (r"\bleft on (?:the skin|in place)", "left on the skin"),
    (r"\bnot (?:removed|washed)", "not removed"), (r"\bmoisturis?zer\b", "moisturiser (leave-on)"),
    (r"\bremained? (?:on the skin )?(?:for|throughout)", "remained on the skin"),
    (r"\bno rinsing\b", "no rinsing step"), (r"\bnot rinsed\b", "not rinsed"),
    (r"\ballowed to remain\b", "allowed to remain on the skin"),
    (r"\bstays? in place\b", "stays in place"),
    (r"\bwithout (?:being )?(?:rinsing|rinsed|washing|washed|removal|removed)\b", "without rinsing"),
    (r"\bserum\b", "serum (leave-on)"), (r"\blotion\b", "lotion (leave-on)"),
]
OCCL_CUES = [
    (r"\bocclu(?:ded|sive|sion)\b", "occluded"), (r"\bcovered with\b", "covered"),
    (r"\bhill ?top chamber", "Hill Top chamber"), (r"\bsealed\b", "sealed"),
    (r"\bunder (?:a )?(?:patch|film|dressing)", "under a patch"), (r"\bparafilm\b", "Parafilm"),
]
UNOCCL_CUES = [
    (r"\bun[- ]?occlu(?:ded|sive)\b", "unoccluded"), (r"\bnon[- ]?occlu(?:ded|sive)\b", "non-occluded"),
    (r"\bopen to (?:the )?(?:air|atmosphere)", "open to the air"), (r"\bleft open\b", "left open"),
    (r"\buncovered\b", "uncovered"), (r"\bwithout occlusion\b", "without occlusion"),
    (r"\bexposed to (?:the )?air", "exposed to air"), (r"\ballowing evaporation", "allowing evaporation"),
]


INFINITE_DOSE_CUES = [
    (r"\binfinite[- ]dose\b", "infinite dose"), (r"\binfinite dose\b", "infinite dose"),
    (r"\bnear[- ]infinite\b", "near-infinite dose"),
    (r"\breplenish(?:ed|ing)?\b", "replenished donor"),
    (r"\bcontinuously (?:supplied|replaced|refreshed)\b", "continuously supplied donor"),
    (r"\bdonor (?:was |is )?(?:kept |held )?(?:at )?(?:a )?(?:constant|saturated)\b",
     "donor held at constant concentration"),
    (r"\breservoir\b", "reservoir donor"),
    (r"\bunlimited (?:donor|supply|reservoir)\b", "unlimited donor"),
]
FINITE_DOSE_CUES = [
    (r"\bfinite[- ]dose\b", "finite dose"), (r"\bfinite dose\b", "finite dose"),
    (r"\bthin film\b", "thin film"), (r"\bfinite film\b", "finite film"),
    (r"\bsingle (?:application|dose)\b", "single application"),
    (r"\bnot replenished\b", "donor not replenished"),
    (r"\bin[- ]use (?:dose|level)\b", "in-use dose"),
]


def _first(text, cues):
    for pat, label in cues:
        m = re.search(pat, text, re.I)
        if m:
            return label, m.group(0)
    return None, None


def _to_hours(value: float, unit: str) -> float:
    u = unit.lower()
    if u.startswith("min"):
        return value / 60.0
    if u.startswith("day") or u == "d":
        return value * 24.0
    if u.startswith("sec") or u == "s":
        return value / 3600.0
    return value


class RuleBackend:
    name = "rule"

    def extract(self, text: str) -> Dict[str, Any]:
        t = " " + text.strip() + " "
        out: Dict[str, Any] = {"intent": "new_project", "evidence": {}}

        if re.search(r"\b(load|open|retrieve|use) (?:the )?(?:saved |previous |existing )?project\b", t, re.I):
            out["intent"] = "retrieve_project"
        elif re.search(r"\b(change|modify|update|re-?run|instead|adjust)\b", t, re.I) and \
                not re.search(r"\bsimulate\b", t, re.I):
            out["intent"] = "modify_case"
        elif re.search(r"\b(what|why|how much|explain|show me the report)\b", t, re.I) and \
                not re.search(r"\bsimulate\b", t, re.I):
            out["intent"] = "query_report"

        def _ev(fieldname, match):
            if match is not None:
                out["evidence"][fieldname] = match.group(0).strip()

        m = re.search(rf"\bpH\s*(?:of|=|:)?\s*({NUM})", t, re.I)
        if m:
            out["pH"] = float(m.group(1)); _ev("pH", m)

        m = re.search(rf"({NUM})\s*(?:°|deg(?:rees)?\.?\s*)?\s*C\b", t)
        if m:
            out["temperature_value"] = float(m.group(1)); out["temperature_unit"] = "C"
            _ev("temperature", m)
        else:
            m = re.search(rf"({NUM})\s*(?:°\s*)?K\b", t)
            if m:
                out["temperature_value"] = float(m.group(1)); out["temperature_unit"] = "K"
                _ev("temperature", m)

        UNIT = r"hours?|hrs?|h|minutes?|mins?|min|days?|d"
        m = re.search(rf"(?:for|over|during)\s+({NUM})\s*[-\s]*({UNIT})\b", t, re.I)
        if not m:
            m = re.search(rf"\b({NUM})\s*-\s*({UNIT})\b", t, re.I)
        if not m:
            m = re.search(rf"\b({NUM})\s*({UNIT})\b", t, re.I)
        if m:
            out["duration_value"] = float(m.group(1)); out["duration_unit"] = m.group(2)
            _ev("duration", m)
        RUNIT = r"hours?|hrs?|h|minutes?|mins?|min"
        m = re.search(rf"(?:then |and )?(?:rinsed|washed|wiped|removed)[^.;]*?after\s+({NUM})\s*[-\s]*"
                      rf"({RUNIT})\b", t, re.I)
        if not m:
            m = re.search(rf"\b(?:for|after)\s+({NUM})\s*[-\s]*({RUNIT})\b[^.;,]{{0,20}}?[,\s]*"
                          rf"(?:and\s+)?then\s+(?:\w+\s+){{0,2}}?(?:rinsed|washed|wiped|removed)", t, re.I)
        if not m:
            m = re.search(rf"(?:rinsed|washed|wiped|removed)[^.;]{{0,20}}?after\s+({NUM})\s*[-\s]*({RUNIT})\b",
                          t, re.I)
        if m:
            out["removal_time_value"] = float(m.group(1)); out["removal_time_unit"] = m.group(2)
            _ev("removal_time", m)

        m = re.search(rf"({NUM})\s*(mL|ml)\s*(?:/|per )\s*cm\s*(?:\^?2|²)", t)
        if m:
            out["dose_value"] = float(m.group(1)); out["dose_unit"] = "mL/cm2"; _ev("dose", m)
        else:
            m = re.search(rf"({NUM})\s*(?:µL|uL|μL|ul)\s*(?:/|per )\s*cm\s*(?:\^?2|²)", t)
            if m:
                out["dose_value"] = float(m.group(1)); out["dose_unit"] = "uL/cm2"; _ev("dose", m)
            else:
                m = re.search(rf"({NUM})\s*(mL|ml|µL|uL|μL|ul)\b[^.;]{{0,40}}?\bover\s+(?:an? )?"
                              rf"({NUM})\s*cm\s*(?:\^?2|²)", t)
                if m:
                    out["dose_value"] = float(m.group(1))
                    out["dose_unit"] = "uL" if m.group(2).lower() in ("µl", "ul", "μl") else "mL"
                    out["area_value"] = float(m.group(3))
                    out["area_unit"] = "cm2"
                    _ev("dose", m); _ev("area", m)

        lab, ev = _first(t, WASH_CUES)
        if lab:
            out["application"] = "wash_off"; out["evidence"]["application"] = ev
        else:
            lab, ev = _first(t, LEAVE_CUES)
            if lab:
                out["application"] = "leave_on"; out["evidence"]["application"] = ev
        lab, ev = _first(t, IN_VIVO_CUES)
        if lab:
            out["test_type"] = "in_vivo"; out["evidence"]["test_type"] = ev
        else:
            lab, ev = _first(t, IN_VITRO_CUES)
            if lab:
                out["test_type"] = "in_vitro"; out["evidence"]["test_type"] = ev
        lab, ev = _first(t, UNOCCL_CUES)
        if lab:
            out["occlusion"] = "unoccluded"; out["evidence"]["occlusion"] = ev
        else:
            lab, ev = _first(t, OCCL_CUES)
            if lab:
                out["occlusion"] = "occluded"; out["evidence"]["occlusion"] = ev

        m = re.search(r"\b(?:of|for)\s+([A-Za-z][A-Za-z0-9 \-]{2,30}?)\s+(?:from|in|delivered)", t)
        if m:
            out["permeant_name"] = m.group(1).strip(); _ev("permeant_name", m)
        m = re.search(r"\b(?:from|in|using)\s+((?:Formulation|formulation)\s+[A-Za-z0-9]+)", t)
        if m:
            out["formulation_name"] = m.group(1).strip(); _ev("formulation_name", m)
        else:
            m = re.search(r"\bfrom (?:an?|the)\s+([a-z0-9 \-]{3,40}?)\s*(?:\(|,|\bat\b|\bon\b|\bfor\b|$)", t, re.I)
            if m:
                out["formulation_name"] = m.group(1).strip()
        m = re.search(r"\bSMILES[:=]?\s*([^\s,;]+)", t, re.I)
        if m:
            out["permeant_smiles"] = m.group(1)

        lab, ev = _first(t, INFINITE_DOSE_CUES)
        if lab:
            out["dose_regime"] = "infinite"; out["evidence"]["dose_regime"] = ev
        else:
            lab, ev = _first(t, FINITE_DOSE_CUES)
            if lab:
                out["dose_regime"] = "finite"; out["evidence"]["dose_regime"] = ev

        out, problems = normalise(out)
        if problems:
            out.setdefault("unit_problems", []).extend(problems)
        return out

    def clarify(self, missing, suggestions) -> str:
        bits = []
        for path, desc in missing:
            s = suggestions.get(path)
            if s:
                bits.append(f"{desc} (closest entries in the database: {', '.join(s)})")
            else:
                bits.append(desc)
        if len(bits) == 1:
            return f"Before I can run the simulation I still need {bits[0]}. Could you provide it?"
        return ("Before I can run the simulation I still need:\n"
                + "\n".join(f"  - {b}" for b in bits)
                + "\nCould you provide these?")

    def report(self, payload: str) -> str:
        return ""


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-opus-5"
TEMPERATURE = 0.0
MAX_TOKENS_TOOL_CALL = 2000
MAX_TOKENS_TEXT = 1500


class APIBackend:

    def __init__(self, model: str = MODEL, api_key: Optional[str] = None,
                 temperature: float = TEMPERATURE):
        self.model = model
        self.temperature = temperature
        self.api_key = ANTHROPIC_API_KEY if api_key is None else api_key
        self.provider = "anthropic"
        self.name = f"api:{self.model}:T{self.temperature:g}"

    def _client(self):
        import anthropic
        if not self.api_key:
            raise RuntimeError(
                "no API key: set ANTHROPIC_API_KEY, or pass api_key= to APIBackend")
        return anthropic.Anthropic(api_key=self.api_key)

    def _tool_call(self, system, user, tool):
        msg = self._client().messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS_TOOL_CALL,
            temperature=self.temperature,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": user}],
        )
        for block in msg.content:
            if block.type == "tool_use":
                return dict(block.input)
        raise RuntimeError("model did not call the tool")

    def _text(self, system, user):
        msg = self._client().messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS_TEXT,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def extract(self, text):
        raw = self._tool_call(EXTRACTION_SYSTEM, text, EXTRACTION_TOOL)
        out, problems = normalise(raw)
        if problems:
            out.setdefault("unit_problems", []).extend(problems)
        return out

    def clarify(self, missing, suggestions):
        payload = json.dumps({"missing": [{"field": p, "description": d} for p, d in missing],
                              "database_suggestions": suggestions}, ensure_ascii=False)
        return self._text(CLARIFY_SYSTEM, payload)

    def report(self, payload):
        return self._text(REPORT_SYSTEM, payload)


def get_backend(kind: str = "api", **kw):
    if kind == "api":
        return APIBackend(**kw)
    if kind == "rule":
        return RuleBackend()
    raise ValueError(f"unknown backend {kind!r}")
