from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

TIME_TO_HOURS = {
    "h": 1.0, "hr": 1.0, "hrs": 1.0, "hour": 1.0, "hours": 1.0,
    "min": 1.0 / 60.0, "mins": 1.0 / 60.0, "minute": 1.0 / 60.0, "minutes": 1.0 / 60.0,
    "s": 1.0 / 3600.0, "sec": 1.0 / 3600.0, "secs": 1.0 / 3600.0,
    "second": 1.0 / 3600.0, "seconds": 1.0 / 3600.0,
    "d": 24.0, "day": 24.0, "days": 24.0,
    "wk": 168.0, "week": 168.0, "weeks": 168.0,
}

AREAL_DOSE_TO_ML_CM2 = {
    "ml/cm2": 1.0, "cm3/cm2": 1.0,
    "ul/cm2": 1e-3, "microlitre/cm2": 1e-3, "microliter/cm2": 1e-3,
    "l/cm2": 1e3,
    "nl/cm2": 1e-6,
    "ml/m2": 1e-4, "ul/m2": 1e-7,
}

VOLUME_TO_ML = {"ml": 1.0, "cm3": 1.0, "l": 1e3, "ul": 1e-3, "nl": 1e-6}

AREA_TO_CM2 = {"cm2": 1.0, "mm2": 1e-2, "m2": 1e4, "in2": 6.4516}

_WORD_FORMS = [
    ("microlitres", "ul"), ("microlitre", "ul"), ("microliters", "ul"), ("microliter", "ul"),
    ("millilitres", "ml"), ("millilitre", "ml"), ("milliliters", "ml"), ("milliliter", "ml"),
    ("nanolitres", "nl"), ("nanolitre", "nl"), ("nanoliters", "nl"), ("nanoliter", "nl"),
    ("milligrams", "mg"), ("milligram", "mg"),
    ("litres", "l"), ("litre", "l"), ("liters", "l"), ("liter", "l"),
    ("squarecentimetres", "cm2"), ("squarecentimetre", "cm2"),
    ("squarecentimeters", "cm2"), ("squarecentimeter", "cm2"),
    ("sqcm", "cm2"), ("squaremetre", "m2"), ("squaremeter", "m2"),
    ("centimetressquared", "cm2"), ("centimeterssquared", "cm2"),
    ("per", "/"), ("each", "/"),
    ("degreescelsius", "c"), ("degreecelsius", "c"), ("celsius", "c"), ("centigrade", "c"),
    ("degreesfahrenheit", "f"), ("degreefahrenheit", "f"), ("fahrenheit", "f"),
    ("degreeskelvin", "k"), ("kelvin", "k"),
    ("degreesc", "c"), ("degreec", "c"), ("degc", "c"), ("degsc", "c"),
    ("degreesf", "f"), ("degreef", "f"), ("degf", "f"),
    ("degreesk", "k"), ("degreek", "k"), ("degk", "k"),
]

_SYMBOLS = {
    "µ": "u", "μ": "u",
    "²": "2", "^2": "2", "**2": "2",
    "°": "", "_": "", "·": "/",
}


def canon(unit: Optional[str]) -> str:
    if not unit:
        return ""
    u = str(unit).strip().lower()
    for src, dst in _SYMBOLS.items():
        u = u.replace(src, dst)
    u = u.replace(" ", "")
    for src, dst in _WORD_FORMS:
        u = u.replace(src, dst)
    return u.strip("/. ")


def to_celsius(value: float, unit: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    u = canon(unit)
    if u in ("c", ""):
        return float(value), None
    if u == "k":
        return float(value) - 273.15, None
    if u == "f":
        return (float(value) - 32.0) * 5.0 / 9.0, None
    return None, f"unrecognised temperature unit {unit!r}"


def to_hours(value: float, unit: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    u = canon(unit)
    if u == "":
        return float(value), None
    if u in TIME_TO_HOURS:
        return float(value) * TIME_TO_HOURS[u], None
    return None, f"unrecognised time unit {unit!r}"


def to_ml_cm2(value: float, unit: Optional[str],
              area_cm2: Optional[float] = None,
              area_unit: Optional[str] = None) -> Tuple[Optional[float], Optional[str]]:
    u = canon(unit)
    if u in AREAL_DOSE_TO_ML_CM2:
        return float(value) * AREAL_DOSE_TO_ML_CM2[u], None
    if u in VOLUME_TO_ML:
        if area_cm2 is None:
            return None, (f"dose given as a volume ({unit}) but no application area was "
                          f"stated, so it cannot be converted to mL/cm2")
        a, err = to_cm2(area_cm2, area_unit)
        if err:
            return None, err
        if not a:
            return None, "application area is zero"
        return float(value) * VOLUME_TO_ML[u] / a, None
    if u == "":
        return float(value), None
    return None, f"unrecognised dose unit {unit!r}"


def to_cm2(value: float, unit: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    u = canon(unit)
    if u == "":
        return float(value), None
    if u in AREA_TO_CM2:
        return float(value) * AREA_TO_CM2[u], None
    return None, f"unrecognised area unit {unit!r}"


PAIRS = {
    "temperature": ("temperature_c", "temp"),
    "duration": ("duration_h", "time"),
    "removal_time": ("removal_time_h", "time"),
    "dose": ("dose_ml_cm2", "dose"),
    "area": ("area_cm2", "area"),
}


def normalise(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    out = dict(raw)
    problems: List[str] = []

    area_cm2 = None
    if raw.get("area_value") is not None:
        area_cm2, err = to_cm2(raw["area_value"], raw.get("area_unit"))
        if err:
            problems.append(err)
        out["area_cm2"] = area_cm2
    elif raw.get("area_cm2") is not None:
        area_cm2 = raw["area_cm2"]

    for field, (canonical, kind) in PAIRS.items():
        if field == "area":
            continue
        vkey, ukey = f"{field}_value", f"{field}_unit"
        if raw.get(vkey) is None:
            continue
        v, u = raw[vkey], raw.get(ukey)
        if kind == "temp":
            val, err = to_celsius(v, u)
        elif kind == "time":
            val, err = to_hours(v, u)
        else:
            val, err = to_ml_cm2(v, u, area_cm2, raw.get("area_unit"))
        if err:
            problems.append(err)
        out[canonical] = val

    return out, problems
