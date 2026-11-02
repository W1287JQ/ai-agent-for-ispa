from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

USER = "user"
CLARIFIED = "clarified"
DATABASE = "database"
LITERATURE = "literature"
TRACED = "traced"
COMPUTED = "computed"
DEFAULT = "default"
FITTED = "fitted"

ORIGIN_LABEL = {
    USER: "stated by the user",
    CLARIFIED: "answered in the clarification loop",
    DATABASE: "formulation / permeant database",
    LITERATURE: "curated measured value",
    TRACED: "traced from the published implementation",
    COMPUTED: "computed by the pipeline",
    DEFAULT: "default from the vendored model",
    FITTED: "CALIBRATED against published results",
}
SOFT_ORIGINS = (FITTED,)
ORIGIN_ORDER = [USER, CLARIFIED, DATABASE, LITERATURE, TRACED, COMPUTED, DEFAULT, FITTED]


@dataclass
class ValueSource:
    field: str
    value: Any
    origin: str
    detail: str = ""
    snippet: str = ""
    reference: str = ""

    def to_dict(self):
        return asdict(self)

    def one_line(self, width: int = 0) -> str:
        bits = [f"[{self.origin}]"]
        if self.detail:
            bits.append(self.detail)
        if self.snippet:
            bits.append(f'"{self.snippet}"')
        if self.reference:
            bits.append(f"({self.reference})")
        s = " ".join(bits)
        return s if width <= 0 or len(s) <= width else s[: width - 1] + "…"


class Provenance:

    def __init__(self):
        self._items: Dict[str, ValueSource] = {}

    def record(self, field_: str, value: Any, origin: str, detail: str = "",
               snippet: str = "", reference: str = "") -> ValueSource:
        vs = ValueSource(field=field_, value=value, origin=origin, detail=detail,
                         snippet=(snippet or "").strip(), reference=reference)
        self._items[field_] = vs
        return vs

    def get(self, field_: str) -> Optional[ValueSource]:
        return self._items.get(field_)

    def __contains__(self, field_):
        return field_ in self._items

    def __len__(self):
        return len(self._items)

    def items(self) -> List[ValueSource]:
        return list(self._items.values())

    def by_origin(self, origin: str) -> List[ValueSource]:
        return [v for v in self._items.values() if v.origin == origin]

    def calibrated(self) -> List[ValueSource]:
        return [v for v in self._items.values() if v.origin in SOFT_ORIGINS]

    def counts(self) -> Dict[str, int]:
        out = {}
        for v in self._items.values():
            out[v.origin] = out.get(v.origin, 0) + 1
        return out

    def to_dict(self):
        return {k: v.to_dict() for k, v in self._items.items()}

    def _fmt_value(self, v) -> str:
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def render_text(self, width: int = 110) -> str:
        if not self._items:
            return "(no provenance recorded)"
        rows = sorted(self._items.values(),
                      key=lambda v: (ORIGIN_ORDER.index(v.origin)
                                     if v.origin in ORIGIN_ORDER else 99, v.field))
        fw = max(len(v.field) for v in rows)
        vw = min(max(len(self._fmt_value(v.value)) for v in rows), 18)
        lines = [f"{'field'.ljust(fw)}  {'value'.ljust(vw)}  origin / where it came from",
                 "-" * min(width, fw + vw + 50)]
        for v in rows:
            lines.append(f"{v.field.ljust(fw)}  {self._fmt_value(v.value)[:vw].ljust(vw)}  "
                         f"{v.one_line(width - fw - vw - 4)}")
        c = self.counts()
        lines.append("")
        lines.append("counts: " + ", ".join(f"{k}={c[k]}" for k in ORIGIN_ORDER if k in c))
        if self.calibrated():
            lines.append(f"NOTE: {len(self.calibrated())} value(s) were calibrated against published "
                         f"results rather than measured or derived; agreement in those directions "
                         f"is partly by construction.")
        return "\n".join(lines)

    def render_markdown(self) -> str:
        if not self._items:
            return "_(no provenance recorded)_\n"
        rows = sorted(self._items.values(),
                      key=lambda v: (ORIGIN_ORDER.index(v.origin)
                                     if v.origin in ORIGIN_ORDER else 99, v.field))
        out = ["| field | value | origin | where it came from |", "| --- | --- | --- | --- |"]
        for v in rows:
            where = v.detail
            if v.snippet:
                where += (" — " if where else "") + f"`{v.snippet}`"
            if v.reference:
                where += f" ({v.reference})"
            out.append(f"| `{v.field}` | {self._fmt_value(v.value)} | {v.origin} | {where} |")
        c = self.counts()
        out.append("")
        out.append("Counts: " + ", ".join(f"**{k}** {c[k]}" for k in ORIGIN_ORDER if k in c) + ".")
        if self.calibrated():
            out.append("")
            out.append(f"> **{len(self.calibrated())} value(s) are calibrated**, i.e. tuned to reproduce "
                       f"published results rather than measured or derived: "
                       + ", ".join(f"`{v.field}`" for v in self.calibrated())
                       + ". Agreement along those directions is partly by construction.")
        return "\n".join(out)
