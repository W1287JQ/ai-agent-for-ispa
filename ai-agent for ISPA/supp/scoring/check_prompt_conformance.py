#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_code.llm import EXTRACTION_SYSTEM, EXTRACTION_TOOL

OUTPUTS = ROOT / "outputs" / "outputs.jsonl"
BENCHMARKS = ROOT / "benchmarks"
SETS = {"set_a": "set_a_scenarios.json", "set_b": "set_b_clarification.json",
        "set_c": "set_c_dose_regime.json", "set_d": "set_d_freetext.json",
        "set_e": "set_e_unstated_removal.json"}
PAIRS = ("temperature", "dose", "duration", "removal_time", "area")
SCENARIO_DESCRIPTORS = ("application", "test_type", "occlusion", "dose_regime")
DERIVED = ("dose_ml_cm2", "temperature_c", "duration_h", "removal_time_h",
           "area_cm2", "unit_problems")


def load_prompts():
    out = {}
    for key, fname in SETS.items():
        path = BENCHMARKS / fname
        if path.exists():
            for item in json.loads(path.read_text()):
                out[(key, str(item["id"]))] = item
    return out


def main():
    if not OUTPUTS.exists():
        print(f"not found: {OUTPUTS}")
        return 1
    items = load_prompts()
    props = set(EXTRACTION_TOOL["input_schema"]["properties"])
    fails, n = [], 0
    stats = {"kelvin_kept": 0, "fahrenheit_kept": 0, "non_hour_kept": 0,
             "evidence_spans": 0, "fields_set": 0, "nulls_where_unstated": 0,
             "unconvertible_reported_as_written": 0,
             "descriptor_withdrawn_by_guardrail": 0,
             "regime_set": 0, "regime_left_to_threshold": 0}

    for line in OUTPUTS.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        call = rec.get("tool_call")
        if not call:
            continue
        n += 1
        where = f"{rec['set']}:{rec['id']}"
        prompt = rec["prompt"]
        item = items.get((rec["set"], str(rec["id"]))) or {}

        extra = set(call) - props - set(DERIVED)
        if extra:
            fails.append(f"{where}: fields outside the tool schema: {sorted(extra)}")

        for base in PAIRS:
            v, u = call.get(f"{base}_value"), call.get(f"{base}_unit")
            if v is None:
                continue
            if base == "temperature":
                if re.search(r"\b\d+\s*K\b", prompt):
                    if str(u).strip().upper() != "K":
                        fails.append(f"{where}: prompt states kelvin but unit is {u!r}")
                    elif v < 100:
                        fails.append(f"{where}: kelvin value {v} looks converted")
                    else:
                        stats["kelvin_kept"] += 1
                elif "°F" in prompt and str(u).strip().upper() == "F":
                    stats["fahrenheit_kept"] += 1
            if base in ("duration", "removal_time"):
                if u and not re.match(r"^h(ours?|rs?)?$", str(u).strip(), re.I):
                    stats["non_hour_kept"] += 1

        ev = call.get("evidence") or {}
        scored = [k for k in call
                  if k not in ("intent", "evidence") and k not in DERIVED
                  and not k.endswith("_unit") and call[k] is not None]
        for k in scored:
            stats["fields_set"] += 1
            base = k[:-6] if k.endswith("_value") else k
            if ev.get(base) or ev.get(k):
                stats["evidence_spans"] += 1

        for f in item.get("expect_clarify", []) or []:
            field = {"permeant": "permeant_name", "dose_ml_cm2": "dose_value",
                     "temperature_c": "temperature_value",
                     "duration_h": "duration_value"}.get(f, f)
            canonical = {"permeant": "permeant_name"}.get(f, f)
            if f in SCENARIO_DESCRIPTORS:
                if call.get(f) is not None:
                    stats["descriptor_withdrawn_by_guardrail"] += 1
                else:
                    stats["nulls_where_unstated"] += 1
                continue
            if call.get(canonical) is not None:
                fails.append(f"{where}: {f} is not determined by the text but "
                             f"{canonical}={call.get(canonical)!r} was returned")
            elif call.get(field) is not None:
                if not call.get(field.replace("_value", "_unit")):
                    fails.append(f"{where}: {f} reported as {call[field]!r} with no unit")
                else:
                    stats["unconvertible_reported_as_written"] += 1
            else:
                stats["nulls_where_unstated"] += 1

        says_regime = re.search(
            r"infinite|finite|replenish|topped back up|continuously refreshed|"
            r"reservoir|thin film|not replenished|not topped up", prompt, re.I)
        if call.get("dose_regime") is not None:
            stats["regime_set"] += 1
            if not says_regime:
                fails.append(f"{where}: dose_regime={call['dose_regime']!r} but the "
                             f"text describes no regime in words")
        else:
            stats["regime_left_to_threshold"] += 1

    print(f"responses checked: {n}")
    print(f"system prompt: EXTRACTION_SYSTEM ({len(EXTRACTION_SYSTEM)} chars)")
    print(f"tool schema  : EXTRACTION_TOOL, {len(props)} fields\n")
    print(f"  kelvin left unconverted                    : {stats['kelvin_kept']}")
    print(f"  fahrenheit left unconverted                : {stats['fahrenheit_kept']}")
    print(f"  non-hour time units left unconverted       : {stats['non_hour_kept']}")
    print(f"  fields carrying an evidence span           : "
          f"{stats['evidence_spans']}/{stats['fields_set']}")
    print(f"  under-determined values left null          : {stats['nulls_where_unstated']}")
    print(f"  unconvertible units reported as written    : "
          f"{stats['unconvertible_reported_as_written']}")
    print(f"  descriptors classified then withdrawn      : "
          f"{stats['descriptor_withdrawn_by_guardrail']}")
    print(f"  dose_regime set from wording               : {stats['regime_set']}")
    print(f"  dose_regime left to the dose threshold     : "
          f"{stats['regime_left_to_threshold']}")
    print()
    if fails:
        print(f"CONFORMANCE FAILURES: {len(fails)}")
        for f in fails[:40]:
            print("  " + f)
        return 1
    print("No conformance failures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
