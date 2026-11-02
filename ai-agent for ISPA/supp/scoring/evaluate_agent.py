#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from e2e_agent.database import Database
from e2e_agent.orchestrator import Agent, _norm_name

PROMPTS = ROOT / "validation" / "prompts"
NUMERIC = ["pH", "dose_ml_cm2", "temperature_c", "duration_h", "removal_time_h"]
DESCRIPTORS = ["application", "test_type", "occlusion"]


def _close(got, ref, rel=0.01, absolute=None):
    if ref is None:
        return got is None
    if got is None:
        return False
    if absolute is not None:
        return abs(got - ref) <= absolute
    if ref == 0:
        return abs(got) <= 1e-9
    return abs(got - ref) / abs(ref) <= rel


def _fresh(args, db):
    return Agent(db=db, backend=args.backend, domain_check="off")


def _regime(agent, case):
    v = getattr(case.scenario.dose_regime, "value", None)
    src = agent.prov.get("scenario.dose_regime")
    how = {"computed": "dose threshold", "user": "stated in the text",
           "clarified": "clarification loop"}.get(getattr(src, "origin", ""), "not set")
    return v, how


def run_set_a(args, db, prompts, strict_temp=False, records=None, setname="set_a"):
    conf = {d: Counter() for d in DESCRIPTORS + ["dose_regime"]}
    field_hits, field_total = Counter(), Counter()
    for p in prompts:
        agent = _fresh(args, db)
        raw = agent.llm.extract(p["prompt"])
        case = agent.parse(p["prompt"])
        ref = p["reference"]
        row = {"set": setname, "id": p["id"], "prompt": p["prompt"],
               "tool_call": raw, "reference": ref, "errors": []}

        for d in DESCRIPTORS:
            if d not in ref:
                continue
            got, want = raw.get(d), ref[d]
            conf[d][(want, got if got is not None else "clarify")] += 1
            if got != want:
                row["errors"].append(f"{d}: got {got!r}, want {want!r}")

        regime, how = _regime(agent, case)
        row["dose_regime"] = {"got": regime, "want": ref.get("dose_regime"), "decided_by": how}
        if ref.get("dose_regime"):
            conf["dose_regime"][(ref["dose_regime"], regime or "clarify")] += 1
            if regime != ref["dose_regime"]:
                row["errors"].append(
                    f"dose_regime: got {regime!r} ({how}), want {ref['dose_regime']!r}")

        for f in NUMERIC:
            if ref.get(f) is None:
                continue
            field_total[f] += 1
            ok = _close(raw.get(f), ref[f],
                        absolute=0.05 if (strict_temp and f == "temperature_c") else None)
            field_hits[f] += ok
            if not ok:
                row["errors"].append(f"{f}: got {raw.get(f)!r}, want {ref[f]!r}")
        if ref.get("permeant_name"):
            field_total["permeant_name"] += 1
            got_name = (raw.get("permeant_name") or "").strip().lower()
            ok = (got_name == ref["permeant_name"].lower()
                  or ref["permeant_name"].lower() in got_name)
            field_hits["permeant_name"] += ok
            if not ok:
                row["errors"].append(f"permeant: got {raw.get('permeant_name')!r}, "
                                     f"want {ref['permeant_name']!r}")
        if ref.get("formulation_name"):
            field_total["formulation_name"] += 1
            ok = _norm_name(raw.get("formulation_name") or "") == \
                _norm_name(ref["formulation_name"])
            field_hits["formulation_name"] += ok
            if not ok:
                row["errors"].append(f"formulation: got {raw.get('formulation_name')!r}, "
                                     f"want {ref['formulation_name']!r}")

        row["unit_problems"] = raw.get("unit_problems") or []
        row["cross_check_disagreements"] = agent.disagreements
        row["warnings"] = agent.parse_warnings
        if records is not None and not strict_temp:
            records.append(row)
    return conf, field_hits, field_total


def run_set_b(args, db, prompts, records=None):
    flagged = silent = stated_ok = 0
    rows = []
    for p in prompts:
        agent = _fresh(args, db)
        raw = agent.llm.extract(p["prompt"])
        case = agent.parse(p["prompt"])
        missing_paths = {m[0].split(".")[-1] for m in case.missing()}
        row = {"set": "set_b", "id": p["id"], "prompt": p["prompt"], "tool_call": raw,
               "reference": p["reference"], "missing": p["missing"],
               "flagged": [], "silent": []}
        for f in p["missing"]:
            if f in missing_paths:
                flagged += 1; row["flagged"].append(f)
            else:
                silent += 1; row["silent"].append(f)
        ref, ok = p["reference"], True
        for f in NUMERIC:
            if ref.get(f) is None:
                continue
            if not _close(getattr(case.exposure, f, None), ref[f]):
                ok = False
                row.setdefault("errors", []).append(
                    f"{f}: got {getattr(case.exposure, f, None)!r}, want {ref[f]!r}")
        stated_ok += ok
        row["stated_fields_ok"] = ok
        row["question"] = agent.llm.clarify(case.missing(), {}) if case.missing() else ""
        rows.append(row)
        if records is not None:
            records.append(row)
    return flagged, silent, stated_ok, rows


_D_FIELDS = {"pH": ("exposure", "pH"), "dose_ml_cm2": ("exposure", "dose_ml_cm2"),
             "temperature_c": ("exposure", "temperature_c"),
             "duration_h": ("exposure", "duration_h"),
             "removal_time_h": ("exposure", "removal_time_h")}


def run_set_d(args, db, prompts, records=None):
    hits = Counter(); total = Counter()
    clarify_ok = clarify_missed = 0
    failures = []
    for p in prompts:
        agent = _fresh(args, db)
        raw = agent.llm.extract(p["prompt"])
        case = agent.parse(p["prompt"])
        ref, expect = p["reference"], set(p["expect_clarify"])
        asked = {m[0].split(".")[-1] for m in case.missing()}
        row = {"set": "set_d", "id": p["id"], "prompt": p["prompt"],
               "tool_call": raw, "reference": ref, "expect_clarify": p["expect_clarify"],
               "asked_for": sorted(asked), "errors": [],
               "unit_problems": raw.get("unit_problems") or [],
               "warnings": agent.parse_warnings}

        for f, (obj, attr) in _D_FIELDS.items():
            want = ref.get(f)
            got = getattr(getattr(case, obj), attr, None)
            if f in expect:
                total[f"clarify:{f}"] += 1
                ok = got is None and f in asked
                hits[f"clarify:{f}"] += ok
                clarify_ok += ok; clarify_missed += (not ok)
                if not ok:
                    row["errors"].append(
                        f"{f}: text does not determine this; got {got!r}, asked={f in asked}")
                continue
            if want is None:
                continue
            total[f] += 1
            ok = _close(got, want, absolute=0.05 if f == "temperature_c" else None)
            hits[f] += ok
            if not ok:
                row["errors"].append(f"{f}: got {got!r}, want {want!r}")

        name = (case.permeant.name or "").strip().lower()
        if "permeant" in expect:
            total["clarify:permeant"] += 1
            ok = not name
            hits["clarify:permeant"] += ok
            clarify_ok += ok; clarify_missed += (not ok)
            if not ok:
                row["errors"].append(f"permeant: none is named; got {case.permeant.name!r}")
        elif ref.get("permeant_name"):
            total["permeant_name"] += 1
            ok = ref["permeant_name"].lower() in name
            hits["permeant_name"] += ok
            if not ok:
                row["errors"].append(
                    f"permeant: got {case.permeant.name!r}, want {ref['permeant_name']!r}")

        for f, got in (("application", getattr(case.scenario.application, "value", None)),
                       ("test_type", getattr(case.scenario.test_type, "value", None)),
                       ("occlusion", getattr(case.scenario.occlusion, "value", None)),
                       ("dose_regime", getattr(case.scenario.dose_regime, "value", None))):
            want = ref.get(f)
            if f in expect:
                total[f"clarify:{f}"] += 1
                ok = got is None and f in asked
                hits[f"clarify:{f}"] += ok
                clarify_ok += ok; clarify_missed += (not ok)
                if not ok:
                    row["errors"].append(
                        f"{f}: text is contradictory or silent; got {got!r}, asked={f in asked}")
                continue
            if want is None:
                continue
            total[f] += 1
            hits[f] += (got == want)
            if got != want:
                row["errors"].append(f"{f}: got {got!r}, want {want!r}")

        if row["errors"]:
            failures.append(row)
        if records is not None:
            records.append(row)
    return hits, total, clarify_ok, clarify_missed, failures


def run_set_e(args, db, prompts, records=None):
    from e2e_agent.orchestrator import application_needs_confirmation
    asked_ok = asked_missed = control_ok = control_tripped = 0
    rows = []
    for p in prompts:
        agent = _fresh(args, db)
        raw = agent.llm.extract(p["prompt"])
        case = agent.parse(p["prompt"])
        asked = {m[0].split(".")[-1] for m in case.missing()}
        got = getattr(case.scenario.application, "value", None)
        must_ask = p["group"] == "must_ask"
        stopped = "application" in asked and got is None
        ok = stopped if must_ask else (not stopped and got == p["reference"]["application"])
        row = {"set": "set_e", "id": p["id"], "group": p["group"],
               "prompt": p["prompt"], "tool_call": raw,
               "model_said": raw.get("application"), "after_guardrail": got,
               "guardrail_fired": application_needs_confirmation(p["prompt"]),
               "sent_to_clarification": stopped, "pass": ok,
               "reference": p["reference"], "warnings": agent.parse_warnings,
               "question": "", "errors": []}
        full = agent.missing_with_reasons(case)
        if full:
            row["question"] = agent.llm.clarify(
                [(x, d) for x, d in full if x == "scenario.application"] or full, {})
        if must_ask:
            asked_ok += ok; asked_missed += (not ok)
            if not ok:
                row["errors"].append(
                    f"removal is not stated, but application came out as {got!r} "
                    f"without asking")
        else:
            control_ok += ok; control_tripped += (not ok)
            if not ok:
                row["errors"].append(
                    f"removal IS stated, but the rule stopped it "
                    f"(application {got!r}, asked={sorted(asked)})")
        rows.append(row)
        if records is not None:
            records.append(row)
    return asked_ok, asked_missed, control_ok, control_tripped, rows


def _print_confusion(conf, name):
    c = conf[name]
    if not c:
        return []
    classes = sorted({k[0] for k in c})
    correct = sum(v for (w, g), v in c.items() if w == g)
    n = sum(c.values())
    print(f"\n{name}: accuracy {correct}/{n} = {100*correct/n:.1f}%")
    hdr = f"{'reference':<14}" + "".join(f"{g:>14}" for g in classes + ["clarify"])
    print(hdr)
    for w in classes:
        print(f"{w:<14}" + "".join(f"{c[(w, g)]:>14}" for g in classes + ["clarify"]))
    return [correct, n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="api")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    a = json.loads((PROMPTS / "set_a_scenarios.json").read_text())
    b = json.loads((PROMPTS / "set_b_clarification.json").read_text())
    c_path = PROMPTS / "set_c_dose_regime.json"
    c = json.loads(c_path.read_text()) if c_path.exists() else []

    db = Database()
    probe = _fresh(args, db)
    backend_name = getattr(probe.llm, "name", type(probe.llm).__name__)
    print(f"backend: {backend_name}")

    records = []
    conf, hits, total = run_set_a(args, db, a, records=records)
    _, hits_strict, total_strict = run_set_a(args, db, a, strict_temp=True)

    print("\n=== Set A: scenario classification (80 prompts) ===")
    summary_conf = {}
    for d in DESCRIPTORS + ["dose_regime"]:
        got = _print_confusion(conf, d)
        if got:
            summary_conf[d] = got
    if conf["dose_regime"] and len({k[0] for k in conf["dose_regime"]}) == 1:
        print("  note: every set-A prompt describes a finite dose, so this row is "
              "single-class; set C tests the infinite branch.")

    print("\n=== Set A: parameter extraction ===")
    tot_h, tot_n = sum(hits.values()), sum(total.values())
    for f in total:
        print(f"  {f:<16} {hits[f]:>3}/{total[f]:<3}  {100*hits[f]/total[f]:.1f}%")
    print(f"  {'TOTAL':<16} {tot_h:>3}/{tot_n:<3}  {100*tot_h/tot_n:.1f}%  (1 % relative tolerance)")
    ts_h, ts_n = sum(hits_strict.values()), sum(total_strict.values())
    print(f"  {'TOTAL (strict T)':<16} {ts_h:>3}/{ts_n:<3}  {100*ts_h/ts_n:.1f}%  "
          f"(+/-0.05 degC on temperature)")

    flagged, silent, stated_ok, rows_b = run_set_b(args, db, b, records=records)
    print(f"\n=== Set B: clarification loop ({len(b)} prompts, {flagged+silent} missing items) ===")
    print(f"  flagged for clarification : {flagged}/{flagged+silent}")
    print(f"  silently guessed          : {silent}/{flagged+silent}")
    fully = sum(1 for r in rows_b if not r["silent"] and r["stated_fields_ok"])
    print(f"  prompts handled fully correctly: {fully}/{len(b)} ({100*fully/len(b):.0f}%)")

    conf_c = None
    if c:
        conf_c, _, _ = run_set_a(args, db, c, records=records, setname="set_c")
        print(f"\n=== Set C: dose regime ({len(c)} prompts) ===")
        _print_confusion(conf_c, "dose_regime")
    d_path = PROMPTS / "set_d_freetext.json"
    setd = None
    if d_path.exists():
        d = json.loads(d_path.read_text())
        hits_d, total_d, cl_ok, cl_missed, failures = run_set_d(args, db, d, records=records)
        print(f"\n=== Set D: free-text robustness ({len(d)} prompts) ===")
        print("  PROVENANCE: written by the same model that answers them. A high score here "
              "is not\n  evidence of accuracy; the failures are the result worth reporting.")
        det = {k: v for k, v in total_d.items() if not k.startswith("clarify:")}
        dh, dn = sum(hits_d[k] for k in det), sum(det.values())
        print(f"\n  determinate values read correctly : {dh}/{dn} "
              f"({100*dh/dn:.1f}%)" if dn else "")
        for k in sorted(det):
            if hits_d[k] != det[k]:
                print(f"      {k:<16} {hits_d[k]}/{det[k]}")
        print(f"  under-determined values asked for : {cl_ok}/{cl_ok+cl_missed}")
        for k in sorted(total_d):
            if k.startswith("clarify:") and hits_d[k] != total_d[k]:
                print(f"      {k:<22} {hits_d[k]}/{total_d[k]}")
        print(f"\n  prompts with at least one failure : {len(failures)}/{len(d)}")
        for f in failures:
            print(f"    [D{f['id']}]")
            for e in f["errors"]:
                print(f"        {e}")
        setd = {"determinate": [dh, dn], "clarify": [cl_ok, cl_ok + cl_missed],
                "fields": {k: [hits_d[k], total_d[k]] for k in total_d},
                "failures": [{"id": f["id"], "errors": f["errors"]} for f in failures]}

    e_path = PROMPTS / "set_e_unstated_removal.json"
    sete = None
    if e_path.exists():
        e = json.loads(e_path.read_text())
        a_ok, a_miss, c_ok, c_trip, rows_e = run_set_e(args, db, e, records=records)
        print(f"\n=== Set E: unstated removal ({len(e)} prompts) ===")
        print(f"  removal not stated -> asked   : {a_ok}/{a_ok + a_miss}")
        print(f"  removal stated     -> untouched: {c_ok}/{c_ok + c_trip}  (controls)")
        print()
        for r in rows_e:
            mark = "ok  " if r["pass"] else "FAIL"
            print(f"  {mark} E{r['id']} [{r['group']}] model said "
                  f"{str(r['model_said']):<9} -> after guardrail "
                  f"{str(r['after_guardrail']):<9}")
            for err in r["errors"]:
                print(f"        {err}")
        q = next((r["question"] for r in rows_e if r["group"] == "must_ask" and r["question"]), "")
        if q:
            print(f"\n  question put to the user:\n    {q}")
        sete = {"asked": [a_ok, a_ok + a_miss], "controls": [c_ok, c_ok + c_trip],
                "rows": [{k: v for k, v in r.items() if k != "tool_call"} for r in rows_e]}

    n_llm = sum(1 for r in records if r.get("tool_call"))
    print(f"\nprompts scored: {len(records)}")

    jsonl = out / "outputs.jsonl"
    with jsonl.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    (out / "agent_evaluation.json").write_text(json.dumps({
        "backend": backend_name,
        "set_a": {"confusion": {d: {f"{w}->{g}": v for (w, g), v in conf[d].items()}
                                for d in DESCRIPTORS + ["dose_regime"]},
                  "fields": {f: [hits[f], total[f]] for f in total},
                  "fields_strict_temperature": {f: [hits_strict[f], total_strict[f]]
                                                for f in total_strict}},
        "set_b": {"flagged": flagged, "silent": silent, "rows": rows_b},
        "set_c": ({"confusion": {f"{w}->{g}": v for (w, g), v in conf_c["dose_regime"].items()},
                   } if c else None),
        "set_d": setd,
        "set_e": sete,
    }, indent=1, default=str))
    print(f"\nwritten to {out/'agent_evaluation.json'}")
    print(f"per-prompt records: {jsonl}  ({n_llm} records)")


if __name__ == "__main__":
    main()
