from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import rdkit
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
ESOL_REFERENCE = ROOT / "data" / "domain_reference" / "esol_delaney.smi"

SIM_OUT_OF_DOMAIN = 0.273
SIM_LOW_CONFIDENCE = 0.333

IN_DOMAIN = "in-domain"
LOW_CONFIDENCE = "low-confidence"
OUT_OF_DOMAIN = "out-of-domain"
_RANK = {IN_DOMAIN: 0, LOW_CONFIDENCE: 1, OUT_OF_DOMAIN: 2}

CRIPPEN_GENERIC_TYPES = {"Me1", "Me2"}


@dataclass
class ToolDomain:
    tool: str
    verdict: str
    reason: str
    max_similarity: Optional[float] = None
    nearest_reference: Optional[str] = None
    reference: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class DomainReport:
    smiles: str
    tools: List[ToolDomain] = field(default_factory=list)
    reference_available: bool = True

    @property
    def verdict(self) -> str:
        if not self.tools:
            return IN_DOMAIN
        return max((t.verdict for t in self.tools), key=lambda v: _RANK[v])

    def flagged(self) -> List[ToolDomain]:
        return [t for t in self.tools if t.verdict != IN_DOMAIN]

    def to_dict(self):
        return {"smiles": self.smiles, "verdict": self.verdict,
                "reference_available": self.reference_available,
                "tools": [t.to_dict() for t in self.tools]}

    def render(self) -> str:
        lines = [f"applicability domain: {self.verdict}"]
        for t in self.tools:
            sim = f"  max Tanimoto {t.max_similarity:.3f}" if t.max_similarity is not None else ""
            lines.append(f"  {t.tool:<14} {t.verdict:<15} {t.reason}{sim}")
        return "\n".join(lines)

    def question(self) -> str:
        bad = self.flagged()
        bits = "; ".join(f"{t.tool}: {t.reason}" for t in bad)
        return (f"This permeant looks {self.verdict} for the Step 1 property estimators "
                f"({bits}). Their output may be unreliable here. Do you want to continue "
                f"with the estimated properties anyway, or supply measured logP / "
                f"solubility / pKa values instead? (continue / I'll supply values)")


@lru_cache(maxsize=1)
def _reference():
    if not ESOL_REFERENCE.exists():
        return None
    smis = [l.strip() for l in ESOL_REFERENCE.read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, mws, rbs = [], [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(gen.GetFingerprint(m))
        mws.append(Descriptors.MolWt(m))
        rbs.append(Lipinski.NumRotatableBonds(m))
    return smis, fps, (min(mws), max(mws)), (min(rbs), max(rbs))


@lru_cache(maxsize=1)
def _crippen_table():
    path = Path(os.path.dirname(rdkit.__file__)) / "Data" / "Crippen.txt"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        patt = Chem.MolFromSmarts(parts[1].strip())
        if patt is not None:
            rows.append((parts[0].strip(), patt))
    return rows


def crippen_atom_types(mol) -> List[Optional[str]]:
    types: List[Optional[str]] = [None] * mol.GetNumAtoms()
    for label, patt in _crippen_table():
        for match in mol.GetSubstructMatches(patt):
            idx = match[0]
            if types[idx] is None:
                types[idx] = label
    return types


def _similarity(mol):
    ref = _reference()
    if ref is None:
        return None, None
    smis, fps, _mw, _rb = ref
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sims = DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(mol), fps)
    best = max(range(len(sims)), key=lambda i: sims[i])
    return sims[best], smis[best]


def check(smiles: str, sites: Optional[Sequence] = None) -> DomainReport:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")
    rep = DomainReport(smiles=smiles, reference_available=_reference() is not None)
    sim, nearest = _similarity(mol)
    mw, rb = Descriptors.MolWt(mol), Lipinski.NumRotatableBonds(mol)

    if sim is None:
        rep.tools.append(ToolDomain(
            "ESOL", LOW_CONFIDENCE,
            "no reference set available, so domain membership could not be tested",
            reference="Delaney 2004"))
    else:
        _s, _f, (mw_lo, mw_hi), (rb_lo, rb_hi) = _reference()
        why = []
        if sim < SIM_OUT_OF_DOMAIN:
            verdict = OUT_OF_DOMAIN
            why.append(f"nearest training compound is only {sim:.2f} similar "
                       f"(below the {SIM_OUT_OF_DOMAIN:.2f} 5th-percentile of the set's own "
                       f"nearest-neighbour similarity)")
        elif sim < SIM_LOW_CONFIDENCE:
            verdict = LOW_CONFIDENCE
            why.append(f"nearest training compound only {sim:.2f} similar")
        else:
            verdict = IN_DOMAIN
            why.append(f"nearest training compound {sim:.2f} similar")
        if not (mw_lo <= mw <= mw_hi):
            verdict = OUT_OF_DOMAIN
            why.append(f"MW {mw:.0f} outside the training range {mw_lo:.0f}-{mw_hi:.0f}")
        if not (rb_lo <= rb <= rb_hi):
            verdict = max(verdict, LOW_CONFIDENCE, key=lambda v: _RANK[v])
            why.append(f"{rb} rotatable bonds outside the training range {rb_lo}-{rb_hi}")
        rep.tools.append(ToolDomain("ESOL", verdict, "; ".join(why), sim, nearest,
                                    "Delaney 2004 training set (1117 compounds)"))

    types = crippen_atom_types(mol)
    unmatched = [mol.GetAtomWithIdx(i).GetSymbol() for i, t in enumerate(types) if t is None]
    generic = [mol.GetAtomWithIdx(i).GetSymbol() for i, t in enumerate(types)
               if t in CRIPPEN_GENERIC_TYPES]
    if unmatched:
        v, why = OUT_OF_DOMAIN, (f"{len(unmatched)} atom(s) match no atom type in the published "
                                 f"table ({', '.join(sorted(set(unmatched)))})")
    elif generic:
        v, why = (LOW_CONFIDENCE,
                  f"{len(generic)} atom(s) ({', '.join(sorted(set(generic)))}) fall into the "
                  f"table's generic metal buckets, which carry a lumped placeholder rather than "
                  f"a fitted contribution")
    elif sim is not None and sim < SIM_OUT_OF_DOMAIN:
        v, why = (LOW_CONFIDENCE,
                  "all atom types are parameterised, but the molecule is far from the chemical "
                  "space of the available reference set (the Crippen fitting set is not public, "
                  "so this is a proxy)")
    else:
        v, why = IN_DOMAIN, "every atom matches a fitted atom type in the published table"
    rep.tools.append(ToolDomain("Crippen logP", v, why, sim, nearest,
                                "Wildman & Crippen 1999 atom-type table"))

    if sites is not None:
        from .step1_properties import PKA_WINDOW, ionisable_groups_without_pka
        missed = ionisable_groups_without_pka(mol, sites)
        in_window = [s for s in sites if PKA_WINDOW[0] <= s.pka <= PKA_WINDOW[1]]
        if missed:
            v, why = (OUT_OF_DOMAIN,
                      f"{len(missed)} ionisable group(s) match no class in the substructure table "
                      f"({', '.join(sorted(set(missed)))}), so they get no pKa and the molecule is "
                      f"treated as less ionisable than it is")
        elif sites and not in_window:
            v, why = (LOW_CONFIDENCE,
                      "every predicted pKa lies outside the pH 1-13 window, so Sw and Dow carry no "
                      "pH dependence at all")
        else:
            v, why = IN_DOMAIN, f"{len(in_window)} ionisable group(s) covered by the table"
        rep.tools.append(ToolDomain("pKa predictor", v, why,
                                    reference="Dimorphite-DL substructure table"))
    return rep


def _calibrate():
    import numpy as np
    ref = _reference()
    if ref is None:
        raise SystemExit(f"no reference set at {ESOL_REFERENCE}")
    _s, fps, _mw, _rb = ref
    loo = []
    for i, fp in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        sims[i] = -1.0
        loo.append(max(sims))
    loo = np.array(loo)
    print(f"reference set: {len(fps)} compounds")
    for q in (1, 5, 10, 25, 50):
        print(f"  {q:>2}th percentile leave-one-out nearest-neighbour Tanimoto: "
              f"{np.percentile(loo, q):.3f}")
    print(f"\ncurrent thresholds: out-of-domain < {SIM_OUT_OF_DOMAIN}, "
          f"low-confidence < {SIM_LOW_CONFIDENCE}")


if __name__ == "__main__":
    import sys
    if "--calibrate" in sys.argv:
        _calibrate()
    else:
        for s in sys.argv[1:]:
            print(check(s).render())
