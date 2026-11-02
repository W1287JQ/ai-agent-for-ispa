## 1. Software environment

All reported results were produced with the following components and versions.

| Component | Used |
|---|---|
| Python | 3.13.16 |
| RDKit | 2026.9.1 |
| NumPy | 2.5.3 |
| pandas | 3.0.5 |
| SciPy | 1.18.1 |
| pydantic | 2.13.5 |
| Database | SQLite 3.45.1 (embedded; Python `sqlite3` module 2.6.0) |
| Fuzzy name matching | `difflib` (Python standard library); no vector index |
| pKa predictor | Dimorphite-DL 2.1.0 (Ropp et al., *J Cheminform* 2019;11:14), 65 substructure classes |
| Plotting | Matplotlib 3.11.2 |
| Orchestration | none; hand-written Python state machine (`agent_code/orchestrator.py`) |
| Service layer | FastAPI 0.151.0 |
| Report output | openpyxl 3.1.5, python-docx 1.2.0 |


## 2. Mechanistic model

The mechanistic model is used unchanged from its published implementations: the
end-to-end workflow and PBPK engine of [38]
(https://github.com/weightking/E2E-PERCUTANEOUS-ABSORPTION) and the evaporation
sub-module of [37] (https://github.com/bendeacon/E2E_Evap_model).

Reference numbers follow the manuscript:

* [37] Deacon BN, Silva S, Lian G, Evans M, Chen T. Computational modelling of
  the impact of evaporation on in-vitro dermal absorption. *Pharm Res*
  2024;41(10):1979-90.
* [38] Zhang D, Deacon BN, Li W, Lian G, Chen T. A computational workflow for
  end-to-end simulation of percutaneous absorption. *Int J Pharm*
  2025;670:125084.
