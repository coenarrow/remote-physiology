# Overhaul Roadmap: rPPG-Toolbox → remote-physiology

Companion to `updating_plan.md` (the *what*); this document is the *order and
the mechanics*, plus the repo cleanup that plan doesn't cover. Phases are
sequenced by dependency: each one makes the next one smaller.

Ordering note: config work is **interleaved with model migration**, not done
up front. A couple of models migrate first against the current config, each
migration ends with a short config retro, and the consolidation happens once
real usage has shaped the schema (Phase 5).

## Cross-cutting rules

- **Dependencies go through `uv add`, never pip.** `pyproject.toml` + `uv.lock`
  are the single source of truth, and every addition considers all three
  platforms (Windows dev, Linux HPC, macOS dev/demos) — see the mamba-ssm
  platform split for what happens when a package doesn't.
- **All tensor reshaping uses einops** (`rearrange` / `reduce` / `einsum`),
  not `view` / `permute` / `reshape` — the shape spelled out at the call site
  is the point. Applies to migrated model code too, even where the original
  architecture used raw reshapes.
- **Testing stays minimal.** This is research code that will be open sourced,
  not production code for consumers. Specs and implementation plans should not
  demand exhaustive test suites: the existing contract tests plus one smoke
  test per migration is the ceiling, and errors get fixed as they appear.
- **The old base classes go, not get adapted.** `BaseLoader` (preprocessing
  machinery) and `BaseTrainer` are artifacts of the old contracts;
  `BaseZarrDataset` and `MultiSignalTrainer` replace them outright. Delete
  rather than maintain compatibility shims.
- Git history is the archive. Anything deleted (loaders, configs, weights,
  scratch) is one checkout of the `pre-overhaul` tag away.

---

## Phase 0 — Land what's in flight

The entire dict-contract changeset (~30 modified + ~30 untracked files,
including `vendor/`, `MultiSignalTrainer`, `batch.py`, the test suite) is
uncommitted. Nothing else happens until it's landed in a few logical commits
(vendoring/pyproject, contract core, tests, tools, docs). Include
`updating_plan.md` and this roadmap.

Then tag the result `pre-overhaul` so everything deleted later is one checkout
away.

## Phase 1 — Rename + repo hygiene

Cheap, mechanical, and best done *before* the overhaul so later diffs are
reviewable.

**Rename** to `remote-physiology` now, not at the end — GitHub redirects old
URLs, and every doc written in Phases 2–8 then carries the right name.

**Root scratch files** (all tracked, all deletable — git history keeps them):

| File | Action |
| --- | --- |
| `debug` | delete (captured debug output) |
| `test.py`, `plt_attention.py` | delete (scratch) |
| `pytorch_learning.py` | delete (confirmed: no relocation needed) |
| `metrics.ipynb`, `neckflix_metrics.ipynb`, `neckflix_metrics.py`, `bp_metrics.py`, `neckflix_example_metrics.csv` | park in `evaluation/prototypes/` — raw material for Phase 7; delete when Phase 7 consumes them |
| `requirements.txt`, `setup.sh` | delete — `pyproject.toml` is the source of truth |

**Tracked outputs**:

- `final_model_release/` — delete from the tree (decided). The 36 upstream
  single-signal `.pth` weights can't load into multi-signal variants and
  upstream still hosts them.
- `model_outputs/PURE_PURE_UBFC_deepphys_outputs.pickle` — untrack, ignore dir.
- `figures/` — deleted entirely (decision 9): the interim README carries no
  images, so nothing references it; Phase 8 pulls images back from history if
  the final README wants any.

**`.gitignore` fixes**:

- Remove `uv.lock` from `.gitignore` — it's tracked (and must be, as the
  reproducible-env artifact); the ignore entry is a no-op lie.
- Start tracking `.slurm_scripts/` (decided). These are **reference
  material** — templates to copy and adapt per experiment, not a maintained
  API — and should be treated as such in docs.
- Add `model_outputs/`; keep `runs/`, `logs/` ignored.

## Phase 2 — Cache contract + delete the legacy pipeline

The `updating_plan.md` "Caching" stage, plus its forced consequences.

1. **Document the cache contract** in README + `docs/architecture.md`: one zarr
   store per recording, `perspective → stream → video/frames + trace/data`,
   root attrs (`complete`, `tool_version`, participant, …), admission rules.
   Neckflix as the worked example.
2. **Replace each legacy loader with a markdown cache spec.** Twelve loaders
   (BP4D+, BP4D+BigSmall, COHFACE, LADH, MMPD, PhysDrive, PURE, SCAMPS, SUMS,
   UBFC-PHYS, UBFC-rPPG, iBVP). For each: read the loader, capture into
   `dataset/data_loader/<NAME>.md` everything a future cache-writer needs —
   raw file layout, video/trace formats, sampling rates, quirks (e.g. PURE's
   image sequences, SCAMPS' mat files) — *then* delete the `.py`. This is the
   only record of those parse details once the code is gone; the spec is the
   deliverable, the deletion is the afterthought.
3. **Consequences** (legacy loaders are load-bearing for the old pipeline):
   - `main.py` dies with the loaders → `neckflix_main.py` generalizes into the
     single entry point (renamed `main.py` at the end of this phase).
   - `BaseLoader.py` goes entirely (face crop, chunking, `.npy` cache, file
     lists) — replaced by `BaseZarrDataset`, no shim.
   - Legacy tools tied to the `.npy` cache (`tools/preprocessing_viz/`,
     `tools/output_signal_viz/`, `tools/motion_analysis/`) — audit, then
     delete.
4. **CLAUDE.md note**: "implementing a cache/loader for a new dataset → read
   the markdown spec in `dataset/data_loader/`."

Keep: `zarr_dataset.py`, `NeckflixLoader.py`, `neckflix_config.py`,
`label_transforms.py`, the unsupervised methods (already migrated).

## Phase 3 — Dataset & dataloading generalization

Mostly landed for Neckflix. Remaining work:

- Promote the Neckflix filter model to `BaseZarrDataset` generically:
  attribute include-filters driven by whatever attrs a store carries, not
  hardcoded posture/perspective/light.
- A new dataset = a `channel_map` subclass + a markdown cache spec.
- Standard dict keys stay owned by `neural_methods/batch.py`.

## Phase 3.5 — Dependency refresh

Deliberately slotted *here*: before this point a bump would break code
scheduled for deletion; after the migrations every model would need
revalidating against new APIs. This is the window where the living surface is
smallest and everything that survives has contract tests.

> **⛔ HARD PAUSE before starting this phase.** Check the system requirements
> on all three platforms before touching anything: Windows dev box (CUDA
> toolkit / MSVC for the vendored `mamba-ssm` build, `triton-windows`
> compatibility), Linux HPC (available CUDA modules and driver versions on the
> cluster — these cap the torch version), and macOS (`mamba-ssm-macos`, MPS
> support). No upgrade target is chosen until the constraint set from all
> three is known. This is a stop-and-review point, not a step in a batch.

Then, in two separate commits:

1. **General pass** — `uv lock --upgrade` for the uncoupled deps (zarr,
   numpy, scipy, einops, plotting, …); pyproject edits for any deliberate
   major bumps. Run the contract tests + one smoke run.
2. **The torch / mamba-ssm / triton cluster** — coupled: the vendored
   `mamba-ssm` compiles against a specific torch, `triton-windows` has its own
   torch matrix, and the HPC CUDA modules cap torch from the other side. One
   commit, verified with a smoke train run on **both** Windows and HPC before
   anything builds on it. If the cluster's modules aren't ready, defer this
   half — it must not block Phase 4.

## Phase 4 — Model migration, wave 1 (+ config retros)

Migrate the easiest models first, against the config system as it stands.
Each migration: `DictModel` with `forward_video(video) -> (B, S, T)`, a
`MODEL_REGISTRY` builder line, a config, a smoke test — and **delete the
legacy trainer** as each model moves onto `MultiSignalTrainer`.

Wave 1 = the 2-D per-frame backbones (near-mechanical via
`SignalDictWrapper(input_mode='frames2d')`):

1. DeepPhys
2. TS-CAN
3. EfficientPhys

**After each migration, a config retro**: note what was awkward to express,
what was duplicated, what the yacs tree forced. These notes drive Phase 5 —
the schema gets designed from friction observed, not speculation.

## Phase 5 — Config consolidation

With wave 1's usage in hand:

- Delete the dead bulk of `config.py` (712 lines of yacs defaults: face
  detection, `BEGIN`/`END`, `DO_PREPROCESS`, per-dataset blocks — most already
  dead after Phase 2).
- Replace with the typed pattern prototyped in
  `dataset/data_loader/neckflix_config.py`: dataclass-style schema, validated
  on load, dataset-specific keys in one namespaced block.
- One config system: kill the Hydra split (`physhydra_configs/`), delete the
  legacy `configs/train_configs/` + `configs/infer_configs/` piles;
  `configs/` holds only current-format experiment files.
- Entry-point loading rebuilt: config → typed object → passed down, no global
  CN mutation.
- Re-point the wave-1 configs at the new schema (three files — the cost of
  migrating models first, accepted deliberately).

## Phase 6 — Model migration, wave 2

The remaining architectures, written directly against the new config, in
rough order of difficulty. Per `updating_plan.md`, the multi-signal head
design is decided collaboratively per model, staying true to each original
architecture:

1. **3-D conv** — PhysNet, iBVPNet, FactorizePhys: output head widens to S
   signals.
2. **Transformers** — PhysFormer, RhythmFormer: head + tokenization decisions.
3. **Special cases** — BigSmall (already multi-task; map its task heads onto
   the signal dict) and PhysHydra (our own model, still on the legacy tuple
   contract).

Standardized plots are implemented **once** in `MultiSignalTrainer` /
`metrics_report.py`, so cross-model consistency is free.

Exit criterion: `neural_methods/trainer/` contains `MultiSignalTrainer` and
nothing else — `BaseTrainer` is deleted with the last legacy trainer, not
kept as a parent.

## Phase 7 — Evaluation & clinical metrics

Can overlap Phases 4–6 — it consumes saved outputs through the stable batch
contract.

1. **Design doc first**: map IEEE 1708-2014 / 1708a-2019, ISO 81060-2:2018 /
   81060-3:2022, and ESH 2023 onto computable metrics — per-beat
   systolic/diastolic detection, mean-error/SD acceptance bands, per-subject
   vs pooled aggregation, grading. Be explicit about which criteria are
   *computable from our data* vs *study-design requirements* (subject counts,
   reference-device protocol, cuff procedure) that a metrics report can note
   but not satisfy.
2. Extend `evaluation/metrics_report.py`; denormalised (mmHg) reporting via
   the `label_stats` already carried in every batch.
3. Consume, then delete, `evaluation/prototypes/` from Phase 1.

## Phase 8 — Docs finalization

- README rewritten for `remote-physiology`: mission, cache contract,
  batch-dict contract, model table, clinical-metrics summary; any images it
  wants come back from git history (`figures/` was deleted in Phase 1);
  upstream rPPG-Toolbox credited as the fork origin. An interim
  accuracy-pass README/CLAUDE.md landed during Phase 1 — this phase is the
  final polish, not the first correction.
- CLAUDE.md refreshed to remove legacy-pipeline instructions.
- `docs/architecture.md`, `docs/changelog.md`, `docs/project_status.md`
  updated; `updating_plan.md` retired into `docs/plans/`.

---

## Decision log (2026-08-31)

1. Repo renamed to **remote-physiology** (executes in Phase 1).
2. `final_model_release/` deleted from the tree in Phase 1.
3. `.slurm_scripts/` tracked, as reference material only.
4. `pytorch_learning.py` deleted, no relocation.
5. Config consolidation happens **after** wave-1 model migrations, informed
   by per-migration retros — not up front.
6. Testing kept minimal throughout (see cross-cutting rules).
7. `BaseLoader` and `BaseTrainer` are deleted, not adapted — the new
   contracts (`BaseZarrDataset`, `MultiSignalTrainer`) replace them.
8. Dependency refresh happens between Phases 3 and 4 (Phase 3.5), preceded by
   a hard pause to check system requirements on all three platforms; the
   torch/mamba-ssm/triton cluster is its own commit and may be deferred.
9. `figures/` deleted entirely in Phase 1 (not pruned in Phase 8): the
   interim README carries no images. Executed superpowers plans/specs and the
   2026-02 UBFC validation plan also deleted — process artifacts, retrievable
   from git history; `docs/architecture.md` is the living contract reference.
10. Phase 3 filter design: one generic `NECKFLIX.FILTERS` yacs node
    (`new_allowed`, so YAML can name any attr) keyed by store root attrs plus
    the `perspective` pseudo-attr; the fixed
    `POSTURES`/`PERSPECTIVES`/`LIGHT`/`SESSIONS` keys are deleted, not
    aliased. Participants remain a separate surface (`PARTICIPANTS` /
    `--test_participants`) because their ids are normalised (`P015` → `015`);
    a `participant` key inside `FILTERS` is refused rather than allowed to
    bypass that normalisation. Experiment names derive their filter segment
    generically from `FILTERS`.
11. Phase 3.5 executed from the three platform reports
    (`docs/plans/platform-reports/`, generated by
    `tools/platform_report.py`): **torch 2.12.1 + cu126 everywhere** — the
    V100 partition is sm_70 (dropped from cu128+/cu130), the cluster driver
    (575.57) caps CUDA at 12.9, and triton-windows (3.8 = torch 2.12) rules
    out 2.13 on Windows. mamba-ssm held at 2.3.1, pinned exact on all
    platforms (2.3.2+ drags tilelang/quack-kernels, Linux-only). Driver
    uniformity across GPU nodes assumed (only k177 was sampled); whether
    triton 3.8 still JITs for sm_70 is verified by the first V100 smoke run,
    with A100/H100 as the fallback partitions. pyproject audited to a
    minimal floor-pinned set: exact pins only where blocking, dev tooling in
    the `dev` group (`uv sync --no-dev` on the HPC), orphans dropped
    (pyqt5, opencv, thop, tensorboardX, scikit-image, neurokit2, pdf/crypto
    tools).

Last updated: 2026-08-31
