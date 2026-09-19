# remote-physiology (fork of rPPG-Toolbox)

## Project Mission

Camera-based estimation of physiological signals — not just heart rate. Every
model predicts **multiple signals at once**: BVP, arterial and central venous
pressure waveforms (ABP, CVP), respiration, and others as they appear. Primary
dataset: the multimodal Neckflix dataset (RGB/IR/Depth video + ABP/CVP/ECG
traces). Validation includes clinical BP standards (IEEE 1708, ISO 81060,
ESH 2023) and large-scale LOSO sweeps on an HPC cluster.

## Design Principle

**Extending the repo should be cheap, because everything shared is written
once.** A new dataset is a `channel_map` subclass plus a markdown cache spec;
a new model is a backbone `nn.Module`, a registry line in
`src/model_config.py` (plus a config class only if it has a switch), a
builder in `src/models.py` and one config YAML under
`configs/original_model_config/`
(`docs/adding_a_model.md` is the recipe) — never a new trainer, loader, loss
module, or plot set. When new work needs something a
shared piece almost does, extend the shared piece for everyone rather than
writing a parallel copy beside it; a second implementation of anything is a
bug in the first one's design. This principle is why the per-model and
per-dataset recipes below are short — keep them that way.

## Overhaul In Progress

The repo is mid-overhaul from the upstream single-signal design to the
multi-signal contract.

## Cross-Cutting Rules

- **Dependencies go through `uv add`, never pip.** `pyproject.toml` + `uv.lock`
  are the single source of truth (`requirements.txt` and `setup.sh` are gone).
  Every addition considers all three platforms: Windows dev, Linux HPC, macOS.
- **All tensor reshaping uses einops** (`rearrange` / `reduce` / `einsum`),
  not `view` / `permute` / `reshape` — including migrated model code.
- **A single-trace design is copied per trace; a multi-trace design is
  built once.** An architecture designed to predict one trace (every
  upstream rPPG-Toolbox model) gets one complete copy per entry of `TRACES`
  and never a widened readout or per-signal heads on a shared trunk. An
  architecture designed from the start to predict several traces
  (CardioConv) is one network returning `(B, S, T)` in `TRACES` order. Which
  one a model is follows from its design, not from convenience, and nothing
  downstream of the model can tell them apart. `docs/adding_a_model.md`,
  "What a model is here".
- **A model is a package, one `nn.Module` per file, named by PEP 8.**
  `neural_methods/model/<name>/<name>.py` holds the top-level class and the
  structural constants, which sub-modules receive as constructor arguments;
  every other module has its own snake_case file, helper functions go in
  `<name>/utils.py`, and anything two models need lives in
  `neural_methods/model/_shared_modules/`. Names are PEP 8 with no exemption
  for upstream spellings (CapWords classes, snake_case attributes and
  arguments, `UPPER_CASE` constants), even where that renames a `state_dict`
  key; `uv run ruff check` is the check. `neural_methods/model/factorizephys/`
  is the reference.
- **Tests are a cost, not a safety net.** This is research code; the suite
  exists to catch a broken build, not to specify behaviour. Write a test only
  when it is a unit test of a pure function that fits in a dozen lines with
  no fixtures; a model migration does not get one. Never add tests to a
  refactor or a design change, never test a test helper, and never add a
  test "for coverage". The verification
  for a change is the run command in `README.md`, not a new test. This
  overrides the test-driven-development and verification skills' defaults.
- **Stale tests are deleted in the same change that stales them.** A test
  that imports a module, name, or fixture argument that no longer exists, or
  that asserts behaviour the current design has replaced, is deleted, not
  fixed, skipped, marked xfail, or left failing. The design is the spec; the
  old tests are not. After a design change, run only the tests that cover
  the files you touched. If a test elsewhere fails, the default is that the
  test is stale: delete it and name it in the summary so a human can veto,
  rather than adapting the code back to it. A collection error from
  `uv run pytest --collect-only -q` names a file to delete, not a file to
  fix.
- **Legacy code is deleted, not adapted.** No compatibility shims; git history
  and the `pre-overhaul` tag are the archive.
- **Every model accepts any frame size and any window length.** Structural
  constants of the upstream code (patch sizes, sequence lengths, fixed token
  grids) become constructor arguments derived from the interface, config
  switches, or adaptive stages around the published network that are exact
  no-ops at the paper's shape — never hard-coded refusals, and never a silent
  crop or truncation. The only refusal left is a frame the stem pools to
  nothing, named by the builder.
- **`configs/original_model_config/<name>_<interface>.yaml` is the paper.**
  That directory is what the per-model config files are for: when migrating
  a model, its file (named after the architecture and the interface it
  encodes, e.g. `deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml`) *is* the
  rPPG-Toolbox configuration of it. Its `INTERFACE` section is the paper's
  rate, window, resize, the single PPG trace the paper predicts and its
  loss; its `TRAIN` section is the paper's optimiser, rate, decay, schedule
  and precision, read off the upstream `train_configs/` file *and* the
  upstream trainer class (the optimiser and schedule live there, not in the
  YAML); the paper's epochs and batch size are the README command's
  `--epochs` and `--batch-size`. Nothing Neckflix-specific goes in it.
  Nothing in code declares or checks the paper setup; the files do. Model
  comparisons run every model on the same standard interface
  (`configs/combined_model_config/`); the paper files are where a migration
  is checked against the paper.
- **A finished migration ends with a command in `README.md`.** When a model
  migrated from rPPG-Toolbox is done, add under "Algorithms" the exact
  `scripts/run.py` command that trains it on the PURE dataset
  (`--datasets pure`) with only the first participant held out
  (`--test-participant-dataset pure --test-participant-id 01`), on its paper
  interface and paper training recipe (the one command also runs the
  model over that participant and scores the records).
  That command is the migration's proof of life; a model without one is
  not finished.
