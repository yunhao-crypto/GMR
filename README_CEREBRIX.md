# GMR — cerebrix integration notes

This file documents how the vendored GMR is expected to run inside
cerebrix's workflow. For the upstream README, see `README.md`. For
provenance / how this fork relates to upstream, see `PROVENANCE.md`.

## TL;DR

* GMR lives at `cerebrix/thirdparties/GMR/` as a **git submodule**
  pointing at this repo (`yunhao-crypto/GMR`, branch `main`).
  cerebrix users initialize it via
  `git submodule update --init --recursive` (or pass
  `--recurse-submodules` to the initial `git clone`).
* GMR runs in **the same `cerebrix` conda env** as the IsaacLab training
  code — installed editable via `cerebrix/docs/install.md` Step 6
  (`pip install -e thirdparties/GMR/` plus `pip install 'numpy<2'`).
  One env, one Python, one set of dependencies.
* The cerebrix training code does **not** import GMR at runtime — the
  boundary is the **InstinctLab-retargetted** ``.npz`` files written
  under `cerebrix/data/motions/`. GMR is invoked offline by
  `cerebrix/scripts/tools/retarget_amass_to_g1.py` to populate that
  directory.
  > Note: cerebrix has a separate single-clip dev format —
  > *cerebrix-schema-v1* — consumed by `NpzMotionBuffer`, **not**
  > produced by GMR. See `cerebrix/docs/extending/motion_reference.md`.
* Editable install: edit code under `general_motion_retargeting/`
  directly and the changes take effect immediately in the active
  cerebrix env (no re-install needed).

## Setting up GMR in the cerebrix env

Already covered by `cerebrix/docs/install.md` Step 6. For reference,
the recipe after `git submodule update --init` is:

```bash
conda activate cerebrix
cd /path/to/cerebrix
pip install --upgrade-strategy only-if-needed -e thirdparties/GMR/
pip install 'numpy<2'   # MUST run — see install.md Step 6.2 for the
                        # pip-quirk explanation
```

Verify:

```bash
python -c "from general_motion_retargeting import GeneralMotionRetargeting; print('GMR OK')"
python -c "import isaaclab, isaaclab_rl; print('IsaacLab still healthy')"
```

## SMPL-X body models

GMR's SMPL-X retargeting paths require the SMPL-X body models under
`assets/body_models/smplx/`:

```
assets/body_models/smplx/
  ├── models_smplx_v1_1.zip   # canonical MPI distribution form (gitignored)
  ├── SMPLX_NEUTRAL.npz       # extracted, used by GMR
  ├── SMPLX_MALE.npz          # extracted, used by GMR
  └── SMPLX_FEMALE.npz        # extracted, used by GMR
```

> ⚠️ These files are research-only, **NOT redistributable** under the
> Max Planck SMPL-X license. **GMR's own `.gitignore`** in this repo
> (the `body_models/` and `assets/body_models/` rules) keeps them out
> of any git history so they can never accidentally land in
> `yunhao-crypto/GMR` or in cerebrix.

On a fresh clone, each developer:

1. Registers at <https://smpl-x.is.tue.mpg.de/> and accepts the
   research-only license.
2. Downloads SMPL-X v1.1 (NPZ format) — yields
   `models_smplx_v1_1.zip` (~870 MB).
3. Drops the zip at `assets/body_models/smplx/` and extracts the 3
   `.npz` files:
   ```bash
   cd assets/body_models/smplx
   unzip -jo models_smplx_v1_1.zip "models/smplx/SMPLX_*.npz"
   ```

See `cerebrix/docs/install.md` Step 6.3 for the full
register-download-extract-verify procedure (including the
`git check-ignore` sanity check that confirms the files cannot
accidentally be committed).

You can skip this entire section if your retargeting workflow only
uses BVH inputs (LAFAN1 / NoKov) — those don't go through SMPL-X.

## Driver script — cerebrix → GMR → InstinctLab-retargetted npz

The thin driver lives **outside** this directory at:

```
cerebrix/scripts/tools/retarget_amass_to_g1.py
```

Typical invocation (run inside the cerebrix env, same env that runs
training):

```bash
conda activate cerebrix

# AMASS / SMPL-X input → InstinctLab-retargetted npz output
python cerebrix/scripts/tools/retarget_amass_to_g1.py \
  --input  cerebrix/data/raw/amass/ACCAD \
  --output cerebrix/data/motions/g1/accad_amass \
  --robot  unitree_g1 \
  --input-format smplx \
  --num-workers 4

# LAFAN1 / BVH input
python cerebrix/scripts/tools/retarget_amass_to_g1.py \
  --input  cerebrix/data/raw/lafan1 \
  --output cerebrix/data/motions/g1/lafan1 \
  --robot  unitree_g1 \
  --input-format bvh \
  --bvh-format lafan1
```

Once the `.npz` files land under `data/motions/g1/<corpus>/`, point
the G1 tracking task at them via env var:

```bash
export CEREBRIX_G1_TRACKING_DATASET=accad_amass
python cerebrix/scripts/reinforcement_learning/instinct_rl/train.py \
    --task Cerebrix-Tracking-Unitree-G1-v0 ...
```

See `cerebrix/docs/motion_pipeline.md` for the end-to-end Stage 1
(data acquisition) → Stage 2 (retargeting) → Stage 3 (training)
runbook.

## License summary

* Upstream GMR source — MIT (Yanjie Ze, see `LICENSE`)
* HoloMotion-side patches (commit `20a3f9c` on `main`) — MIT
  (HoloMotion team)
* cerebrix-side glue (`PROVENANCE.md`, `README_CEREBRIX.md`) —
  Apache-2.0 (cerebrix contributors)
* SMPL-X body models — research-only, NOT redistributable; users
  obtain their own copy from <https://smpl-x.is.tue.mpg.de/>. The
  files under `assets/body_models/smplx/` (and `body_models/` in
  general) are gitignored to enforce this.
* AMASS motion corpora — same license terms as SMPL-X.
  `cerebrix/data/raw/amass/` is gitignored at the cerebrix level.
