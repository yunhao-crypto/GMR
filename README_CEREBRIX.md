# GMR — cerebrix integration notes

This file documents how the vendored GMR is expected to run inside
cerebrix's workflow. For the upstream README, see `README.md`. For
provenance / how this directory was created, see `PROVENANCE.md`.

## TL;DR

* GMR runs in **the same `cerebrix` conda env** as the IsaacLab
  training code — installed editable via the install-guide Step 6
  (``pip install -e thirdparties/GMR/`` plus ``pip install 'numpy<2'``).
  One env, one Python, one set of dependencies.
* The cerebrix training code does **not** import GMR at runtime — the
  boundary is the cerebrix-schema-v1 ``.npz`` files written under
  ``data/motions/``. GMR is invoked offline by
  ``scripts/tools/retarget_amass_to_g1.py`` to populate that directory.
* The vendored layout is editable — modify code under
  ``general_motion_retargeting/`` directly and the changes take effect
  immediately in the active cerebrix env (no re-install needed).

## Setting up GMR in the cerebrix env

Already covered by `docs/install.md` Step 6. For reference, the
recipe is:

```bash
conda activate cerebrix
cd /path/to/cerebrix
pip install --upgrade-strategy only-if-needed -e thirdparties/GMR/
pip install 'numpy<2'   # MUST run — see install.md Step 6 for the
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
  ├── SMPLX_NEUTRAL.npz
  ├── SMPLX_MALE.npz
  ├── SMPLX_FEMALE.npz
  └── ... (template / kid / shape / etc.)
```

`cerebrix/.gitignore` keeps this directory out of git because the
files are subject to the SMPL-X license (research-only, redistribution-
restricted). On a fresh clone:

1. Register at https://smpl-x.is.tue.mpg.de/ and accept the license.
2. Download SMPL-X v1.1 (NPZ format).
3. Drop the unpacked `.npz` files into `assets/body_models/smplx/`.

## Driver script — cerebrix → GMR → cerebrix-schema-v1 npz

The thin driver lives **outside** this directory at:

```
cerebrix/scripts/tools/retarget_amass_to_g1.py
```

Typical invocation (run inside the cerebrix env, same env that runs
training):

```bash
conda activate cerebrix

# AMASS / SMPL-X input → cerebrix npz output
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
  --input-format bvh
```

Once the `.npz` files land under `data/motions/g1/<corpus>/`, point
the G1 tracking task at them via env var:

```bash
export CEREBRIX_G1_TRACKING_DATASET=accad_amass
python scripts/train.py --task Cerebrix-Tracking-Unitree-G1-v0 ...
```

## License summary

* GMR — MIT (Yanjie Ze, see `LICENSE`)
* cerebrix integration glue (`PROVENANCE.md`, `README_CEREBRIX.md`,
  `scripts/tools/retarget_amass_to_g1.py`) — Apache-2.0 (cerebrix
  contributors)
* SMPL-X body models — research-only, NOT redistributable; users must
  obtain their own copy from <https://smpl-x.is.tue.mpg.de/>. The
  files under `assets/body_models/smplx/` (and the `body_models/`
  directory in general) are gitignored to enforce this.
* AMASS motion corpora — same license as SMPL-X. `data/raw/amass/` is
  gitignored.
