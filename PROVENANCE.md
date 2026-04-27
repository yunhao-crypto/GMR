# Vendored GMR provenance

This directory is a vendored copy of [General Motion Retargeting (GMR)](https://github.com/YanjieZe/GMR)
by Yanjie Ze, licensed under the MIT License (see `LICENSE`).

## Source

| Field | Value |
|---|---|
| Upstream repo | https://github.com/YanjieZe/GMR.git |
| Upstream commit | `5cd312e501db070faad588ce4e510ce1771f7410` |
| Vendored from | `HoloMotion/thirdparties/GMR/` (Horizon Robotics' submodule pin of the same upstream commit) |
| Vendored on | 2026-04-26 |
| Vendored by | cerebrix |
| Method | `rsync -a --exclude='.git'` — code copied verbatim, the submodule pointer file was dropped so the cerebrix tree is self-contained |

## Why vendor instead of submodule

The cerebrix monorepo deliberately keeps motion-retargeting infrastructure
in-tree so retargeted artefacts can be regenerated reproducibly from
source without an external clone step. cerebrix is allowed to ship MIT-
licensed third-party code as long as the upstream `LICENSE` file
remains alongside the source — see `LICENSE` in this directory.

## What changed vs. upstream

**Nothing changed in the source code.** The upstream commit is
faithfully preserved. cerebrix-side wrappers and integration glue
live outside this directory:

* `cerebrix/scripts/tools/retarget_amass_to_g1.py` — thin driver that
  invokes GMR's `GeneralMotionRetargeting` Python API and writes
  cerebrix-schema-v1 ``.npz`` motion files (Track-2 Phase H).
* `cerebrix/thirdparties/GMR/README_CEREBRIX.md` — local
  installation / runtime guidance specific to the cerebrix workflow.

## Updating to a newer GMR

```bash
# 1. fetch the desired upstream commit into a temporary workspace
# 2. rsync into cerebrix/thirdparties/GMR/ (preserving --exclude='.git')
# 3. update the "Upstream commit" + "Vendored on" fields in this file
# 4. run scripts/tools/retarget_amass_to_g1.py against a known clip
#    and diff its output against a baseline snapshot
```

## Asset layout

The upstream `assets/` directory ships robot URDFs/STL meshes for ~15
humanoid robots **and** the SMPL-X body models (`assets/body_models/`,
~2.7 GB). The body models are not redistributable under SMPL-X's
license — `cerebrix/.gitignore` keeps that subdirectory out of git
while leaving the per-robot mesh assets tracked. Re-populate
`assets/body_models/smplx/` from your local SMPL-X licence on a fresh
clone.
