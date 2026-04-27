# Vendored GMR provenance

This is cerebrix's vendor fork of [General Motion Retargeting (GMR)](https://github.com/YanjieZe/GMR)
by Yanjie Ze, MIT-licensed.

## Source

| Field | Value |
|---|---|
| Upstream repo | <https://github.com/YanjieZe/GMR.git> (MIT, Yanjie Ze) |
| Upstream pin commit | `5cd312e501db070faad588ce4e510ce1771f7410` ("Revise demo section in README.md") |
| Cerebrix vendor fork | <https://github.com/yunhao-crypto/GMR.git> (private fork) |
| Vendor branch (cerebrix-tracked) | `main` |
| Mirror branch (upstream sync) | `master` (synced with `YanjieZe/GMR/master` via GitHub "Sync fork") |
| Consumed by | cerebrix, as a git submodule at `thirdparties/GMR/` (pinned to a commit on `main`) |

## Branch model

The fork carries two long-lived branches:

* **`main`** — what cerebrix actually consumes. Built on the upstream pin
  `5cd312e` plus cerebrix-side commits (this PROVENANCE.md, README_CEREBRIX.md,
  HoloMotion-side IK patches). cerebrix's `.gitmodules` records `branch = main`.
* **`master`** — passive mirror of `YanjieZe/GMR/master`. Maintained via
  GitHub's "Sync fork" button so the fork stays measured against the latest
  upstream tip.

Pulling new upstream changes into cerebrix:

```bash
# In yunhao-crypto/GMR (web UI or CLI):
#   - "Sync fork" on the master branch (merges YanjieZe/GMR/master into ours)
#   - then locally:
git checkout main
git merge master                 # or rebase if cleaner
git push origin main

# In cerebrix:
git submodule update --remote thirdparties/GMR        # advances the pointer
git -C thirdparties/GMR log --oneline -3              # verify the new HEAD
git add thirdparties/GMR
git commit -m "chore(thirdparties): bump GMR submodule"
```

## What changed vs. upstream

Two cerebrix/HoloMotion commits sit on top of `5cd312e`:

1. **`20a3f9c` — Import HoloMotion-side modifications to GMR.** Imported
   from HoloMotion's vendored copy of GMR (the same `5cd312e` baseline);
   originally authored by the HoloMotion team prior to vendoring. Modifies:

   * `general_motion_retargeting/motion_retarget.py` — first-frame IK
     damping bump, `max_iter` bump, and a `PostureTask` + prev-frame
     `PostureTask` for posture regularization. Eliminates the occasional
     first-frame drift on high-DOF humanoids.
   * `scripts/smplx_to_robot_dataset.py` — `HOLOMOTION_GMR_DEVICE` env
     var lets callers pin a GPU index or fall back to CPU; the
     `torch.cuda.empty_cache` call is now guarded by
     `torch.cuda.is_available()` for CPU-only hosts.

2. **`a90c403` — Add cerebrix-side provenance + helper docs.**
   This file (`PROVENANCE.md`) and `README_CEREBRIX.md`. No source code
   changes.

The cerebrix-side glue that *consumes* GMR (driver script, integration
tests, dataset configs) lives **outside this directory** in the cerebrix
repository — see `README_CEREBRIX.md` for the boundaries.

## Asset layout — license-sensitive directories

The upstream `assets/` directory ships robot URDFs/STL meshes for ~15
humanoid robots **and** the SMPL-X body models (`assets/body_models/`,
~2.7 GB). The body models are not redistributable under SMPL-X's
research-only license — GMR's own `.gitignore` excludes
`assets/body_models/`, so neither this fork nor any cerebrix submodule
pointer can accidentally track them. Each developer re-populates
`assets/body_models/smplx/` on their local machine using their own
MPI license — see `cerebrix/docs/install.md` Step 6.3 for the
`models_smplx_v1_1.zip` flow.

## License summary

* Upstream GMR source — MIT (Yanjie Ze, see `LICENSE`)
* HoloMotion-side patches (commit `20a3f9c`) — MIT (HoloMotion team,
  inherited from upstream license)
* cerebrix-side glue (`PROVENANCE.md`, `README_CEREBRIX.md`) —
  Apache-2.0 (cerebrix contributors)
* SMPL-X body models — research-only, NOT redistributable; obtain
  per-developer from <https://smpl-x.is.tue.mpg.de/>
