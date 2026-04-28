"""Shared MuJoCo geom helpers used by retargeting and post-processing."""

from __future__ import annotations

import mujoco as mj
import numpy as np


def geom_min_z(model, data, geom_id: int) -> float:
    """Lowest z extent of a geom in world coordinates."""
    geom_type = model.geom_type[geom_id]
    xpos = data.geom_xpos[geom_id]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]
    if geom_type == mj.mjtGeom.mjGEOM_CYLINDER:
        radius = float(size[0])
        half_height = float(size[1])
        axis = xmat[:, 2]
        z_extent = (
            half_height * abs(axis[2])
            + radius * np.sqrt(max(0.0, 1.0 - axis[2] * axis[2]))
        )
        return float(xpos[2] - z_extent)
    return float(xpos[2] - np.max(size))


def collect_geom_ids_by_token(model, name_token: str) -> list[int]:
    """Return all geom IDs whose name contains name_token."""
    geom_ids = []
    for geom_id in range(model.ngeom):
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name is not None and name_token in geom_name:
            geom_ids.append(geom_id)
    return geom_ids
