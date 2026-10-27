import numpy as np
import pytest
import trimesh


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO3D_DATA", str(tmp_path))
    return tmp_path


def tilt(points: np.ndarray, deg: float = 12.0) -> np.ndarray:
    """Rotate about X so 'up' is no longer +Y (a pure tilt: the minimal alignment undoes it exactly)."""
    return trimesh.transformations.transform_points(points, trimesh.transformations.rotation_matrix(
        np.radians(deg), [1, 0, 0]))


def plane_mesh(size_x, size_z, y=0.0, step=0.05) -> trimesh.Trimesh:
    """A flat grid in the XZ plane at height y."""
    m = trimesh.creation.box(extents=[size_x, 0.001, size_z]).subdivide_to_size(step)
    m.apply_translation([0, y, 0])
    return m


def with_colours(m: trimesh.Trimesh, rgb) -> trimesh.Trimesh:
    m.visual.vertex_colors = np.tile(np.array([*rgb, 255], dtype=np.uint8), (len(m.vertices), 1))
    return m


def write_raw(path, meshes, deg=12.0):
    """Merge meshes, tilt them, save as a GLB like the Space's output."""
    merged = trimesh.util.concatenate(meshes)
    merged.vertices = tilt(np.asarray(merged.vertices), deg)
    merged.export(path)
    return path
