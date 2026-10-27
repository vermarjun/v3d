import json

import numpy as np
import trimesh

from conftest import plane_mesh, with_colours, write_raw
from video3d import postprocess as pp


def scene_vertices(path):
    s = trimesh.load(path)
    geoms = s.geometry.values() if isinstance(s, trimesh.Scene) else [s]
    return np.vstack([g.vertices for g in geoms])


def test_tilted_floor_and_box_are_aligned(tmp_path):
    floor = with_colours(plane_mesh(6, 6, y=-1.3), (120, 120, 130))
    box = with_colours(trimesh.creation.box(extents=[1, 1, 1]).subdivide_to_size(0.05), (200, 40, 40))
    box.apply_translation([1.0, -1.3 + 0.5, 0.5])  # sits on the floor
    raw = write_raw(tmp_path / "raw.glb", [floor, box], deg=12)

    meta = pp.process(raw, tmp_path / "out", "abc", "Tilted", num_frames=3, backend="hf-space")

    v = scene_vertices(tmp_path / "out" / "scene.glb")
    assert meta["floor_y"] == 0.0 and meta["up_axis"] == "+Y"
    assert (v[:, 1] < -0.05).mean() < 0.001            # nothing below the floor
    assert abs(v[:, 1].max() - 1.0) < 0.06             # box is 1 m tall
    assert abs(meta["bbox_min"][1]) < 0.06
    assert abs((meta["bbox_min"][0] + meta["bbox_max"][0]) / 2) < 1e-3   # centred on x and z
    assert abs((meta["bbox_min"][2] + meta["bbox_max"][2]) / 2) < 1e-3
    assert abs(meta["bbox_max"][0] - meta["bbox_min"][0] - 6) < 0.1      # floor is 6 m wide
    assert json.loads((tmp_path / "out" / "meta.json").read_text())["timings_s"]["postprocess"] >= 0


def test_colours_survive(tmp_path):
    floor = with_colours(plane_mesh(4, 4), (10, 200, 30))
    raw = write_raw(tmp_path / "raw.glb", [floor], deg=5)
    pp.process(raw, tmp_path / "out", "abc", "Colours")
    m = trimesh.load(tmp_path / "out" / "scene.glb")
    g = list(m.geometry.values())[0]
    assert np.allclose(g.visual.vertex_colors[:, :3].mean(0), [10, 200, 30], atol=1)


def test_stretched_triangles_and_islands_are_dropped():
    floor = plane_mesh(4, 4, step=0.05)
    v, f = np.asarray(floor.vertices), np.asarray(floor.faces)
    # one flying-pixel triangle: two floor vertices joined to a point 3 m up
    top = v.shape[0]
    v2 = np.vstack([v, [[0.0, 3.0, 0.0]]])
    sliver = np.array([[0, 1, top]])
    f2 = np.vstack([f, sliver])
    keep = pp.drop_stretched(v2, f2)
    assert not keep[-1]
    assert keep[:-1].mean() > 0.99

    cube = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    cube.apply_translation([20, 0, 0])
    vv = np.vstack([v2, cube.vertices])
    ff = np.vstack([f2, cube.faces + len(v2)])
    alive = pp.drop_islands(len(vv), ff, 300)
    assert not alive[len(f2):].any() and alive[:len(f)].all()


def test_wall_dominant_scene_picks_floor(tmp_path):
    wall = with_colours(trimesh.creation.box(extents=[8, 3, 0.001]).subdivide_to_size(0.05), (90, 90, 120))
    wall.apply_translation([0, 1.5 - 1.0, -2])                        # 8 x 3 m wall, floor at y = -1
    floor = with_colours(plane_mesh(3, 1.5, y=-1.0), (80, 80, 80))    # smaller than the wall
    raw = write_raw(tmp_path / "raw.glb", [wall, floor], deg=8)
    meta = pp.process(raw, tmp_path / "out", "w", "Wall")
    assert meta["floor_y"] == 0.0
    assert abs(meta["bbox_max"][1] - 3.0) < 0.1  # wall is 3 m tall, so up really is +Y


def test_lowest_horizontal_plane_wins_over_bigger_table(tmp_path):
    floor = with_colours(plane_mesh(6, 6, y=0.0), (80, 80, 80))
    table = with_colours(plane_mesh(5, 5, y=0.75), (160, 120, 80))    # bigger share of points? same density, close
    raw = write_raw(tmp_path / "raw.glb", [floor, table], deg=6)
    meta = pp.process(raw, tmp_path / "out", "t", "Table")
    assert abs(meta["bbox_max"][1] - 0.75) < 0.05                     # the floor is y = 0, the table is above it


def test_no_plane_leaves_floor_null(tmp_path):
    blob = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    raw = write_raw(tmp_path / "raw.glb", [with_colours(blob, (100, 100, 100))], deg=0)
    meta = pp.process(raw, tmp_path / "out", "n", "Blob")
    assert meta["floor_y"] is None
    assert abs(meta["bbox_min"][1]) < 1e-3


def test_decimation_hits_budget_and_keeps_colour(tmp_path):
    floor = with_colours(plane_mesh(6, 6, step=0.05), (50, 100, 150))
    raw = write_raw(tmp_path / "raw.glb", [floor], deg=3)
    meta = pp.process(raw, tmp_path / "out", "d", "Dec", max_faces=5000)
    assert meta["stats"]["final_faces"] <= 5000
    m = trimesh.load(tmp_path / "out" / "scene.glb")
    g = list(m.geometry.values())[0]
    assert np.allclose(g.visual.vertex_colors[:, :3].mean(0), [50, 100, 150], atol=2)


def test_meta_preserves_scale_and_splat_on_reprocess(tmp_path):
    raw = write_raw(tmp_path / "raw.glb", [with_colours(plane_mesh(4, 4), (1, 2, 3))], deg=2)
    out = tmp_path / "out"
    pp.process(raw, out, "r", "Re")
    meta = json.loads((out / "meta.json").read_text())
    meta["scale_factor"] = 1.7
    meta["splat"] = {"url": "/scenes/r/splat.ply", "format": "ply", "aligned": True}
    (out / "meta.json").write_text(json.dumps(meta))
    again = pp.process(raw, out, "r", "Re")
    assert again["scale_factor"] == 1.7 and again["splat"]["format"] == "ply"


def test_cameras_follow_the_alignment(tmp_path):
    floor = with_colours(plane_mesh(6, 6, y=-1.3), (120, 120, 130))
    raw = write_raw(tmp_path / "raw.glb", [floor], deg=12)  # tilted 12 degrees about X
    cams = [{"position": [0, 0, 0], "forward": [0, 0, -1]}, {"position": [1, 0, 0], "forward": [1, 0, 0]}]
    meta = pp.process(raw, tmp_path / "out", "c", "Cams", cameras=cams)
    assert len(meta["cameras"]) == 2
    c0 = meta["cameras"][0]
    # the camera was 1.3 m above the floor along the original up axis, so about 1.3 m above y = 0 after alignment
    assert 1.2 < c0["position"][1] < 1.4
    d = np.array(c0["look_at"]) - np.array(c0["position"])
    assert abs(np.linalg.norm(d) - 1.0) < 1e-3
    # default: one camera (frame 0 at the origin) when no poses are given
    assert len(pp.process(raw, tmp_path / "out2", "c", "Cams")["cameras"]) == 1
