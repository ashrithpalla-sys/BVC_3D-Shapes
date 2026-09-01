import numpy as np

from stoolgen.visualization import contact_sheet, render_cloud


def test_rendering_writes_image(tmp_path):
    points = np.random.default_rng(1).normal(size=(32, 3))
    assert render_cloud(points, 100).size == (100, 100)
    output = contact_sheet(np.stack((points, points)), tmp_path / "sheet.png", columns=2, cell_size=100)
    assert output.exists()
