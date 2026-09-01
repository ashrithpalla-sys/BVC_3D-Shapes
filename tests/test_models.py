import torch

from stoolgen.losses import chamfer_distance
from stoolgen.models import PointCloudAutoencoder, PointCloudVAE


def test_model_shapes_and_gradients():
    points = torch.randn(2, 32, 3)
    for model in (PointCloudAutoencoder(32, 8, 32), PointCloudVAE(32, 8, 32)):
        output = model(points)
        assert output["points"].shape == points.shape
        loss = chamfer_distance(output["points"], points)
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_chamfer_identity_is_zero():
    points = torch.randn(2, 16, 3)
    assert torch.allclose(chamfer_distance(points, points), torch.tensor(0.0))
