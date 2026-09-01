import torch

from stoolgen.config import choose_device, load_config, seed_everything


def test_seed_is_reproducible():
    seed_everything(11)
    first = torch.rand(4)
    seed_everything(11)
    assert torch.equal(first, torch.rand(4))


def test_quick_config_loads():
    config = load_config("configs/quick.yaml")
    assert config["data"]["points_per_shape"] >= 32
    assert choose_device("cpu").type == "cpu"
