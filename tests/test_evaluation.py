import numpy as np
import torch

from stoolgen.evaluation import _sample_quality


def test_structural_quality_recognizes_simple_stool_cloud():
    rng = np.random.default_rng(3)
    seat = np.column_stack((rng.uniform(-0.5, 0.5, 40),
                            rng.uniform(-0.5, 0.5, 40), rng.uniform(0.7, 0.9, 40)))
    legs = []
    for x, y in ((-.4, -.4), (-.4, .4), (.4, -.4), (.4, .4)):
        legs.append(np.column_stack((rng.normal(x, .02, 20), rng.normal(y, .02, 20),
                                     rng.uniform(-1, .7, 20))))
    cloud = np.concatenate((seat, *legs)).astype(np.float32)
    metrics = _sample_quality(np.stack((cloud, cloud)), np.stack((cloud, cloud)), torch.device("cpu"))
    assert metrics["generated_stool_structure_fraction"] == 1.0
