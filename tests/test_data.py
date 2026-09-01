from stoolgen.data import generate_dataset, validate_archive, StoolPointCloudDataset


def test_dataset_round_trip(tmp_path):
    path = generate_dataset(tmp_path / "tiny.npz", 10, 64, seed=2)
    stats = validate_archive(path)
    assert stats["shapes"] == 10
    train = StoolPointCloudDataset(path, "train", seed=2)
    assert train[0]["points"].shape == (64, 3)
