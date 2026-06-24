from src.training.config import load_config


def test_config_base_override_merges_nested_sections():
    cfg = load_config("configs/nu_jepa_3dcal_12k_robust.yaml")

    assert cfg["data"]["dataset_path"] == "/scratch/fcufino/events_v8.0_1000"
    assert cfg["model"]["embed_dim"] == 128
    assert cfg["training"]["lr"] == 0.0001
    assert cfg["training"]["checkpoint_dir"] == "/scratch/fcufino/nu_jepa/checkpoints_12k_robust"
