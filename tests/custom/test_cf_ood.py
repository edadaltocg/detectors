import torch

from detectors import create_detector


def test_cf_ood_minimal_flow():
    D = 8
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(D, 1),
    )
    det = create_detector(
        "cf_ood",
        model=model,
        features_nodes=["1"],
        pooling_op_name="none",
        aggregation_method_name="none",
    )
    det.device = torch.device("cpu")

    N = 20
    det.train_features = {"1": torch.randn(N, D)}
    det._fit_params()

    X = torch.randn(10, D)
    s = det._layer_score(X, layer_name="1")

    assert s.shape == (10,)
    assert torch.isfinite(s).all()
