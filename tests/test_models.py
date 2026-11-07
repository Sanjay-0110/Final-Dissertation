import copy

import pytest
import torch

from data.dataset import load_config
from models.proposed_model import ProposedClassifier, ProposedSegmentationModel


def config_for(mode: str) -> dict:
    config = copy.deepcopy(load_config())
    config["model"]["encoder"]["compression_mode"] = mode
    return config


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("mode, expected", [("asymmetric", 859_664), ("uniform", 860_357)])
def test_segmentation_param_counts_match_the_report(mode, expected):
    model = ProposedSegmentationModel(config_for(mode))
    assert count_params(model) == expected
    assert count_params(model) < load_config()["model"]["target_max_params"]


def test_classifier_param_count_matches_the_report():
    assert count_params(ProposedClassifier(config_for("asymmetric"))) == 350_849


def test_segmentation_output_matches_input_size():
    model = ProposedSegmentationModel(config_for("asymmetric")).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 352, 352))
    assert output.shape == (2, 1, 352, 352)


def test_classifier_encoder_loads_into_segmentation_model():
    """The warm start depends on these two encoders having identical keys."""
    config = config_for("asymmetric")
    classifier = ProposedClassifier(config)
    segmentation = ProposedSegmentationModel(config)
    segmentation.encoder.load_state_dict(classifier.encoder.state_dict(), strict=True)


def test_warm_start_across_compression_modes_fails_loudly():
    classifier = ProposedClassifier(config_for("asymmetric"))
    segmentation = ProposedSegmentationModel(config_for("uniform"))
    with pytest.raises(RuntimeError):
        segmentation.encoder.load_state_dict(classifier.encoder.state_dict(), strict=True)
