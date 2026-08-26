from unittest import mock

import numpy as np
import pytest

from openpi.policies.libero_policy import ACPAdvantageTag


BASE_PROMPT = "Pick up the test tube."


def test_dynamic_prompt_keeps_tag_when_dropout_does_not_fire() -> None:
    transform = ACPAdvantageTag(dropout_prob=0.3)
    with mock.patch("numpy.random.rand", return_value=0.8):
        result = transform({"prompt": BASE_PROMPT, "acp_indicator": np.array([1])})
    assert result["prompt"] == f"{BASE_PROMPT}\nAdvantage: positive"


def test_dynamic_prompt_drops_tag() -> None:
    transform = ACPAdvantageTag(dropout_prob=0.3)
    with mock.patch("numpy.random.rand", return_value=0.2):
        result = transform({"prompt": BASE_PROMPT, "acp_indicator": np.array([0])})
    assert result["prompt"] == BASE_PROMPT


def test_baked_prompt_keeps_single_tag() -> None:
    transform = ACPAdvantageTag(dropout_prob=0.3)
    prompt = f"{BASE_PROMPT}\nAdvantage: negative"
    with mock.patch("numpy.random.rand", return_value=0.8):
        result = transform({"prompt": prompt, "acp_indicator": np.array([0])})
    assert result["prompt"] == prompt


def test_baked_prompt_removes_tag_when_dropout_fires() -> None:
    transform = ACPAdvantageTag(dropout_prob=0.3)
    prompt = f"{BASE_PROMPT}\nAdvantage: positive"
    with mock.patch("numpy.random.rand", return_value=0.2):
        result = transform({"prompt": prompt, "acp_indicator": np.array([1])})
    assert result["prompt"] == BASE_PROMPT


def test_inference_adds_positive_and_does_not_duplicate_existing_tag() -> None:
    transform = ACPAdvantageTag(dropout_prob=0.3)
    assert transform({"prompt": BASE_PROMPT})["prompt"] == f"{BASE_PROMPT}\nAdvantage: positive"
    baked = f"{BASE_PROMPT}\nAdvantage: positive"
    assert transform({"prompt": baked})["prompt"] == baked


def test_baked_prompt_must_match_indicator() -> None:
    transform = ACPAdvantageTag(dropout_prob=0.3)
    with pytest.raises(ValueError, match="disagrees"):
        transform(
            {
                "prompt": f"{BASE_PROMPT}\nAdvantage: positive",
                "acp_indicator": np.array([0]),
            }
        )


@pytest.mark.parametrize("dropout", [-0.1, 1.1])
def test_dropout_range_is_validated(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout_prob"):
        ACPAdvantageTag(dropout_prob=dropout)
