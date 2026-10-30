"""Hand-calculated scientific metrics and patient-level uncertainty checks."""

import numpy as np
import pytest

from cardio_audio.metrics import classification_metrics, patient_bootstrap


def test_metrics_match_hand_calculated_four_class_confusion():
    truth = [0, 0, 1, 1, 2, 2, 3, 3]
    prediction = [0, 1, 1, 2, 2, 3, 3, 0]
    result = classification_metrics(truth, prediction)
    assert result["confusion_matrix"] == [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]]
    for metric in ["macro_f1", "accuracy", "balanced_accuracy", "icbhi_score", "icbhi_sensitivity", "icbhi_specificity"]:
        assert result[metric] == pytest.approx(0.5)
    for item in result["per_class"]:
        assert item["support"] == 2
        assert item["sensitivity"] == pytest.approx(0.5)
        assert item["specificity"] == pytest.approx(5 / 6)


def test_icbhi_requires_correct_abnormal_subclass_not_just_abnormal_flag():
    result = classification_metrics([0, 1, 2, 3], [0, 2, 3, 1])
    assert result["icbhi_sensitivity"] == 0.0
    assert result["icbhi_specificity"] == 1.0
    assert result["icbhi_score"] == 0.5
    assert result["macro_f1"] == pytest.approx(0.25)


def test_missing_true_class_has_undefined_sensitivity_not_invented_success():
    result = classification_metrics([0, 0, 0], [0, 0, 0])
    assert result["per_class"][1]["sensitivity"] is None
    assert result["icbhi_sensitivity"] is None
    assert result["icbhi_score"] is None


@pytest.mark.parametrize("truth,prediction", [([], []), ([0, 1], [0]), ([0, 4], [0, 1]),
                                            ([0, 1], [0, -1]), ([0.2, 1], [0, 1])])
def test_invalid_metric_input_is_rejected(truth, prediction):
    with pytest.raises(ValueError):
        classification_metrics(truth, prediction)


def test_bootstrap_perfect_predictions_remain_perfect():
    truth = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    result = patient_bootstrap(truth, truth, ["A"] * 4 + ["B"] * 4, repetitions=20, seed=17)
    assert result["lower"] == 1.0
    assert result["upper"] == 1.0
    assert result["patients"] == 2
    assert result["method"] == "bootstrap por paciente"


def test_bootstrap_is_deterministic_for_same_seed():
    truth = [0, 1, 2, 3, 0, 1, 2, 3]
    prediction = [0, 1, 1, 3, 0, 2, 2, 3]
    groups = ["A"] * 4 + ["B"] * 4
    first = patient_bootstrap(truth, prediction, groups, repetitions=30, seed=19)
    second = patient_bootstrap(truth, prediction, groups, repetitions=30, seed=19)
    assert first == second
    assert 0 <= first["lower"] <= first["upper"] <= 1
