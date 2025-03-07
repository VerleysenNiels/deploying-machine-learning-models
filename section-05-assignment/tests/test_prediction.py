import math

import numpy as np
from sklearn.metrics import accuracy_score

from model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 131

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    y_hat = list(predictions)
    y_true = sample_input_data["survived"]
    assert accuracy_score(y_hat, y_true) > 0.7
