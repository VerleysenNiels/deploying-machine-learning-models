import pytest
from sklearn.model_selection import train_test_split

from model.config.core import config
from model.processing.data_manager import _load_raw_dataset


@pytest.fixture
def sample_input_data():
    data = _load_raw_dataset(file_name=config.app_config.raw_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data,  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    return X_test
