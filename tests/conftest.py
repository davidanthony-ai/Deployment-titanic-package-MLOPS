import pytest
import logging
from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset
from sklearn.model_selection import train_test_split 

logger = logging.getLogger(__name__)

# @pytest.fixture()
# def sample_input_data():
#     return load_dataset(file_name=config.app_config.test_data_file)

@pytest.fixture
def sample_input_data():
    # data can be find here : https://www.openml.org/data/get_csv/16826755/phpMYEkMl
    data = load_dataset(file_name=config.app_config.titanic_data)

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