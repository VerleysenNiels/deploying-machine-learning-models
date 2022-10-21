from model.config.core import config
from model.processing.features import TemporalVariableTransformer


def test_temporal_variable_transformer(sample_input_data):
    # Given
    transformer = TemporalVariableTransformer(
        variables=config.model_config.cabin
    )
    assert sample_input_data["cabin"].iat[0] == "B5"

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["cabin"].iat[0] == "B"
