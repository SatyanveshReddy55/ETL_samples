"""
   Test cases for Mock generator
"""

from datetime import datetime

import pytest

from mock_data_generator.src.mock_generator import MockDataGenerator

start_time = datetime(2023, 8, 16, 8, 0, 0)
end_time = datetime(2023, 8, 16, 16, 5, 0)


@pytest.fixture
def mock_data_generator():
    return MockDataGenerator()


def test_generate_event_sequence_unique_values(mock_data_generator):
    """
    Method to test if event sequences are unique for a large number of generated values
    """
    num_values = 1000
    sequences = set()
    for _ in range(num_values):
        sequence = mock_data_generator.generate_event_sequence()
        sequences.add(sequence)

    assert len(sequences) == num_values  # Check if all generated sequences are unique


def test_generate_random_string_default_length(mock_data_generator):
    """
    Method to test if random strings are generated with the default length (10 characters)
    """
    random_string = mock_data_generator.generate_random_string()

    assert (
        len(random_string) == 10
    )  # Check if the generated string has the expected default length


def test_generate_random_value_invalid_data_type(mock_data_generator):
    """
    Method to test if an empty string is returned for an invalid data type
    """
    invalid_data_type = "invalid_type"
    result = mock_data_generator.generate_random_value(
        invalid_data_type, None, None, None, None, None
    )

    assert result == ""  # Check if an empty string is returned for an invalid data type


def test_generate_random_value_float_within_range(mock_data_generator):
    """
    Method to test if random float values are generated within the specified range
    """
    min_value = 10.0
    max_value = 100.0
    random_float = mock_data_generator.generate_random_value(
        "float", None, min_value, max_value, None, None
    )

    assert (
        min_value <= random_float <= max_value
    )  # Check if the generated float is within the specified range


def test_generate_random_value_int_within_range(mock_data_generator):
    """
    Method to test if random integer values are generated within the specified range
    """
    min_value = 10
    max_value = 100
    random_int = mock_data_generator.generate_random_value(
        "int", None, None, None, min_value, max_value
    )

    assert (
        min_value <= random_int <= max_value
    )  # Check if the generated integer is within the specified range


def test_generate_random_value_with_allowed_values(mock_data_generator):
    """
    Method to test to define a schema with an allowed values constraint
    """
    schema = {
        "timestamp": "Timestamp",
        "p_Up": "int",
        "records": [{"id": "int", "value": "Float"}],
    }

    # Define allowed values for the "p_Up" column
    allowed_values = {
        "p_Up": [0, 1],
    }
    ids = [1, 2]

    # Generate random values based on the schema and allowed values
    random_values = mock_data_generator.generate_mock_data(
        schema,
        "test_event",
        start_time,
        end_time,
        60,
        allowed_values,
        None,
        None,
        ids,
        None,
        None,
        None,
        None,
        None,
    )

    for data_point in random_values:
        p_Up = data_point["p_Up"]
        assert (
            p_Up in allowed_values["p_Up"]
        )  # Check if the generated status value is in the allowed values list
