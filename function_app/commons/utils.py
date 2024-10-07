"""module for general functionalities such as logger, timer."""

import time
import json
from jsonschema import validate
from function_app.commons.logger import logger
import function_app.commons.globals as gl

def timer(func):
    """
    Decorator that measures the execution time of a function.

    Args:
        func (callable): The function to be timed.

    Returns:
        callable: The wrapper function.
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function that measures the execution time of the input function.

        Args:
            *args: Positional arguments passed to the input function.
            **kwargs: Keyword arguments passed to the input function.

        Returns:
            Any: The result of the input function.
        """
        if gl.DEBUG_MODE:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
            return result
        # If debug mode is False, simply call the original function without measuring time
        return func(*args, **kwargs)

    return wrapper

def read_json_file(file_path):
    """
    Read json data given file path
    Args: 
        file_path: Path of the json file to read

    Returns:
        dict: Python dictionary
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except Exception as exception:
        raise exception

def validate_schema(data, schema):
    """
    Validate if data is following the correct schema
    Args: 
        data: json object
        schema: json object

    Returns:
        bool: True when schema matches, Raise exception for traceback when schema doesn't match
    """
    try:
        if isinstance(data, dict):
            validate(instance=data, schema=schema)
        elif isinstance(data, list):
            for item in data:
                validate(instance=item, schema=schema)
        return True
    except Exception as exception:
        logger.error(f"Error in reading: {exception}")
        raise exception

def within_time_range(timestamp1, timestamp2, time_difference):
    """
    Checks if the absolute difference between two timestamps is within a specified time difference.

    Parameters:
        timestamp1 (datetime): The first timestamp
        timestamp2 (datetime): The second timestamp
        time_difference (int): The time difference in seconds within which the timestamps are considered to be within range.

    Returns:
        bool: True if the absolute difference between the two timestamps is less than or equal to the specified time difference, False otherwise.
    """
    # Calculate time difference
    time_diff = abs(timestamp2 - timestamp1).total_seconds()

    # Check if the time difference is less than or equal to the specified time range
    return time_diff <= time_difference
