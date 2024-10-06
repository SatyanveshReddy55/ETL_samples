"""
    This module provides a mock data generator class for generating 
    sample data without the need for actual sources.
"""
import json
import os
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

import yaml


class MockDataGenerator:
    """
    A class for generating various types of mock data,
    such as random numbers, strings, dates, and more.
    """

    def __init__(self):
        self.current_value = str(random.randint(10**15, 10**16 - 1))

    def generate_event_sequence(self):
        """
        Method to generate event sequence
        """
        self.current_value = str(int(self.current_value) + 1).zfill(16)
        return self.current_value

    def generate_random_string(self, length=10):
        """
        Method to generate random string
        """
        letters = string.ascii_letters
        return "".join(random.choice(letters) for _ in range(length))

    def generate_random_value(
        self,
        data_type,
        event_datetime,
        f_min_value,
        f_max_value,
        i_min_value,
        i_max_value,
    ):
        """
        Generates a random value within a specified range using datatype
        """
        data_type = data_type.lower()
        data_type_mapping = {
            "string": self.generate_random_string,
            "float": lambda: round(
                random.uniform(
                    f_min_value if f_min_value is not None else -400.0,
                    f_max_value if f_max_value is not None else 1200.0,
                ),
                6,
            ),
            "decimal": lambda: round(random.uniform(-400.0, 1000.0), 6),
            "int": lambda: random.randint(
                i_min_value if i_min_value is not None else 0,
                i_max_value if i_max_value is not None else 100,
            ),
            "number": lambda: random.randint(0, 1),
            "bool": lambda: random.choice([0, 1]),
            "fbool": lambda: bool(random.choice([0])),
            "timestamp": lambda: event_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "discrete": lambda: round(random.uniform(0.0, 100.0), 6),
        }
        return data_type_mapping.get(data_type, lambda: "")()

    def generate_mock_data(
        self,
        schema,
        event_type,
        start_time,
        end_time,
        interval_seconds,
        allowed_values,
        product_mapping,
        col_nm,
        ids,
        columns_with_none,
        f_min_value,
        f_max_value,
        i_min_value,
        i_max_value,
    ):
        """
        Method to generate mock data
        """

        mock_data = []

        if col_nm:
            ids = ids
        else:
            ids = ["#NA"]

        current_time = start_time
        while current_time <= end_time:
            data_point = {}
            data_point_1 = {}

            data_point["sequenceID"] = self.generate_event_sequence()

            current_data_time = current_time
            data_list = []

            for i, id in enumerate(ids):
                for key, value in schema.items():
                    if isinstance(value, list):
                        for val in value:
                            for col, dtype in val.items():
                                if col == "id":
                                    data_point_1[col] = ids[i]
                                if (
                                    allowed_values
                                    and str(id) in allowed_values
                                    and col == "value"
                                ):
                                    column_values = allowed_values[str(id)]
                                    data_point_1[col] = random.choice(column_values)
                                elif col == "value":
                                    if columns_with_none and col in columns_with_none:
                                        data_point_1[col] = (
                                            None
                                            if random.random() < 0.2
                                            else self.generate_random_value(
                                                dtype,
                                                current_data_time,
                                                f_min_value,
                                                f_max_value,
                                                i_min_value,
                                                i_max_value,
                                            )
                                        )
                                    else:
                                        data_point_1[col] = self.generate_random_value(
                                            dtype,
                                            current_data_time,
                                            f_min_value,
                                            f_max_value,
                                            i_min_value,
                                            i_max_value,
                                        )
                                else:
                                    pass
                            i += 1
                            data_list.append(data_point_1)
                            data_point_1 = dict(data_point_1)
                    else:
                        col = key
                        dtype = value
                        if allowed_values and col in allowed_values and col != col_nm:
                            column_values = allowed_values[col]
                            data_point[col] = random.choice(column_values)
                        elif col != col_nm:
                            if columns_with_none and col in columns_with_none:
                                data_point[col] = (
                                    None
                                    if random.random() < 0.2
                                    else self.generate_random_value(
                                        dtype,
                                        current_data_time,
                                        f_min_value,
                                        f_max_value,
                                        i_min_value,
                                        i_max_value,
                                    )
                                )
                            else:
                                data_point[col] = self.generate_random_value(
                                    dtype,
                                    current_data_time,
                                    f_min_value,
                                    f_max_value,
                                    i_min_value,
                                    i_max_value,
                                )
                        else:
                            pass

                current_data_time += timedelta(seconds=interval_seconds)
            data_point["records"] = data_list
            mock_data.append(data_point.copy())
            current_time += timedelta(seconds=interval_seconds)
        return mock_data


def read_config_file(working_dir: str, config_file_path: str) -> dict:
    """
    Method to load the config file
    """
    config_file_full_path = os.path.join(working_dir, config_file_path)
    with open(config_file_full_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def main(
    schema,
    event_type,
    start_time,
    end_time,
    interval_seconds,
    allowed_values,
    product_mapping,
    col_nm,
    ids,
    columns_with_none,
    f_min_value,
    f_max_value,
    i_min_value,
    i_max_value,
):
    """
    Main method
    """
    generator = MockDataGenerator()
    data = generator.generate_mock_data(
        schema,
        event_type,
        start_time,
        end_time,
        interval_seconds,
        allowed_values,
        product_mapping,
        col_nm,
        ids,
        columns_with_none,
        f_min_value,
        f_max_value,
        i_min_value,
        i_max_value,
    )
    return data


def read_variable_from_config(value):
    """
    Method to read all the variables in the config file.
    """
    schema = value.get("schema")
    start_time = datetime.strptime(value.get("start_time"), "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(value.get("end_time"), "%Y-%m-%d %H:%M:%S")
    interval_seconds = value.get("interval_seconds")
    allowed_values = value.get("allowed_values", None)
    product_mapping = value.get("product_mapping", None)
    col_nm = value.get("col_nm", None)
    ids = value.get("ids", None)
    columns_with_none = value.get("columns_with_none", None)
    f_min_value = value.get("f_min_value", None)
    f_max_value = value.get("f_max_value", None)
    i_min_value = value.get("i_min_value", None)
    i_max_value = value.get("i_max_value", None)
    return {
        "schema": schema,
        "start_time": start_time,
        "end_time": end_time,
        "interval_seconds": interval_seconds,
        "allowed_values": allowed_values,
        "product_mapping": product_mapping,
        "col_nm": col_nm,
        "ids": ids,
        "columns_with_none": columns_with_none,
        "f_min_value": f_min_value,
        "f_max_value": f_max_value,
        "i_min_value": i_min_value,
        "i_max_value": i_max_value,
    }


if __name__ == "__main__":
    working_dir = os.getenv(
        "WORK_DIR",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
        ),
    )
    config = read_config_file(working_dir, "config.yaml")
    combined_data = []

    for x in config["event_types"]:
        value = config["event_types"][x]
        inputs = read_variable_from_config(value)
        inputs["event_type"] = x
        event_type_data = main(**inputs)
        combined_data.append(event_type_data)

    # Save the generated mock data as a json file
    file_path = Path.cwd() / "LocalInputs/mock_data.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as json_file:
        for data in combined_data:
            for event in data:
                json.dump(event, json_file)
                json_file.write("\n")
