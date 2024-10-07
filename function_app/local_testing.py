"""This module is for reading data from local files for local testing."""
import os
import polars as pl

from function_app.commons.utils import timer
from function_app.commons.logger import logger


class LocalTesting:
    @timer
    def read_data(self):
        """Read local files for featurestore event and historical data
        Args: None

        Returns:
            dataframe: dataframe with the local feature store event data
            dataframe: dataframe with the local feature store historical data
        """
        try:
            hist_event_input = pl.read_json(os.path.join(os.getcwd(), "featurestore/data/input1.json"))
            event_data = hist_event_input.tail(1)
            eventtype2_data = pl.read_json(os.path.join(os.getcwd(), "featurestore/data/input2.json"))
            logger.info(f"Feature Store input: {event_data}")
            logger.info("Triggering feature Store")
        except Exception as exception:
            logger.error("An error occurred with failure on reading local feature store input files: %s", str(exception))
            raise exception
        return event_data, hist_event_input, eventtype2_data
