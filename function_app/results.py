"""This module is for formatting and saving the feature store output result."""
import json
import pandas as pd
import polars as pl

from function_app.io.event_publisher import EventHubProducer
from function_app.commons.logger import logger
from function_app.commons.utils import timer

class Results:
    def __init__(self, current_features, anticipated_features, no_plant_conditions_tags,
                plant_conditions_tags, event_timestamp, flag):
        """
        Args:
            dataframe: dataframe with all the current features
            dataframe: dataframe with all the anticipated features
            list: list of features other than plant conditions
            list: list of plant conditions features
            event_timestamp: Timestamp from triggering event record
            flag: Flag variable for optimizer recommendations to save or not
            """
        # dataframe with all the current features
        self.current_features = current_features
        # dataframe with all the anticipated features
        self.anticipated_features = anticipated_features
        # list of features other than plant conditions
        self.no_plant_conditions_tags = no_plant_conditions_tags
        # list of plant conditions features
        self.plant_conditions_tags = plant_conditions_tags
        # Timestamp from triggering event record
        self.event_timestamp = event_timestamp
        # Flag for optimizer recommendations to save or not
        self.flag = flag

    @timer
    def format_results(self):
        """format the current and anticipated features dataframes to be in a proper json format

        Returns:
            string: formatted feature store output with all the current, anticipated features, plant conditions for event_timestamp
        """
        try:
            # check if there are empty inputs
            assert self.current_features is not None, "The current features is None"
            assert not self.current_features.is_empty(), "The current features is empty"
            assert self.anticipated_features is not None, "The anticipated features is None"
            assert not self.anticipated_features.is_empty(), "The anticipated features is empty"

            # Format currentFeatureRecords
            # exclude plant conditions
            current = self.current_features.select(self.no_plant_conditions_tags)
            current = current.filter(pl.col("*") == self.event_timestamp).clone()
            current = (
                current.melt(id_vars=["*"], value_name="*")
                .rename({"variable": "*"})
                .select(pl.col("*"), pl.col("*").cast(pl.Int16), pl.col("*"))
                .sort("*")
            )
            current = current.with_columns(pl.col("*").cast(pl.Datetime).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            # due to Polars limitation on nested dictionaries, we need to convert to pandas
            pd_current = current.to_pandas()
            current = (
                pd_current.groupby(["*"]).apply(
                    lambda x: x[["*", "*"]].to_dict("*")).reset_index().rename(columns={0: "currentFeatureRecords"})
            )

            # Format anticipatedFeatureRecords
            anticipated = self.anticipated_features.select(self.no_plant_conditions_tags)
            anticipated = anticipated.filter(pl.col("*") == self.event_timestamp).clone()
            anticipated = (
                anticipated.melt(id_vars=["*"], value_name="*")
                .rename({"variable": "*"})
                .select(pl.col("*"), pl.col("*").cast(pl.Int16), pl.col("*"))
                .sort("id")
            )
            anticipated = anticipated.with_columns(pl.col("*").cast(pl.Datetime).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            pd_anticipated = anticipated.to_pandas()
            anticipated = (
                pd_anticipated.groupby(["*"])
                .apply(lambda x: x[["*", "*"]].to_dict("*"))
                .reset_index()
                .rename(columns={0: "anticipatedFeatureRecords"})
            )

            # Format plantConditions
            conditions = (
                self.current_features.select(self.plant_conditions_tags)
                .filter(pl.col("*") == self.event_timestamp).clone()
            )

            conditions = (
                conditions.melt(id_vars=["*"], value_name="*")
                .rename({"variable": "*"})
                .select(pl.col("*"), pl.col("*").cast(pl.Int16), pl.col("*"))
                .sort("*")
            )
            conditions = conditions.with_columns(pl.col("*").cast(pl.Datetime).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            pd_conditions = conditions.to_pandas()
            conditions = (
                pd_conditions.groupby(["*"]).apply(lambda x: x[["*", "*"]].to_dict("*")).reset_index().rename(columns={0: "plantConditions"})
            )

            # Join current, anticiapted and plant conditions and convert to json format string
            feature_store_output = pd.merge(
                pd.merge(current, anticipated, on=["*"], how="inner"),
                conditions,
                on=["*"],
                how="inner",
            )
            # add eventType column after the timestamp
            feature_store_output.insert(1, "eventType", "rtfsOutput")

            # add flag column as output
            feature_store_output.insert(1, "flag", self.flag)

            # make sure there is right number of attributes
            assert feature_store_output.shape[1] == 6, "The featurestore returned wrong number of attributes"

            feature_store_output = feature_store_output.iloc[0].to_json(orient="index")
        except Exception as exception:
            logger.error("An error occurred with failure on formatting results: %s", str(exception))
            raise exception
        return feature_store_output

    @timer
    def save_results(self, results: dict, eventhub_connection_string=None, feature_store_output_eh=None):
        """save the results in json format
        Args:
            dictionary: dictionary with all the features

        Returns:
            None
        """
        try:
            if eventhub_connection_string is not None:
                event_producer = EventHubProducer(eventhub_connection_string, feature_store_output_eh)
                event_producer.process_data_to_eventhub(data=results)
                logger.info("Site data - Sent to azure eventhub successfully")
            else:
                with open("featurestore/data/featurestore_output.json", "w") as file:
                    json.dump(results, file)
        except Exception as exception:
            logger.error("An error occurred with failure on saving feature store results: %s", str(exception))
            raise exception
