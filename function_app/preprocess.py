"""This module is for feature store input and configs preprocessing."""
from datetime import timedelta
import polars as pl

from function_app.commons.utils import timer
from function_app.commons.logger import logger

class PreProcess:
    def __init__(self, hist_event_input, trigger_event_data, features_config, tags_config, plant_config):
        """
        Args:
            hist_event_input: DataFrame containing the historical data.
            trigger_event_data (pl.DataFrame): DataFrame containing the trigger event data.
            features_config (pl.DataFrame): DataFrame containing the features configuration.
            tags_config (pl.DataFrame): DataFrame containing the tags configuration.
        """

        self.hist_event_input = hist_event_input
        self.trigger_event_data = trigger_event_data
        self.features_config = features_config
        self.tags_config = tags_config
        self.trigger_features = plant_config.select("*")[0,0].cast(pl.Utf8).to_list()
        self.plantup_time = str(plant_config.select("*")[0,0])
        self.plantup_switch = str(plant_config.select("*")[0,0])
        self.plantup_minutes = str(plant_config.select("*")[0,0])

    def custom_lowercase(self, series: pl.Series) -> pl.Series:
        """
        Convert strings in a Polars Series to lowercase and strip leading and trailing whitespace.

        Args:
            series (pl.Series): The input Series.

        Returns:
            pl.Series: The transformed Series with strings converted to lowercase and stripped of whitespace.
        """
        # To Convert strings to lowercase with strip
        return series.str.to_lowercase().str.strip_chars(" ")


    def get_plant_condition_tags(self, features_config: pl.DataFrame, plant_condition: str) -> list:
        """
        Get a list of plant condition tags from a Polars DataFrame.

        Args:
            features_config (pl.DataFrame): DataFrame containing feature configurations.
            plant_condition (str): The plant condition to filter by.

        Returns:
            list: A list of plant condition tags.
        """
        return features_config.with_columns(id=pl.col("*").cast(pl.Utf8)).filter(pl.col("*") == plant_condition)["*"].to_list()

    def get_specific_day_tags(self, features_config: pl.DataFrame, required_minutes: int):
        """
        Get a list of features requiring specific minutes of data

        Args:
            features_config (pl.DataFrame): DataFrame containing feature configurations.
            required_minutes (int): Minutes we need to have data for the specific features.
        """
        if required_minutes == 2880:
            self.two_day_features = features_config.filter(pl.col("*")==required_minutes)["*"].cast(pl.Utf8).to_list()

    @timer
    def fetch_input_data(self):
        """Get the feature_store_input and the required config files.
        Perform required transformations to use these files in the functions.

        Returns:
            dataframe: dataframe with the feature store input values
            datetime: timestamp of the triggering event
        """
        # make sure the inputs are not empty
        assert not self.trigger_event_data.is_empty(), "Event input is empty"
        try:
            feature_store_input_calc = (
                self.trigger_event_data.explode("*")
                .unnest("*")
                .rename({"*": "*", "*": "*", "*": "*"})
                .with_columns(Value=pl.col("*").cast(pl.Float64))
                .pivot(index=["*", "*"], columns="*", values="*")
                .with_columns([pl.col("*").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%Z", strict=False)])
            )
            feature_store_input_columns = feature_store_input_calc.columns
            feature_store_input = feature_store_input_calc.select(sorted(feature_store_input_columns))
            event_timestamp = feature_store_input["*"][0]
        except Exception as exception:
            logger.error("An error occurred with failure on processing feature store input data: %s", str(exception))
            raise exception
        return feature_store_input, event_timestamp

    @timer
    def process_features_config(self):
        """Perform required transformation on features config.

        Returns:
            dataframe: dataframe with the features config
            list: list of features other than plant conditions
            list: list of plant conditions features
        """
        assert not self.features_config.is_empty(), "Features config is empty"
        try:
            features_config = self.features_config.explode("*").unnest("*")  # Process the features configurations
            # get a list of plant conditions and not plant conditions tags to be used in format results
            no_plant_conditions_tags = self.get_plant_condition_tags(features_config, "false")
            plant_conditions_tags = self.get_plant_condition_tags(features_config, "true")

            # Explode features_config file into multiple rows split by Equipment col
            features_config = (
                features_config.with_columns(pl.col("*").str.split(",").alias("*"))
                .explode("*")
                .select(
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*",
                    "*"
                )
                .rename({"*": "*"})
            )
            lowercase_cols = ["*", "*", "*", "*", "*", "*", "*", "*"]
            features_config = features_config.with_columns(id=pl.col("*").cast(pl.Utf8))
            transformed_columns = {col_name: self.custom_lowercase(features_config[col_name]) for col_name in lowercase_cols}
            features_config = features_config.with_columns(pl.DataFrame(transformed_columns))

        except Exception as exception:
            logger.error("An error occurred with failure on processing feature store input data: %s", str(exception))
            raise exception
        return features_config, no_plant_conditions_tags, plant_conditions_tags

    @timer
    def process_etl_config(self, feature_store_input):
        """Perform required transformations on the etl config.

        Args:
            feature_store_input (DataFrame): DataFrame containing the feature_store input values
        Returns:
            dataframe: dataframe with the tags config
            dataframe: dataframe with the tags config transformed \n
            to be used in evaluate_run_status_function
        """
        assert not self.tags_config.is_empty(), "ETL configuration (tags config) is empty"
        try:
            # process tags config
            tags_config = (
                self.tags_config.explode("*")
                .explode("*")
                .unnest("*")
                .select("*", "*", "*", "*", "*")
                .unnest("*")
                .rename({"*": "*", "*": "*"})
            )

            lowercase_cols = ["*", "*", "*", "*", "*"]
            transformed_columns = {col_name: self.custom_lowercase(tags_config[col_name]) for col_name in lowercase_cols}
            tags_config = tags_config.with_columns(pl.DataFrame(transformed_columns))

            # if a tag from ETL config doesn't exist in the input data, drop it
            feature_store_input_columns_set = set(feature_store_input.columns)
            tags_config = (tags_config.with_columns(
                enterpriseId=pl.col("*").cast(pl.Utf8)).filter(
                    pl.col("*").is_in(feature_store_input_columns_set))
                )

            # Create run_tag_config for on-off logic
            tags_config_run_tags = tags_config.filter(
                (pl.col("*") == "*") & (pl.col("*") == "*")).rename(
                {"*": "*"}
            )
            tags_config_no_run_tags = tags_config.filter(~((pl.col("*") == "*") & (pl.col("*") == "*")))

            tags_config_extended = tags_config_no_run_tags.join(tags_config_run_tags, on="*", how="*")
            run_tag_config = tags_config_extended.group_by(["*"]).agg(pl.col("*").explode())
            run_tag_config = run_tag_config.filter(~pl.col("*").is_null())
            run_tag_config = run_tag_config.with_columns(
                enterpriseId=pl.col("*").cast(pl.List(pl.Utf8)),
                RunTagID=pl.col("*").cast(pl.Utf8),
            )
        except Exception as exception:
            logger.error("Failure on processing the ETL configuration (tags config): %s", str(exception))
            raise exception
        return tags_config, run_tag_config

    @timer
    def combine_configs(self, features_config, tags_config):
        """
        Merge and Perform required transformations to use the configs in the functions.

        Args:
            features_config (DataFrame): DataFrame containing the features configuration.
            tags_config (DataFrame): DataFrame containing the tags configuration.

        Returns:
            dataframe: dataframe with the features config merged \n
            with tags config
        """
        try:
            # Merge the  tags and feature config
            feature_tag_config = features_config.join(
                tags_config,
                on=["*", "*", "*", "*", "*"],
                how="left",
            )
            feature_tag_config = feature_tag_config.group_by(
                ["*", "*", "*", "*", "*", "*", "*", "*", "*"]
            ).agg(pl.col("*").explode())

            feature_tag_config = feature_tag_config.with_columns(
                enterpriseId=pl.col("*").cast(pl.List(pl.Utf8)), customInput=pl.col("*").str.split(",").cast(pl.List(pl.Utf8))
            )

            feature_tag_config = feature_tag_config.sort("*")

        except Exception as exception:
            logger.error("Failure on merging the ETL config and features config: %s", str(exception))
            raise exception

        return feature_tag_config

    @timer
    def trigger_calc_configs(self, features_config, tags_config, run_tag_config, feature_tag_config):
        """
        Create configuration files required for the trigger logic calculations by filtering \n
        the config files for the features required in the trigger calculation.

        Args:
            features_config (DataFrame): DataFrame containing the features configuration.
            tags_config (DataFrame): DataFrame containing the tags configuration.
            run_tag_config (Dataframe): Dataframe with the tags config transformed \n
            to be used in evaluate_run_status_function
            feature_tag_config (Dataframe): Dataframe with the features config merged \n
            with tags config

        Returns:
            trigger_features_config (DataFrame): DataFrame containing the features configuration for trigger features.
            trigger_tags_config (DataFrame): DataFrame containing the tags configuration for tags required for the trigger features.
            trigger_run_tag_config (Dataframe): Dataframe with the tags config transformed \n
            trigger_to be used in evaluate_run_status_function for run tags required for the trigger features.
            trigger_feature_tag_config (Dataframe): Dataframe with the features config merged \n
            with tags config for trigger features.

        """
        # filter the features_config for the trigger features
        trigger_features_config = features_config.filter(pl.col("*").is_in(self.trigger_features))
        # filter the feature_tag_config for the trigger features
        trigger_feature_tag_config = feature_tag_config.filter(pl.col("*").is_in(self.trigger_features))
        # find the tags required for trigger features and filter tags_config for them
        trigger_calc_columns = trigger_feature_tag_config.explode("*").select("*").drop_nulls()[:,0].to_list()
        trigger_tags_config = tags_config.filter(pl.col("*").is_in(set(trigger_calc_columns)))
        # find the run tags required for tags being used in the trigger features if any
        run_tags_list = []
        for run_tag, row in run_tag_config.iter_rows():
            # in the run_tag_config, if there are any trigger tags with a corresponding run tag,
            # append them to run_tags_list and the trigger_calc_columns
            if any(element in trigger_calc_columns for element in row):
                trigger_calc_columns.append(run_tag)
                run_tags_list.append(run_tag)
            if len(run_tags_list) != 0:
                trigger_run_tag_config = run_tag_config.filter(pl.col("*").is_in(run_tags_list))
            else:
                # if there are no required run tags, return None for trigger_run_tag_config
                trigger_run_tag_config = None
        # add the feature ID of the required trigger features to the trigger_calc_columns
        #  to have both features and tags required for triggers
        trigger_feature_id = trigger_feature_tag_config.select("*")[:,0].cast(pl.Utf8).to_list()
        trigger_calc_columns.extend(trigger_feature_id)

        return trigger_features_config, trigger_feature_tag_config, trigger_tags_config, trigger_run_tag_config, trigger_calc_columns

    @timer
    def two_day_configs(self, features_config, tags_config, run_tag_config, feature_tag_config):
        """
        Create configuration files required for the two days features by filtering \n
        the config files for the features required in the two day features.

        Args:
            features_config (DataFrame): DataFrame containing the features configuration.
            tags_config (DataFrame): DataFrame containing the tags configuration.
            run_tag_config (Dataframe): Dataframe with the tags config transformed \n
            to be used in evaluate_run_status_function
            feature_tag_config (Dataframe): Dataframe with the features config merged \n
            with tags config

        Returns:
            two_day_features_config (DataFrame): DataFrame containing the features configuration for two_day features.
            two_day_tags_config (DataFrame): DataFrame containing the tags configuration for tags required for the two_day features.
            two_day_run_tag_config (Dataframe): Dataframe with the tags config transformed \n
            two_day to be used in evaluate_run_status_function for run tags required for the two_day features.
            two_day_feature_tag_config (Dataframe): Dataframe with the features config merged \n
            with tags config for two_day features.

        """
        # filter the features_config for the two_day features
        two_day_features_config = features_config.filter(pl.col("*").is_in(self.two_day_features))
        # filter the feature_tag_config for the two_day features
        two_day_feature_tag_config = feature_tag_config.filter(pl.col("*").is_in(self.two_day_features))
        # find the tags required for two_day features and filter tags_config for them
        two_day_required_columns = two_day_feature_tag_config.explode("*").select("*").drop_nulls()[:,0].to_list()
        two_day_tags_config = tags_config.filter(pl.col("*").is_in(set(two_day_required_columns)))
        # find the run tags required for tags being used in the two_day features if any
        run_tags_list = []
        for run_tag, row in run_tag_config.iter_rows():
            # in the run_tag_config, if there are any two_day tags with a corresponding run tag,
            # append them to run_tags_list and the two_day_required_columns
            if any(element in two_day_required_columns for element in row):
                two_day_required_columns.append(run_tag)
                run_tags_list.append(run_tag)
            if len(run_tags_list) != 0:
                two_day_run_tag_config = run_tag_config.filter(pl.col("*").is_in(run_tags_list))
            else:
                # if there are no required run tags, return None for two_day_run_tag_config
                two_day_run_tag_config = None
        # add the feature ID of the required two_day features to the two_day_required_columns
        #  to have both features and tags required for two_days
        two_day_feature_id = two_day_feature_tag_config.select("*")[:,0].cast(pl.Utf8).to_list()
        two_day_required_columns.extend(two_day_feature_id)

        # drop the two_day feature from the configs used in 2 hrs
        features_config = features_config.filter(~pl.col("*").is_in(self.two_day_features))
        feature_tag_config = feature_tag_config.filter(~pl.col("*").is_in(self.two_day_features))

        return features_config, feature_tag_config, two_day_features_config, two_day_feature_tag_config, two_day_tags_config, two_day_run_tag_config, two_day_required_columns


    @timer
    def fetch_historical_data(self, feature_store_input: pl.DataFrame, event_timestamp):
        """Get the historical events for 2 hours before the feature store input

        Args:
            dataframe: dataframe with the historical events input values
            dataframe: dataframe with the feature_store input values
            event_timestamp: Timestamp from triggering event record

        Returns:
            dataframe: dataframe with the processed historical events for 2 hours
            dataframe: dataframe with the processed historical events for 2 days without plantup_switch column
        """

        # check if any duplicates in in the historian records; if there are deduplicate and warn
        if any(self.hist_event_input.select("*").is_duplicated()):
            self.hist_event_input = self.hist_event_input.unique(subset=["*"])
            logger.warning("Historical data has duplicates!")

        try:
            hist_event_input = (
                self.hist_event_input.rename({"*": "*"})
                .explode("*")
                .select("*","*", "*")
                .unnest("*")  
                .with_columns(
                [pl.col("*").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%Z", strict=False)], value=pl.col("*").cast(pl.Float64))
                .pivot(index=["*", "*"], columns="*", values="*")
            )
            # filter for 2 days and 7 hrs ago
            two_day_offset_datetime = event_timestamp - timedelta(hours=55)
            two_day_hist_event_input =  hist_event_input.filter((pl.col("*") >= two_day_offset_datetime) & (pl.col("*") < event_timestamp))

            # to get the plantup_switch we need to calculate plantup_time for 2.5-2 hrs ago
            offset_datetime_switch = event_timestamp - timedelta(hours=2.5)
            # to filter the hist_event_input
            offset_datetime = event_timestamp - timedelta(hours=2)

            # filter hist_event_input for the last 2.5 hours
            hist_event_input = hist_event_input.filter((pl.col("*") >= offset_datetime_switch) & (pl.col("*") < event_timestamp))

            # if there is no historical data in the last 2.5 hours
            if hist_event_input.is_empty():
                # create an empty polars dataframe for hist_event_input with all the minutes
                empty_df = pl.DataFrame(
                    {
                        "timestamp": pl.datetime_range(
                            start=offset_datetime_switch,
                            end=event_timestamp,
                            interval="1m",
                            eager=True,
                        )
                    }
                )
                # since hist_event_input is empty, join with the event data to get all the required columns and then filter out the event_timestamp
                hist_event_input = (
                    empty_df.join(feature_store_input, on="*", how="left")
                    .filter(pl.col("*") < event_timestamp)
                )

                logger.warning("There are no historical events in the last 2 hours!")
            # if there is not historical data for every minute in the last 2.5 hours
            elif len(hist_event_input) < 150:
                # create an empty polars dataframe for hist_event_input with all the minutes
                empty_df = pl.DataFrame(
                    {
                        "timestamp": pl.datetime_range(
                            start=offset_datetime_switch,
                            end=event_timestamp,
                            interval="1m",
                            eager=True,
                        )
                    }
                )
                # join with the hist_event_input
                hist_event_input = (
                    empty_df.join(hist_event_input, on="*", how="left")
                    .filter(pl.col("*") < event_timestamp)
                )
                logger.warning("Gaps found in the historical data!")

            # make plantUp as 0 for all the null values
            hist_event_input = hist_event_input.with_columns(pl.when(pl.col("*").is_null()).then(0).otherwise(pl.col("*")).alias("*"))

            # calculate the plantUp for 2.5-2 hrs ago and add that column to hist_event_input
            plantup_switch_value = hist_event_input.filter((pl.col("*") >= offset_datetime_switch) & (pl.col("*") < offset_datetime)).select(
                pl.mean("*")
            )[0, 0]

            if plantup_switch_value is None:
                plantup_switch_value = 0

            # add plantup_switch column to hist_event_input indicating plant was switched on 2.5-2 hrs ago
            hist_event_input = hist_event_input.with_columns(pl.when(plantup_switch_value < 0.2).then(1).otherwise(0).alias("*"))

            # if the plant is up less than 20% of the time between 2-2.5 hrs ago log the info
            if plantup_switch_value < 0.2:
                logger.info("Plant was down for more than 80% of the time between 2-2.5 hrs ago.")

            # filter for 2 hrs ago
            hist_event_input = hist_event_input.filter((pl.col("*") >= offset_datetime) & (pl.col("*") < event_timestamp))
            hist_event_input = hist_event_input.select(sorted(hist_event_input.columns))

            # if two_day_hist_event_input, use the created hist_event_input and drop the extra plantup_switch column
            if two_day_hist_event_input.is_empty():
                two_day_hist_event_input = hist_event_input.drop("*")

        except Exception as exception:
            logger.error("An error occurred with failure on fetching and processing the historical data: %s", str(exception))
            raise exception
        return hist_event_input, two_day_hist_event_input

    @timer
    def combine_historical_event_data(self, hist_event_input: pl.DataFrame, feature_store_input: pl.DataFrame, trigger_calc_columns: list, event_timestamp):
        """Combine the historical event data with the feature store event input and add the plantUp features

        Args:
            hist_event_input (dataframe): dataframe with the processed historical events
            feature_store_input (dataframe): dataframe with the feature_store input values
            trigger_calc_columns (List): List of features and tags required for trigger logic calculation
            event_timestamp: Timestamp from triggering event record

        Returns:
            dataframe: dataframe with the feature_store input and the historical data
            dataframe: dataframe with the feature_store input and the historical data
            only for the features and tags required for trigger logic calculation
        """
        try:
            # add plantup_switch column to the feature_store_input with the same value as the plantup_switch in hist_event_input
            feature_store_input = feature_store_input.with_columns(plantup_switch=hist_event_input["*"][0])
            feature_store_input = feature_store_input.select(sorted(feature_store_input.columns))

            # concatenate the event data with the historical data excluding the plantup_switch column
            feature_store_input = pl.concat([feature_store_input, hist_event_input])

            # process plant conditions
            # if the plant is up 80% of the last hour and 80% of the hour before, then plantup_time is 1, otherwise 0
            offset_1hr = event_timestamp - timedelta(hours=1)
            offset_2hr = event_timestamp - timedelta(hours=2)
            plantup_time_1hr = feature_store_input.filter(
                (pl.col("*") >= offset_1hr) & (pl.col("*") < event_timestamp)).select(
                    pl.mean("*"))[0, 0]
            plantup_time_2hr = feature_store_input.filter(
                (pl.col("*") >= offset_2hr) & (pl.col("*") < offset_1hr)).select(
                    pl.mean("*"))[0, 0]

            # add warning messages if plant is up less than 80%
            if plantup_time_1hr <= 0.8:
                logger.warning(f"Plant was not up for more than 80% of the last hour! It has been up for {round(plantup_time_1hr, 2) * 100}% of the last hour.")
            if plantup_time_2hr <= 0.8:
                logger.warning(f"Plant was not up for more than 80% of the hour before! It has been up for {round(plantup_time_2hr, 2) * 100}% of the hour before.")

            # add the column plantup_time to the feature_store_input based on the plantup_time_1hr and plantup_time_2hr
            feature_store_input = feature_store_input.with_columns(
                pl.when((plantup_time_1hr > 0.8) & (plantup_time_2hr > 0.8)).then(1).otherwise(0).alias("plantup_time"))

            # create and add plantup_minutes
            plantup_time_minutes = feature_store_input.filter(
                (pl.col("*") >= offset_1hr) & (pl.col("*") < event_timestamp)).select(
                    pl.sum("*"))[0, 0]
            feature_store_input = feature_store_input.with_columns(plantup_minutes=plantup_time_minutes)

            feature_store_input_all_cols = (
                feature_store_input.rename(
                    {"*": self.plantup_time, "*": self.plantup_switch, "*": self.plantup_minutes})
                .sort("*")
            )
            # filter the trigger_calc_columns to only keep the ones already existing in the input
            #  since some features will be created later in generate_features stage
            trigger_calc_columns = [col for col in trigger_calc_columns if col in feature_store_input_all_cols.columns]
            trigger_calc_columns.append("*")
            trigger_feature_store_input = feature_store_input_all_cols.select(pl.col(trigger_calc_columns))
        except Exception as exception:
            logger.error("An error occurred with failure on combining the event and the historical data: %s", str(exception))
            raise exception
        return feature_store_input_all_cols, trigger_feature_store_input

    @timer
    def two_day_combine_historical_event_data(self, two_day_hist_event_input: pl.DataFrame,
                                             feature_store_input: pl.DataFrame, two_day_required_columns: list):
        """Combine the historical event data with the feature store event input for 2 day features

        Args:
            two_day_hist_event_input (dataframe): dataframe with the processed historical events for 2 days
            feature_store_input (dataframe): dataframe with the feature_store input values
            two_day_required_columns (List): List of features and tags required for two_day features

        Returns:
            dataframe: dataframe with the feature_store input and the historical data of only the two_day features
        """
        try:
            # filter the trigger_calc_columns to only keep the ones already existing in the input
            #  since some features will be created later in generate_features stage
            two_day_required_columns = [col for col in two_day_required_columns if col in feature_store_input.columns]
            two_day_required_columns.append("*")

            feature_store_input = feature_store_input.select(pl.col(sorted(two_day_required_columns)))
            two_day_hist_event_input = two_day_hist_event_input.select(pl.col(sorted(two_day_required_columns)))

            # concatenate the event data with the historical data excluding the plantup_switch column
            two_day_feature_store_input = pl.concat([feature_store_input, two_day_hist_event_input])

            two_day_feature_store_input = (
                two_day_feature_store_input.sort("*")
            )
        except Exception as exception:
            logger.error("An error occurred with failure on combining the event and the historical data: %s", str(exception))
            raise exception
        return two_day_feature_store_input

    def run_preprocess(self):
        logger.info("Started pre-processing the inputs")
        logger.info("Transforming feature store input event")
        feature_store_input, event_timestamp = self.fetch_input_data()
        logger.info("Transforming features config")
        features_config, no_plant_conditions_tags, plant_conditions_tags = self.process_features_config()
        logger.info("Transforming etl config")
        tags_config, run_tag_config = self.process_etl_config(feature_store_input)
        logger.info("Combining features and etl configs")
        feature_tag_config = self.combine_configs(features_config, tags_config)
        logger.info("Creating trigger logic configs")
        (trigger_features_config, trigger_feature_tag_config, trigger_tags_config,
        trigger_run_tag_config, trigger_calc_columns) = self.trigger_calc_configs(
            features_config, tags_config, run_tag_config, feature_tag_config)
        logger.info("Getting two_day features")
        self.get_specific_day_tags(features_config, 2880)
        logger.info("Creating two_day features configs")
        (features_config, feature_tag_config, two_day_features_config, two_day_feature_tag_config,
         two_day_tags_config, two_day_run_tag_config, two_day_required_columns) =self.two_day_configs(
            features_config, tags_config, run_tag_config, feature_tag_config)
        logger.info("Transforming historical input")
        hist_event_input, two_day_hist_event_input = self.fetch_historical_data(feature_store_input, event_timestamp)
        logger.info("Combining feature store event and historical inputs")
        feature_store_input_all_cols, trigger_feature_store_input = self.combine_historical_event_data(
            hist_event_input, feature_store_input, trigger_calc_columns, event_timestamp)
        logger.info("Combining feature store event and historical inputs for two_day features")
        two_day_feature_store_input = self.two_day_combine_historical_event_data(
            two_day_hist_event_input, feature_store_input, two_day_required_columns)
        return (
            feature_store_input_all_cols,
            trigger_feature_store_input,
            features_config,
            trigger_features_config,
            tags_config,
            trigger_tags_config,
            run_tag_config,
            trigger_run_tag_config,
            feature_tag_config,
            trigger_feature_tag_config,
            no_plant_conditions_tags,
            plant_conditions_tags,
            event_timestamp,
            two_day_feature_store_input,
            two_day_features_config,
            two_day_feature_tag_config,
            two_day_tags_config,
            two_day_run_tag_config,
        )
