"""This module is for feature store transformations."""
import datetime

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from function_app.commons.utils import timer
from function_app.commons.logger import logger

class FeatureStore:
    def __init__(
        self,
        feature_store_input,
        features_config,
        tags_config,
        run_tag_config,
        feature_tag_config,
        plant_config,
        run_type=None
    ):
        self.feature_store_input = feature_store_input
        self.features_config = features_config
        self.tags_config = tags_config
        self.run_tag_config = run_tag_config
        self.feature_tag_config = feature_tag_config
        self.moisture_tag_id = plant_config.select("*")[0,0].cast(pl.Utf8).to_list()
        self.moisture_target_value = plant_config.select("*")[0,0]
        self.skip_low_pass_filter_tags = plant_config.select("*")[0,0].cast(pl.Utf8).to_list()
        self.event_timestamp = self.feature_store_input["*"][-1]
        self.run_type = run_type

    @timer
    def evaluate_run_status_function(self, feature_store_input: pl.DataFrame, run_tag_config: pl.DataFrame):
        """Evaluate the inputs: If a run tag is off, make all the other tags\n
        for the corresponding equioment as Null
        Args:
            dataframe: dataframe with the feature_store input values
            dataframe: dataframe with the tags config transformed to be used in\n
            evaluate_run_status_function

        Returns:
            dataframe: dataframe with applied run logic
        """
        try:
            run_status_output = feature_store_input.clone()

            if run_tag_config is None:
                logger.info("There are no tags with a Run tag to check the run status")
            else:
                # loop through the run tags and if it is off, \n
                # make all the other tags for the corresponding equioment as Null
                # if a tag doesn't have a run_tag nothing happens to it
                for run_tag, row in run_tag_config.iter_rows():
                    if all(element in run_status_output.columns for element in row):
                        run_status_output = run_status_output.with_columns(
                        pl.when(
                        pl.col(run_tag)==1).then(pl.col(row))
                        .otherwise(None))
                    elif any(element in run_status_output.columns for element in row):
                        selected_columns = [x for x in row if row in run_status_output.columns]
                        run_status_output = run_status_output.with_columns(
                        pl.when(
                        pl.col(run_tag)==1).then(pl.col(selected_columns))
                        .otherwise(None))
                    else:
                        # if there are no run tags to check, skip this step
                        logger.info(f"Run tag {run_tag} does not exist in the data to check the run status")
        except Exception as exception:
            logger.error("An error occurred with failure on evaluating run status: %s", str(exception))
            raise exception
        return run_status_output

    @timer
    def evaluate_low_pass_filter_function(self, run_status_output: pl.DataFrame, tag_config: pl.DataFrame):
        """Apply the low pass filter to the input
        Args:
            dataframe: dataframe output from evaluate_run_status_function
            dataframe: dataframe with the tags config

        Returns:
            dataframe: dataframe with applied low pass filter

        """
        try:
            low_pass_tags = (
                tag_config.filter((pl.col("tagType") == "*") & (~pl.col("*").is_in(self.skip_low_pass_filter_tags))
                                  )["*"].to_list())
            low_pass_output = (
                run_status_output.clone().sort("*").set_sorted("*")
            )
            # check if there are any columns to apply the low pass filter on them
            input_columns_set = set(low_pass_output.columns)
            low_pass_tags_set = set(low_pass_tags)
            if input_columns_set & low_pass_tags_set:
                # due to Polars limitation on rolling median of data with null values, we need to convert to pandas
                pd_low_pass_output = low_pass_output.to_pandas().set_index("*")
                pd_low_pass_output[low_pass_tags] = (
                    pd_low_pass_output[low_pass_tags]
                    .rolling(window="15min", min_periods=7, center=True)
                    .median()
                )
                low_pass_output = pl.from_pandas(pd_low_pass_output, include_index=True)
            else:
                logger.info("There are no tags requiring low-pass filter step.")
        except Exception as exception:
            logger.error("An error occurred with failure on evaluating low pass filter: %s", str(exception))
            raise exception
        return low_pass_output

    @timer
    def thickener_underflow_mass_rate(self, dataframe: pl.DataFrame, column: str, feature: str):
        """Process the thickener underflow features
        Args:
            dataframe: dataframe with the low pass filter values
            column: column defining the enterpriseId
            feature: feature id 

        Returns:
            dataframe: dataframe with new ad-hoc(thickener underflow) features
        """
        try:
            dataframe = dataframe.with_columns(
                density_flow_1=dataframe[(column[0]).replace(" ","")]
                * dataframe[(column[1]).replace(" ","")]
            )
            dataframe = dataframe.with_columns(
                density_flow_2=dataframe[(column[2]).replace(" ","")]
                * dataframe[(column[3]).replace(" ","")]
            )
            dataframe = dataframe.with_columns(
                pl.coalesce(
                    pl.col("*"), pl.col("*")
                ).alias(feature)
            )
        except Exception as exception:
            logger.error("An error occurred with failure on thickener_underflow_mass_rate calculation: %s", str(exception))
            raise exception
        return dataframe

    @timer
    def moisture_target(self, dataframe: pl.DataFrame, target_value: float, feature: str):
        """
        add the moisture target to the dataframe from the defined value in the config file
        Args: 
            dataframe: dataframe
            target_value: moisture target value
            feature: column name of the moisture target
            
        Returns:
            dataframe: dataframe with the moisture target
        """
        try:
            dataframe = dataframe.with_columns(pl.lit(target_value).alias(feature))
        except Exception as exception:
            logger.error("An error occurred with failure on adding moisture_target: %s", str(exception))
            raise exception
        return dataframe

    @timer
    def greater_function(self, dataframe: pl.DataFrame, threshold_value: float, input_column: str, output_column: str):
        """
        Creates a boolean feature based on the threshold_value defined in the config file
        Args: 
            dataframe: dataframe
            threshold_value: threshold value
            input_column: tag or feature id to be compared against the threshold value
            output_column: column name of the created feature
            
        Returns:
            dataframe: dataframe with the new boolean feature
        """
        try:
            # if the tag value is greater than the threshold value, the feature is set to 1, otherwise 0
            dataframe = dataframe.with_columns(pl.when(
                                    pl.col(input_column) >= int(threshold_value))
                                    .then(1)
                                    .otherwise(0)
                                    .alias(output_column))
        except Exception as exception:
            logger.error("An error occurred with failure on greater function: %s", str(exception))
            raise exception
        return dataframe

    @timer
    def passthrough(self, dataframe: pl.DataFrame, column: str, feature: str):
        """Passthrough data for a combination of tags and fill na values with 0
        Args:
            dataframe: dataframe with the low pass filter values
            column: column defining the enterpriseId
            feature: feature id 

        Returns:
            dataframe: dataframe with new passthrough features as list
        """
        try:
            dataframe = dataframe.with_columns(
                dataframe.select(pl.concat_list(pl.col(column)).alias(feature))
            )
        except Exception as exception:
            logger.error("An error occurred with failure on generating tags data as list: %s", str(exception))
            raise exception
        return dataframe

    @timer
    def evaluate_feature_function(self, low_pass_output: pl.DataFrame, feature_tag_config: pl.DataFrame):
        """Create new features by applying different aggregations
        Args:
            dataframe: dataframe output from evaluate_low_pass_filter_function
            dataframe: dataframe with the features config merged \n
            with tags config

        Returns:
            dataframe: dataframe with new features by applying functions

        """
        try:
            # make sure the low_pass_output is not empty
            assert not low_pass_output.is_empty(), "The low_pass_output is empty"

            # create feature tag config to be used for this function
            feature_tag_config_function = feature_tag_config.select(
                "*", "*", "*", "*"
            )

            feature_function_output = low_pass_output.clone()

            for row in feature_tag_config_function.iter_rows(named=True):
                # function being mean and there are tags to take the mean of them
                if row["*"] == "*" and row["*"] != [None]:
                    feature_function_output = feature_function_output.with_columns(
                        pl.concat_list(row["*"]).list.mean().alias(row["id"])
                    )
                # function being max and there are tags to take the max of them
                elif row["*"] == "*" and row["*"] != [None]:
                    feature_function_output = feature_function_output.with_columns(
                        pl.concat_list(row["*"]).list.max().alias(row["*"])
                    )
                # function being thickener_underflow_mass_rate which are ad-hoc
                elif row["*"] == "*":
                    feature_function_output = self.thickener_underflow_mass_rate(feature_function_output, row["*"], row["*"])
                # function being moisture_target
                elif row["*"] == "*":
                    feature_function_output = self.moisture_target(feature_function_output, self.moisture_target_value, row["*"])
                elif row["*"] == "*":
                    feature_function_output = self.greater_function(feature_function_output, row["*"][0], row["*"], row["*"])
                #function being a passthrough for tags data to reach timeaggregation
                elif row["*"] == "*":
                    feature_function_output = self.passthrough(feature_function_output, row["*"], row["*"])

            # get a list of all the columns needed in featurestore output
            selected_columns = feature_tag_config_function["*"].to_list()
            selected_columns.append("*")
            if "*" in feature_function_output.columns:
                selected_columns.append("*")
            # select all the columns needed in featurestore output
            feature_function_output = feature_function_output.select(selected_columns)
        except Exception as exception:
            logger.error("An error occurred with failure on evaluate feature function: %s", str(exception))
            raise exception
        return feature_function_output

    @timer
    def evaluate_null_impute(self, feature_function_output: pl.DataFrame, feature_tag_config: pl.DataFrame):
        """Impute null values of features if specified in nullImpute field in features config.
        Null imputation only happens when the plant is up
        Args:
            dataframe: dataframe output from evaluate_feature_function
            dataframe: dataframe with the features config merged \n
            with tags config
        
        Returns:
            dataframe: dataframe with specific features being imputed
        """
        feature_null_impute_output = feature_function_output.clone()

        for row in feature_tag_config.iter_rows(named=True):
            # if there is a value in null_impute and the plantUp column exists in the data
            if row["*"] is not None and "*" in feature_null_impute_output.columns:
                # get a list of columns with list datatype
                list_type_cols = feature_null_impute_output.select(
                    pl.col(pl.List(float)), pl.col(pl.List(int))).columns
                # if a feature is of type List, fill the null values existing in the list
                if row["*"] in list_type_cols:
                    # get a list of columns required for grouping by to reconstruct the feature as a list
                    grouping_cols = feature_null_impute_output.drop(row["*"]).columns
                    # explode the list of tags, if the plant is up, impute the nulls and then reconstruct the feature as a list
                    feature_null_impute_output = feature_null_impute_output.explode(row["*"]).with_columns(
                        pl.when(
                            pl.col("*")==1).then(
                                pl.col(row["*"]).fill_null(row["*"])).otherwise(
                                    pl.col(row["*"]))
                                    ).group_by(grouping_cols).agg(pl.col(row["*"]).explode())
                # if the plant is up replace the null values with that value; otherwise keep it as null
                feature_null_impute_output = feature_null_impute_output.with_columns(
                        pl.when(
                            pl.col("*")==1).then(
                                pl.col(row["*"]).fill_null(row["*"])).otherwise(
                                    pl.col(row["*"])))
        # get the feature store columns to sort the dataframe                            
        feature_store_input_columns = feature_null_impute_output.columns
        feature_null_impute_output = feature_null_impute_output.sort("*").select(
            sorted(feature_store_input_columns)).drop("*")
        return feature_null_impute_output

    @timer
    def evaluate_current_lag(self, feature_null_impute_output: pl.DataFrame, feature_tag_config: pl.DataFrame):
        """Lag the features
        Args:
            dataframe: dataframe output from evaluate_null_impute
            dataframe: dataframe with the features config merged \n
            with tags config

        Returns:
            dataframe: dataframe with current features being lagged

        """
        try:
            # make sure the feature_null_impute_output is not empty
            assert not feature_null_impute_output.is_empty(), "The data from evaluate_null_impute step (feature_null_impute_output) is empty"

            # create feature tag config to be used for this function
            features_config_lag = feature_tag_config.select(
                "*", "*"
            )

            feature_current_lag_output = feature_null_impute_output.clone()

            for row in features_config_lag.iter_rows(named=True):
                if row["*"] is not None:
                    feature_current_lag_output = feature_current_lag_output.with_columns(
                        pl.col(row["*"]).shift(int(row["*"])).alias(row["*"])
                    )
        except Exception as exception:
            logger.error("An error occurred with failure on evaluate lag: %s", str(exception))
            raise exception
        return feature_current_lag_output

    def evaluate_anticipated_lag(self, feature_null_impute_output: pl.DataFrame, feature_tag_config: pl.DataFrame):
        """Lag the anticipated features for only the features with lag more than 60 minutes 
        Args:
            dataframe: dataframe output from evaluate_null_impute
            dataframe: dataframe with the features config merged \n
            with tags config
 
        Returns:
            dataframe: dataframe with anticipated features being lagged
 
        """
        try:
            # make sure the feature_null_impute_output is not empty
            assert not feature_null_impute_output.is_empty(), "The data from evaluate_null_impute step (feature_null_impute_output) is empty"

            # create feature tag config for which lag value is more than 60 to be used for this function
            features_config_lag = feature_tag_config.filter(pl.col("*") > 60).select("*", "*")
            feature_anticipated_lag_output = feature_null_impute_output.clone()

            # for rows where lag value is more than 60, substract the lag value by 60 for lagging the anticipated features
            for row in features_config_lag.iter_rows(named=True):
                if row["*"] is not None:
                    feature_anticipated_lag_output = feature_anticipated_lag_output.with_columns(
                        pl.col(row["id"]).shift(int(row["*"])-60).alias(row["*"])
                    )
        except Exception as exception:
            logger.error("An error occurred with failure on evaluate anticipated lag: %s", str(exception))
            raise exception
        return feature_anticipated_lag_output

    def torque_constraint(self, input_series: pd.Series, aggregation_minutes: int):
        """
        Determine whether thickener torque warrants imposing a constraint on plant feed.
        Trend is calculated by estimating whether the 95% CI for an observation an hour in the future\n
         based on a linear regression of the past two hours overlaps with the mean value over the last two hours:
        if the whole CI is above the past mean, trend is up, if it"s below then trend is down and if there is\n
          overlap there is no clear trend.
        Constraint levels:
            1: (torque is between high and high-high and trend is up) OR (torque is above high-high and trend is down)
            2: torque is above high-high and trend is not down
            0: otherwise
        """
        try:
            high_limit = 15
            high_high_limit = 20

            # Instead of 120T, use 119T
            aggregation_minutes = str(aggregation_minutes - 1)+ "T"

            # make sure to have exactly 120 minutes of data if there are missing rows
            series = input_series.resample("1T").mean().loc[input_series.index[-1] - pd.Timedelta(aggregation_minutes):]
            assert len(series) == 120, "The length of data required for torque calculation is not 120"
            if series.isna().sum() > 0.2*120: #if too many null values, trend cannot be determined
                return np.nan
            series = series.interpolate() #statsmodels can't deal with nulls
            series = series.values
            series_arange = np.arange(len(series))
            assert series.shape == (120, ), "The series in torque_constraint does not have 120 rows"

            current = np.nanmean(series)
            if current < high_limit:
                return 0

            fit = sm.OLS(series, sm.add_constant(series_arange)).fit()
            prediction = fit.get_prediction([1, 120+60]).summary_frame()

            if prediction["*"].values[0] >= current:
                trend = 1
            elif prediction["*"].values[0] <= current:
                trend = -1
            else:
                trend = 0

            if current < high_high_limit:
                if trend == 1:
                    torque = 1
                else:
                    torque = 0
            else:
                if trend == -1:
                    torque = 1
                else:
                    torque = 2
        except Exception as exception:
            logger.error("An error occurred with failure on torque constraint calculation: %s", str(exception))
            raise exception
        return torque

    @timer
    def ratio(self, input_series: pd.DataFrame, aggregation_minutes: int):
        """
        Determine the ratio over the last 60mins
        """
        try:
            input_series = pd.DataFrame(input_series[input_series.columns[0]].tolist(), columns=["*","*"],index = input_series.index, )
            input_series = input_series.tail(aggregation_minutes)
            if input_series["*"].mean() == 0:
                ratio = None
            else:
                ratio = input_series["*"].mean()/input_series["*"].mean()
        except Exception as exception:
            logger.error("An error occurred with failure on ratio calculation: %s", str(exception))
            raise exception
        return ratio

    @timer
    def evaluate_null_threshold(
        self,
        feature_function_output: pl.DataFrame,
        feature_tag_config: pl.DataFrame
    ):
        """Evaluate features to have null values less than the defined null threshold\n
        and raise exception if not
        Args:
            dataframe: dataframe output from evaluate_feature_function
            dataframe: dataframe with the features config merged \n
            with tags config
        
        """
        try:
            # get the null count of each feature
            df_nulls = feature_function_output.null_count()
            # get the list of columns which have some null values
            col_list = [ col.name for col in df_nulls.select(df_nulls[0] > 0) if col.all() ]
            # filter the feature_tag_config for features which have null values
            filtered_feature_tag_config = feature_tag_config.filter(pl.col("*").is_in(col_list))
            null_list = []
            # loop through the filtered feature_tag_config and check if the null threshold is met
            for row in filtered_feature_tag_config.iter_rows(named=True):
                null_list.extend(col.name for col in df_nulls.select(
                    pl.col(row["*"])/len(feature_function_output) > row["*"]) if col.all())
            # if there are features with nulls more than the null threshold, raise exception
            if len(null_list) > 0:
                # for trigger_check run, it only logs the msg
                if self.run_type == "Trigger_check":
                    logger.info(f"Trigger_check: There are features with nulls more than the null threshold: {null_list}")
                else:
                    # for runs other than Trigger_check, it sends error
                    logger.error(f"There are features with nulls more than the null threshold: {null_list}")
        except Exception as exception:
            logger.error("An error occurred with failure on evaluate null threshold: %s", str(exception))
            raise exception

    @timer
    def evaluate_time_aggregation(
        self,
        feature_current_lag_output: pl.DataFrame,
        feature_anticipated_lag_output: pl.DataFrame,
        feature_tag_config: pl.DataFrame
    ):
        """Aggregate for a time
        Args:
            dataframe: dataframe output from evaluate_current_lag for current state
            dataframe: dataframe output from feature_function \n
            to be used for anticipated state
            dataframe: dataframe with the features config merged \n
            with tags config
        Returns:
            dataframe: dataframe with new features being aggregated \n
            for 1 hour for current state
            dataframe: dataframe with new features being aggregated \n
            for 1 hour for anticipated state
        """
        try:
            # make sure the feature_current_lag_output is not empty
            assert not feature_current_lag_output.is_empty(), "The data from evaluate_current_lag step (feature_current_lag_output) is empty"
            # for current state, take the feature_current_lag_output
            feature_current_output = (
                feature_current_lag_output.clone().to_pandas().set_index("*")
            ).sort_index()
            # for anticipated state, take the feature_function_output
            feature_anticipated_output = (
                feature_anticipated_lag_output.clone().to_pandas().set_index("*")
            ).sort_index()
            # make sure the columns are the same between currentFeatureRecords and anticipatedFeatureRecords
            if not sorted(list(feature_current_output.columns)) == sorted(list(feature_anticipated_output.columns)):
                raise Exception("Features mismatch between currentFeatureRecords and anticipatedFeatureRecords")
            # select required columns from feature_tag_config
            # due to polars limitation on resample function (missing origin parameter),
            #   it is converted to pandas
            features_config = feature_tag_config.select(
                "*", "*","*", "*", "*"
            ).to_pandas()
            feature_current_output_df_dict = feature_current_output.copy()
            feature_anticipated_output_df_dict = feature_anticipated_output.copy()
            feature_current_output_dict = {}
            feature_anticipated_output_dict = {}
            # loop over features_config rows and apply the aggregation function
            for _feature, row in features_config.iterrows():
                # create duration minutes for the aggregation time
                duration_mins = str(row["*"])+ "min"
                # applying to current output
                if str(row["*"]).lower() == "*":
                    feature_current_output_dict[row["*"]] = feature_current_output_df_dict[row["*"]].resample(
                        duration_mins, origin="end").mean().iloc[-1]
                elif str(row["*"]).lower() == "*":
                    feature_current_output_dict[row["*"]] = feature_current_output_df_dict[row["*"]].resample(
                        duration_mins, origin="end").last().iloc[-1]
                elif str(row["*"]).lower() == "*":
                    feature_current_output_dict[row["*"]] = self.torque_constraint(
                        feature_current_output_df_dict[row["*"]], row["*"])
                elif str(row["*"]).lower() == "*":
                    feature_current_output_dict[row["*"]] = self.ratio(
                        feature_current_output_df_dict[[row["*"]]], row["*"])
                elif str(row["*"]).lower() == "*":
                    feature_current_output_df_dict[row["*"]] = feature_current_output_df_dict[row["*"]].resample(
                        duration_mins, origin="end").mean().iloc[-1]
                    feature_current_output_pl = pl.from_pandas(feature_current_output_df_dict, include_index=True).tail(1)
                    feature_current_output_dict[row["*"]] = self.greater_function(feature_current_output_pl, row["*"][0], row["*"], row["*"]).select(row["*"])[0,0]
                elif row["*"] is None:
                    feature_current_output_dict[row["*"]] = np.nan
                # applying to anticipated output
                if str(row["*"]).lower() == "*":
                    feature_anticipated_output_dict[row["*"]] = feature_anticipated_output_df_dict[row["*"]].resample(
                        duration_mins, origin="end").mean().iloc[-1]
                elif str(row["*"]).lower() == "*":
                    feature_anticipated_output_dict[row["*"]] = feature_anticipated_output_df_dict[row["*"]].resample(
                        duration_mins, origin="end").last().iloc[-1]
                elif row["*"] is None:
                    feature_anticipated_output_dict[row["*"]] = np.nan

            # Append timestamp to dict
            feature_current_output_dict["*"] = self.event_timestamp
            feature_anticipated_output_dict["*"] = self.event_timestamp

            feature_anticipated_output = pl.from_dict(feature_anticipated_output_dict)
            feature_current_output = pl.from_dict(feature_current_output_dict)
        except Exception as exception:
            logger.error("An error occurred with failure on evaluate time aggregation: %s", str(exception))
            raise exception

        return feature_current_output, feature_anticipated_output

    def run_feature_store(self):
        """run feature store tasks to get all the current and anticipated features
        Args:

        Returns:
            dataframe: dataframe with current features
            dataframe: dataframe with anticipated features
        """
        logger.info("Started Generating Features")
        try:
            now = datetime.datetime.now()
            logger.info("Evaluating On/Off logic for Run Status tags")
            run_status_output = self.evaluate_run_status_function(
                self.feature_store_input, self.run_tag_config
            )
            logger.info("Evaluating Low Pass Filter")
            low_pass_output = self.evaluate_low_pass_filter_function(
                run_status_output, self.tags_config
            )
            # Evaluate Current & Anticiapted features
            logger.info("Evaluating Feature fuctions")
            feature_function_output = self.evaluate_feature_function(
                low_pass_output, self.feature_tag_config
            )
            # Evaluate Null Threshold of features
            logger.info("Evaluating Null Threshold")
            self.evaluate_null_threshold(
                feature_function_output, self.feature_tag_config
            )
            # Evaluate Null imputation of features
            logger.info("Evaluating Null Imputation")
            feature_null_impute_output = self.evaluate_null_impute(
                feature_function_output, self.feature_tag_config
            )
            logger.info("Evaluating Feature current lag")
            feature_current_lag_output = self.evaluate_current_lag(
                feature_null_impute_output, self.feature_tag_config
            )
            logger.info("Evaluating Feature anticipated lag")
            feature_anticipated_lag_output = self.evaluate_anticipated_lag(
                feature_null_impute_output, self.feature_tag_config
            )
            logger.info("Evaluating time aggregation")
            feature_current_output, feature_anticipated_output = self.evaluate_time_aggregation(
                feature_current_lag_output,
                feature_anticipated_lag_output,
                self.feature_tag_config,
            )
            logger.info(
                f"Total time taken for evaluating Feature Store results : {datetime.datetime.now() - now}"
            )
        except Exception as exception:
            logger.error("An error occurred with failure on run feature store: %s", str(exception))
            raise exception
        return feature_current_output, feature_anticipated_output
