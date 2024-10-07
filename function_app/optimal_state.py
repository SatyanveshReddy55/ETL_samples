"""This module is for optimal state calculation."""
import polars as pl
from function_app.commons.logger import logger


class OptimalState:
    """
    Class for calculating optimal state based on event data and tags configuration.
    """
    def __init__(self, event_data, tags_config, plant_config):
        self.event_data = event_data
        self.tags_config = tags_config
        self.plant_config = plant_config
        self.optimal_tag_id = plant_config["*"][0]
        self.plant_feed_optimal = plant_config["*"][0]

    def optimal_state(self):
        """
        Calculate optimal state based on event data and tags configuration.

        Returns:
        - polars.DataFrame: DataFrame representing the optimal state.
        """
        if any(self.event_data.select("*").is_duplicated()):
            self.event_data = self.event_data.unique(subset=["*"])
            logger.info("Historical data has duplicates!")
        # Prepare tags configuration DataFrame
        tags_config = (
                        self.tags_config.explode("*")
                        .explode("*")
                        .unnest("*")
                        .select("*", "*", "*", "*", "*")
                        .unnest("*")
                        .rename({"name": "*", "id": "*"})
                    )
        dataset = self.event_data.explode('*').select('*','*','*').unnest('*')
        # Select timestamps with critical values less than or equal to 600
        ts_with_critical_eql_zero_lst = (self.event_data.explode('*').select('*','eventtype','*').unnest('*').filter((pl.col('*') == self.optimal_tag_id) & ( (pl.when(pl.col('*').is_null()).then(0).otherwise(pl.col('*'))) <= self.plant_feed_optimal)).select("*").get_columns()[0] )
        
        # Select enterprise IDs for critical run status tags
        critical_run_status_tags = (
                        tags_config.filter((pl.col("*") == "Critical Run Status"))[
                            "*"
                        ]
                        .to_list())

        # Transform "*" column and create columns : *, *
        optimal_logic_df = (
            dataset.with_columns(
                pl.when(
                    pl.col("*")
                    .is_in(ts_with_critical_eql_zero_lst)
                    .and_(~(pl.col("*").is_in(critical_run_status_tags)))
                )
                .then(pl.lit(None))
                .otherwise(pl.col("*"))
                .alias("*")
            )
            .with_columns(
                pl.when(pl.col("*").is_in(ts_with_critical_eql_zero_lst))
                .then(pl.lit(0))
                .otherwise(pl.lit(1))
                .alias("*")
            )
        )
        # Group by *, *, and *, then aggregate records column
        optimal_logic_df = optimal_logic_df.select('*','*','*',pl.struct('*','*').alias('*'))
        optimal_logic_df = optimal_logic_df.groupby('*','*','*').agg(pl.col('*'))
        optimal_logic_df = optimal_logic_df.sort("*")
        return optimal_logic_df
