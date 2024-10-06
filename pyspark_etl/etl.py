"""
This module includes methods required by ETL process
"""

from datetime import date, timedelta
from delta.tables import DataFrame
from pyspark.sql.functions import col
from utils.blobconnector import Blobconnector
from utils.etllogging import etl_logger

from pyspark_etl.variables import (
    container_account,
    container_name,
    delta_database,
    delta_table,
    sample_table_container,
    sample_container_account)

logger = etl_logger()

#list of tables to be loaded
tables_list = [
    "test",
    "test_historical",
    "result",
    "result_historical",
    "analysis",
    "goal"]

rename_columns = ["field_1", "field_2", "field_3", "field_4", "field_5"]

partition_cols = ["field_date"]

def column_rename(dataframe, columns, prefix):
    """
    Renaming columns in dataframe
    Args:
        dataframe: spark dataframe
        columns: columns list to be renamed
        prefix: prefix for the new cloumn name

    Returns: spark dataframe
    """
    for column in columns:
        dataframe = dataframe.withCoulmnRenamed(column, prefix + "_" + column)
    return dataframe

class ETL(Blobconnector):
    """
    Class includes methods for etl process
    """

    def __init__(self):
        super().__init__()

    def etl_processor(self):
        """
        join tables and create final dataframe to upload
        Returns: spark dataframe
        """
        # load the sample table
        logger.info("Reading data from sample table in a container.")
        sample_df = self.read_table(
            "sample", sample_table_container, sample_container_account)

        list_dfs = []
        for table in tables_list:
            lst = self.read_table(table, container_name, container_account)
            logger.info("Reading data from %s table", table)
            list_dfs.append(lst)

        
        # load the test table
        refined_test = (column_rename(list_dfs[0], rename_columns, "TEST")
                        .withColumnRenamed("SAMPLE_ID", "TEST_SAMPLE_ID")
                        .withColumnRenamed("*", "TEST_*")
                        .withColumnRenamed("**", "TEST_**")
                        )

        refined_historical_test = (column_rename(list_dfs[1], rename_columns, "TEST")
                        .withColumnRenamed("SAMPLE_ID", "TEST_SAMPLE_ID")
                        .withColumnRenamed("*", "TEST_*")
                        .withColumnRenamed("**", "TEST_**")
                        )
        
        union_test = refined_test.union(refined_historical_test).drop(col('**_id')).distinct()

        # load the result table
        refined_result = (column_rename(list_dfs[2], rename_columns, "RESULT")
                        .withColumnRenamed("TEST_ID", "RESULT_TEST_ID")
                        .withColumnRenamed("*", "RESULT_*")
                        .withColumnRenamed("**", "RESULT_**")
                        )

        refined_historical_result = (column_rename(list_dfs[3], rename_columns, "RESULT")
                        .withColumnRenamed("TEST_ID", "RESULT_TEST_ID")
                        .withColumnRenamed("*", "RESULT_*")
                        .withColumnRenamed("**", "RESULT_**")
                        )
        
        union_result = refined_result.union(refined_historical_result).drop(col('**_id')).distinct()

        # load the analysis table
        refined_analysis = (column_rename(list_dfs[4], rename_columns, "ANALYSIS")
                        .withColumnRenamed("aid", "ANALYSIS_aid")
                        .withColumnRenamed("*", "ANALYSIS_*")
                        .withColumnRenamed("**", "ANALYSIS_**")
                        .distinct()
                        )
        
        # load the goal table
        refined_goal = (column_rename(list_dfs[5], rename_columns, "GOAL")
                        .withColumnRenamed("", "GOAL_id")
                        .withColumnRenamed("*", "GOAL_*")
                        .withColumnRenamed("**", "GOAL_**")
                        .distinct()
                        )

        logger.info("joining sample and test tables")
        aggregated_df = (
            sample_df\
                .join(union_test, (sample_df["SAMPLE_ID"] == union_test["TEST_SAMPLE_ID"], "left"))
                .join(union_result, (union_test["test_id"] == union_result["RESULT_TEST_ID"]), "left")
                .join(refined_analysis, (union_result["RESULT_*"] == refined_analysis["ANALYSIS_*"]))
            )

        final_df = aggregated_df.dropDuplicates().join(
            refined_goal, ["*_id", "*_id"], "outer"
            )
        return final_df
    
    @staticmethod
    def filter_date_window(
        dataframe: DataFrame, load_method: str, filtered_records=None) -> DataFrame:
        """
        This method is used to filter the dataframe based on load_method
        Args:
            dataframe: dataframe
            load_method: str
            filtered_records: dataframe

        Returns: spark dataframe
        """
        if load_method == "full":
            filtered_records = dataframe
        elif load_method == "overwrite":
            try:
                latest_date = date.today()
                previous_date = latest_date - timedelta(weeks=265)
                logger.info(f"loading the data from:{previous_date}")
                filtered_records = dataframe.filter(
                    col("sample_date").between(previous_date, latest_date)
                )
            except Exception as e:
                logger.error(e)
                raise

        elif load_method == "incremental":
            try:
                latest_date = date.today()
                previous_date = latest_date - timedelta(days=15)
                logger.info(f"loading the data from:{previous_date}")
                filtered_records = dataframe.filter(
                    col("sample_date").between(previous_date, latest_date)
                )
            except Exception as e:
                logger.error(e)
                raise e
        else:
            logger.error("Incorrect load-method specified")
        
        return filtered_records
    

    def load_selection(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on load selection returns dataframe
        Args:
            dataframe: spark dataframe

        Returns: spark dataframe
        """
        if self.spark_session._jsparkSession.catalog().tableExists(
            delta_database, delta_table):
            filtered_records = self.filter_date_window(dataframe, load_method="incremental")
            logger.info("Incremental load is selected")
        else:
            filtered_records = self.filter_date_window(dataframe, load_method="full")
            logger.info("Full load is selected")

        return filtered_records
