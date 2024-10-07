"""This module is for feature store transformations."""
from datetime import timedelta, datetime, timezone
import os
import json
import polars as pl
import azure.functions as func

from function_app.commons.eventManipulation import EventManipulation
from function_app.generate_features import FeatureStore
from function_app.trigger import TriggersCheck
from function_app.io.blob import BlobWrite
from function_app.commons.logger import logger
from function_app.preprocess import PreProcess
from function_app.results import Results
from function_app.local_testing import LocalTesting
from function_app.optimal_state import OptimalState
from function_app.commons.utils import read_json_file, validate_schema, within_time_range
import function_app.commons.globals as gl


def main(event: func.EventHubEvent):
    logger.info("Starting Function App")
    logger.info("Reading config files")
    # Read features config
    features_config = pl.read_json(os.path.join(os.getcwd(), "configurations/datafiles/featuresConfig.json"))
    # Read tags configurations
    tags_config = pl.read_json(os.path.join(os.getcwd(), "configurations/datafiles/etlconfiguration.json"))
    # Read plant conditions config
    plant_config = pl.read_json(os.path.join(os.getcwd(), "configurations/datafiles/plantConfig.json"))

    if isinstance(event, func.EventHubEvent):
        logger.info("Event trigger function processed a request.")
        event_hub_input_connection_string = os.environ["EventHubConnectionStringTrigger"]
        event_hub_output_connection_string = os.environ["EventHubConnectionStringOutput"]
        feature_store_output_eh = os.environ["OUTPUT_EH_NAME"]
        # Blob
        sa_conn_string = os.environ["SA_CONN_STRING"]
        sa_container_name = os.environ["SA_CONTAINER_NAME"]

        # event manipulation object
        event_object = EventManipulation(event)

        # Validate trigger event
        try:
            input_schema = read_json_file("input_schema.json")
            validate_schema(event_object.get_event_as_json_dict(), input_schema)
            logger.info("Schema validation successful for trigger event.")
        except Exception as exception:
            logger.error(f"Trigger event schema is incorrect: {exception}")
            raise

        # Event Data to Polars Dataframe
        event_data = event_object.get_event_as_dataframe()
        # Optimal State Calculation
        if event_object.get_event_as_json_dict()['eventtype'] == "eventtype1":
            trigger_event_optim_state = OptimalState(event_data, tags_config, plant_config)
            event_data = trigger_event_optim_state.optimal_state()
        logger.info(f"Trigger Event: {event_data}")

        # Append trigger event data into blob
        try:
            data_to_append = json.dumps(event_data.to_dicts()[0])
            blob_name = event_data["eventtype"][0]
            blob_client = BlobWrite(sa_conn_string, sa_container_name)
            blob_client.append_data(data=data_to_append, blob_name=blob_name)
        except Exception as exception:
            logger.error(f"Failed to append data in blob: {exception}")

        logger.info("Triggering feature store")

        # Read all data from blob container
        # change key if the eventtype is updated
        blob_data = blob_client.read_all_blobs()
        if "eventtype1" in blob_data:
            hist_event_input = blob_data["eventtype1"]
        else:
            logger.info("No historical eventtype1 data available.")
            # Stop processing

        if "eventtype2" in blob_data:
            eventtype2_data_blob = blob_data["eventtype2"]
        else:
            eventtype2_data_blob = None

        if len(hist_event_input) <= 1:
            logger.info("Historical eventtype1 data is not available.")
            logger.info("Data not available to Calculate the features.")
            logger.info("Stopping feature store")
            return None

    else:
        logger.info("Inside else")
        logger.info("Running feature store locally")
        gl.DEBUG_MODE = False
        local_instance = LocalTesting()
        event_data, hist_event_input, eventtype2_data_blob = local_instance.read_data()
        # check if any duplicates in in the eventtype1 records; if there are deduplicate and warn
        hist_event_optim_state = OptimalState(hist_event_input, tags_config, plant_config)
        hist_event_input = hist_event_optim_state.optimal_state()

    try:
        # Read input data
        logger.info("Starting Pre-processing the inputs")
        # create historical and event dataframes
        hist_event_input = hist_event_input.sort("*")
        trigger_event_data = hist_event_input.tail(1)
        hist_event_input = hist_event_input.head(-1)
        preprocess_input = PreProcess(hist_event_input, trigger_event_data, features_config, tags_config, plant_config)
        (
            feature_store_input,
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
        ) = preprocess_input.run_preprocess()
        no_plant_conditions_tags.append("*")
        plant_conditions_tags.append("*")
        logger.info("Pre-processing the inputs completed")

        # Run for trigger conditions
        trigger_conditions = TriggersCheck(
            trigger_feature_store_input,
            trigger_features_config,
            trigger_tags_config,
            trigger_run_tag_config,
            trigger_feature_tag_config,
            plant_config,
            eventtype2_data_blob,
        )
        trigger, flag = trigger_conditions.trigger_run()
        # if trigger conditions are not met, do not proceed
        if trigger == 0:
            logger.info("Trigger conditions are not met.")
            logger.info("Stopping feature store pipeline")
        # if trigger conditions are met, proceed to the next steps
        else:
            logger.info("Trigger conditions are succesfully met.")
            # Evaluate feature store output
            logger.info("Starting Generating Features")
            eval_features = FeatureStore(feature_store_input, features_config, tags_config, run_tag_config, feature_tag_config, plant_config)
            (feature_current_output, feature_anticipated_output) = eval_features.run_feature_store()
            # Generate 2 day features
            two_day_eval_features = FeatureStore(
                two_day_feature_store_input, two_day_features_config, two_day_tags_config, two_day_run_tag_config, two_day_feature_tag_config, plant_config
            )
            (two_day_feature_current_output, two_day_feature_anticipated_output) = two_day_eval_features.run_feature_store()

            feature_current_output = feature_current_output.join(two_day_feature_current_output, on="*", how="inner")
            feature_anticipated_output = feature_anticipated_output.join(two_day_feature_anticipated_output, on="*", how="inner")
            logger.info("Features Generation completed")

            # Format feature store output
            logger.info("Starting formatting output")
            process_results = Results(
                feature_current_output, feature_anticipated_output, no_plant_conditions_tags, plant_conditions_tags, event_timestamp, flag
            )
            feature_store_output = process_results.format_results()
            logger.info("Feature Store format output completed")

            if isinstance(event, func.EventHubEvent):
                logger.info("Publishing the Feature Store results to EventHub")
                # Send the Feature store output to event hub
                process_results.save_results(feature_store_output, event_hub_output_connection_string, feature_store_output_eh)
            else:
                logger.info("Publishing the Feature Store results locally")
                # Save the Feature store output locally
                process_results.save_results(feature_store_output)
            logger.info("Feature Store output saved successfully")
            logger.info("Exiting the Feature Store")
    except Exception as exception:
        feature_store_output = None
        logger.error(exception)

    return None


if __name__ == "__main__":
    local_run = LocalTesting()
    event, hist_event_input, eventtype2_data_blob = local_run.read_data()
    main(event)
    logger.info(f"Feature Store results: Done")
