""" module to test the feature store"""
import os
import re
from datetime import datetime, timedelta, timezone
import polars as pl
import pytest
from function_app.generate_features import FeatureStore
from function_app.preprocess import PreProcess
from function_app.results import Results
from function_app.commons.logger import logger
from function_app.trigger import TriggersCheck

# import json
# to enable catching all the logs
logger.propagate = True

@pytest.fixture
def setup():
    '''creates the setup for the testing data

    Args:
        None

    Returns:
        dataframe: dataframe with the test feature_store input values
        dataframe: dataframe with the test features config
        dataframe: dataframe with the test tags config
        dataframe: dataframe with the test tags config transformed \n
        to be used in evaluate_run_status_function
        dataframe: dataframe with the test features config merged \n
        with tags config
        list: list of features other than plant conditions
        list: list of plant conditions features
    '''
    # Read the features configurations
    features_config = pl.read_json(os.path.join(os.getcwd(), "../function_app_tests/test_data/featuresConfig.json"))
    # Read tags configurations
    tags_config = pl.read_json(os.path.join(os.getcwd(), "../function_app_tests/test_data/etlconfiguration.json"))
    # Read mock input for short period
    hist_event_input = pl.read_json(os.path.join(os.getcwd(), "../function_app_tests/test_data/mock_historical_data.json"))
    #get event and historical data
    hist_event_input = hist_event_input.head(-1)
    trigger_event_data = hist_event_input.tail(1)
    # Read plant conditions config
    plant_config = pl.read_json(os.path.join(os.getcwd(), "../function_app_tests/test_data/plantConfig.json"))
    eventtype2_data = pl.read_json(os.path.join(os.getcwd(), "../function_app_tests/test_data/mock_eventtype2_data.json"))

    # Read mock input for two days
    hist_event_input_two_days = pl.read_json(os.path.join(os.getcwd(), "../function_app_tests/test_data/mock_two_days_data.json"))
    #get event and historical data
    hist_event_input_two_days = hist_event_input_two_days.head(-1)
    trigger_event_data_two_days = hist_event_input_two_days.tail(1)

    return(
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        hist_event_input_two_days,
        trigger_event_data_two_days,
        plant_config,
        eventtype2_data
    )


def test_evaluate_run_status_function(setup):
    '''test the evaluate_run_status_function\n
    for a run tag and 2 dependent tags
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, run_tag_config, _trigger_run_tag_config, feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    eval_features = FeatureStore(feature_store_input_all_cols, features_config, tags_config, run_tag_config, feature_tag_config, plant_config)
    run_status_output = eval_features.evaluate_run_status_function(
        feature_store_input_all_cols, run_tag_config
    ).filter(pl.col('timestamp')==datetime(2023, 10, 1, 6, 30)).select('*', '*', '*')

    output_dict = run_status_output.to_dict(as_series=False)
    expected_data = {'*': [0.0], '*': [None], '*': [None] }
    assert output_dict == expected_data



def test_evaluate_low_pass_filter_function(setup):
    '''test the evaluate_low_pass_filter_function\n
      for a numeric and a discrete tag
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input,  trigger_event_data,  features_config, tags_config, plant_config)
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, run_tag_config, _trigger_run_tag_config, feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()
    eval_features = FeatureStore(
        feature_store_input_all_cols, features_config, tags_config, run_tag_config, feature_tag_config, plant_config
    )
    low_pass_output = eval_features.evaluate_low_pass_filter_function(
        feature_store_input_all_cols, tags_config
    )
    outcome_value_numeric = low_pass_output.filter(pl.col('timestamp')==datetime(2023, 10, 1, 6, 30)).select('*')[0].item()
    #discrete value should not be changed by applying low pass function
    outcome_value_discrete = low_pass_output.filter(pl.col('timestamp')==datetime(2023, 10, 1, 6, 30)).select('*')[0].item()
    assert outcome_value_numeric == 95.121219281
    assert outcome_value_discrete == 1.0


def test_evaluate_feature_function(setup):
    '''test the evaluate_feature_function for\n
    '*': tag with a mean function
    '****': moistureTarget tag with evaluate_feature_function
    '****': plantUptime tag
    '****': plantup_minutes
    '****': torque_100 with mean function
    '*': tag with mean function
    '**': tag with mean function
    '**': ad-hoc tag with thickener_underflow_mass_rate function
    '*': tag with mean function
    '**': tag with max function
    '**': tag with greater function

    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, run_tag_config, _trigger_run_tag_config, feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()
    eval_features = FeatureStore(
        feature_store_input_all_cols, features_config, tags_config, run_tag_config, feature_tag_config, plant_config
    )
    feature_function_output = eval_features.evaluate_feature_function(
        feature_store_input_all_cols, feature_tag_config
    )
    feature_function_output = feature_function_output.filter(pl.col('timestamp')==datetime(2023, 10, 1, 6, 30))
    feature_function_output = feature_function_output.drop('*')
    output_dict = feature_function_output.to_dict(as_series=False)

    expected_data = {"*": [56.23232341], "****": [5.0], "****": [8.1], "****": [1], "****": [16.0291201274],
                      "****": [0], "****": [[97.3993008931, 200.0]], "****": [60], "**": [17.0331903458],
                       "*": [88.6470565796], "**": [None], "**": [290.7223383175876], "**": [0], "*": [1.0], '*': [1]}
    assert output_dict == expected_data

def test_evaluate_null_impute(setup):
    '''test the evaluate_null_impute function
    '**': tag with null imputation
    '****': list type tag with null
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, run_tag_config, _trigger_run_tag_config, feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()
    eval_features = FeatureStore(
        feature_store_input_all_cols, features_config, tags_config, run_tag_config, feature_tag_config, plant_config
    )
    feature_function_output = eval_features.evaluate_feature_function(
        feature_store_input_all_cols, feature_tag_config
    )
    # make a value in feature 1011 tags as null
    feature_function_output= feature_function_output.with_columns(pl.when(pl.col('timestamp')==datetime(2023, 10, 1, 6, 30)).then([None, None]).otherwise(pl.col('1011')).alias('1011'))
    # Evaluate Null imputation of features
    feature_null_impute_output = eval_features.evaluate_null_impute(
        feature_function_output, feature_tag_config
    )
    # assert that feature 42 does not have any nulls after the null impute
    assert feature_null_impute_output.select('**').select(pl.all().is_null().sum())[0,0] == 0

    feature_null_impute_output = feature_null_impute_output.filter(pl.col('timestamp')==datetime(2023, 10, 1, 6, 30))
    feature_null_impute_output = feature_null_impute_output.drop('*')
    output_dict = feature_null_impute_output.to_dict(as_series=False)

    expected_data = {"*": [89.12832913], "****": [8.6], "****": [8.1], "****": [1], "****": [16.0291201274],
                    "****": [0], "****": [[0, 0]], "****": [60], "**": [17.0331903458], "*": [89.6470565796],
                    "**": [0.0], "**": [234.12893123], "**": [0], "*": [1.0]}
    assert output_dict == expected_data

def test_evaluate_current_lag(setup):
    '''test the evaluate_current_lag function
    '*': tag with no lag
    '****': moistureTarget tag with no lag
    '****': plantUptime tag with no lag
    '****': plantup_minutes with no lag
    '****': torque_100 with no lag
    '*': tag with 1 hour lag
    '**': tag with no lag
    '**': ad-hoc tag with no lag
    '*': tag with no lag
    '**': tag with no lag
    '**': tag with greater function
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, run_tag_config, _trigger_run_tag_config, feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()
    eval_features = FeatureStore(
        feature_store_input_all_cols, features_config, tags_config, run_tag_config, feature_tag_config, plant_config
    )
    feature_function_output = eval_features.evaluate_feature_function(
        feature_store_input_all_cols, feature_tag_config
    )
    feature_null_impute_output = eval_features.evaluate_null_impute(
        feature_function_output, feature_tag_config
    )
    feature_lag_output = eval_features.evaluate_current_lag(feature_null_impute_output, feature_tag_config)
    feature_lag_output = feature_lag_output.filter(pl.col('timestamp')==datetime(2023, 10, 1, 6, 30))
    feature_lag_output = feature_lag_output.drop('*')
    output_dict = feature_lag_output.to_dict(as_series=False)

    expected_data = {"*": [67.6470565796], "****": [8.6], "****": [8.1], "****": [1], "****": [19.0291201274],
                      "****": [0], "****": [[97.3993008931, 200.0]], "****": [60], "**": [19.0331903458],
                        "*": [67.6470565796], "**": [0.0], "**": [361.7223383175876], "**": [0], "*": [1.0]}
    assert output_dict == expected_data



def test_evaluate_time_aggregation(setup):
    '''test evaluate_time_aggregation function
    '*': tag with 60 min mean aggregation
    '****': moistureTarget tag with 60 min mean aggregation
    '****': plantUptime tag with 120 min mean aggregation
    '****': plantup_minutes with 60 min mean aggregation
    '****': torque_100 with 120 min mean aggregation
    '*': tag with 60 min mean aggregation
    '**': tag with 60 min mean aggregation
    '**': ad-hoc tag with 60 min mean aggregation
    '*': tag with 60 min mean aggregation
    '**': tag with 60 min mean_greater aggregation
    '**': tag with greater function
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, run_tag_config, _trigger_run_tag_config, feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()
    eval_features = FeatureStore(
        feature_store_input_all_cols, features_config, tags_config, run_tag_config, feature_tag_config, plant_config
    )
    feature_function_output = eval_features.evaluate_feature_function(
        feature_store_input_all_cols, feature_tag_config
    )
    feature_null_impute_output = eval_features.evaluate_null_impute(
        feature_function_output, feature_tag_config
    )
    feature_current_lag_output = eval_features.evaluate_current_lag(
        feature_null_impute_output, feature_tag_config
    )

    feature_anticipated_lag_output = eval_features.evaluate_anticipated_lag(
        feature_null_impute_output, feature_tag_config
    )
    (
        feature_current_output, feature_anticipated_output
    ) = eval_features.evaluate_time_aggregation(
        feature_current_lag_output,
        feature_anticipated_lag_output,
        feature_tag_config
    )
    # Current Features
    feature_current_output = feature_current_output.filter(pl.col('timestamp')==datetime(2023, 10, 1, 6, 30))
    feature_current_output = feature_current_output.drop('timestamp')
    output_current_dict = feature_current_output.to_dict(as_series=False)
    expected_current_data = {"*": [67.6470565796], "****": [8.6], "****": [8.1], "****": [1.0], "****": [1],
                            "****": [0.0], "****": [0.4788443991449083], "****": [60.0], "**": [1],
                            "*": [67.6470565796], "**": [0.0], "**": [361.7492071832892], "**": [0.6333333333333333], "*": [1.0]}

    # Anticipated Features
    feature_anticipated_output = feature_anticipated_output.filter(pl.col('timestamp')==datetime(2023, 10, 1, 6, 30))
    feature_anticipated_output = feature_anticipated_output.drop('timestamp')
    feature_anticipated_output = feature_anticipated_output.fill_nan(None)
    output_anticipated_dict = feature_anticipated_output.to_dict(as_series=False)

    expected_anticipated_data = {'*': [None], '****': [None], '****': [None], '****': [None], '****': [None],
                                  '****': [None],'****': [None], '****': [None], '**': [17.923515004195],
                                    '*': [67.6470565796], '**': [0.0], '**': [361.7492071832892], '**': [0], '*': [1.0]}
    assert output_current_dict == expected_current_data
    assert output_anticipated_dict == expected_anticipated_data


def test_run_feature_store(setup):
    '''test run_feature_store function
    '''
    (
        features_config,
        tags_config,
        _hist_event_input,
        _trigger_event_data,
        hist_event_input_two_days,
        trigger_event_data_two_days,
        plant_config,
        eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input_two_days, trigger_event_data_two_days,  features_config, tags_config, plant_config)
    (feature_store_input, trigger_feature_store_input, features_config, trigger_features_config,
    tags_config, trigger_tags_config, run_tag_config, trigger_run_tag_config, feature_tag_config,
    trigger_feature_tag_config, no_plant_conditions_tags, plant_conditions_tags, event_timestamp,
    two_day_feature_store_input, two_day_features_config, two_day_feature_tag_config,
    two_day_tags_config, two_day_run_tag_config
    ) = preprocess_input.run_preprocess()
    no_plant_conditions_tags.append('*')
    plant_conditions_tags.append('*')

    trigger_conditions = TriggersCheck(
            trigger_feature_store_input, trigger_features_config,
            trigger_tags_config, trigger_run_tag_config,
            trigger_feature_tag_config, plant_config, eventtype2_data)
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
        two_day_eval_features = FeatureStore(two_day_feature_store_input, two_day_features_config, two_day_tags_config,
                                              two_day_run_tag_config, two_day_feature_tag_config, plant_config)
        (two_day_feature_current_output, two_day_feature_anticipated_output) = two_day_eval_features.run_feature_store()

        feature_current_output = feature_current_output.join(two_day_feature_current_output, on="*", how="inner")
        feature_anticipated_output = feature_anticipated_output.join(two_day_feature_anticipated_output, on="*", how="inner")

        process_results = Results(feature_current_output, feature_anticipated_output, no_plant_conditions_tags,
                                   plant_conditions_tags, event_timestamp, flag)
        feature_store_output = process_results.format_results()

    expected_feature_store_output = "{}"
    assert expected_feature_store_output == feature_store_output

def test_features_mismatch(setup):
    '''test if the number of features between current and anticipated features match in run_feature_store function
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, run_tag_config, _trigger_run_tag_config, feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()
    eval_features = FeatureStore(
        feature_store_input_all_cols, features_config, tags_config, run_tag_config, feature_tag_config, plant_config
    )
    feature_function_output = eval_features.evaluate_feature_function(
        feature_store_input_all_cols, feature_tag_config
    )
    feature_null_impute_output = eval_features.evaluate_null_impute(
        feature_function_output, feature_tag_config
    )
    feature_current_lag_output = eval_features.evaluate_current_lag(
        feature_null_impute_output, feature_tag_config
    )
    feature_anticipated_lag_output = eval_features.evaluate_anticipated_lag(
        feature_null_impute_output, feature_tag_config
    )
    # dropping a column from feature_anticipated_lag_output to raise the Exception
    feature_anticipated_lag_output = feature_anticipated_lag_output.drop('1')

    with pytest.raises(Exception, match=re.escape("Features mismatch between currentFeatureRecords and anticipatedFeatureRecords")):
        eval_features.evaluate_time_aggregation(
        feature_current_lag_output,
        feature_anticipated_lag_output,
        feature_tag_config
    )

def test_duplicate_hist_event_input(setup, caplog):
    '''
    test if having duplicates in the historical events input raises a warning message
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    # add a duplicate record
    output_extra = hist_event_input.filter(pl.col('timestamp')==str(datetime(2023, 10, 1, 6, 0).strftime('%Y-%m-%dT%H:%M:%S.%fZ')))
    hist_event_input.extend(output_extra)
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    # Call the function that triggers the warning
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, _run_tag_config, _trigger_run_tag_config, _feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    # test if it got deduplicated
    assert not any(feature_store_input_all_cols.select('timestamp').is_duplicated())
    # test if the number of rows for feature_store_input are 121 at the end
    assert len(feature_store_input_all_cols) == 121
    # Check if the warning message is captured in caplog
    assert "Historical data has duplicates!" in [rec.message for rec in caplog.records]


def test_empty_hist_event_input(setup, caplog):
    '''
    test if not having an historical event in the past 2 hours raises a warning message
    '''
    (
        features_config,
        tags_config,
        _hist_event_input,
        _trigger_event_data,
        hist_event_input_two_days,
        trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    # make the hist_event_input empty
    hist_event_input_empty = hist_event_input_two_days.clear(n=0)
    preprocess_input = PreProcess(hist_event_input_empty, trigger_event_data_two_days, features_config, tags_config, plant_config)

    # Call the function that triggers the warning
    (feature_store_input, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, _run_tag_config, _trigger_run_tag_config, _feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()
    # test if the number of rows for feature_store_input is 121 at the end
    assert len(feature_store_input) == 121

    # Check if the warning message is captured in caplog to warn about no data
    assert "There are no historical events in the last 2 hours!" in [rec.message for rec in caplog.records]


def test_gap_hist_event_input(setup, caplog):
    '''
    test if having duplicates in the historical events input raises a warning message
    '''
    # first scenario
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    # make a gap at the beginning of the hist_event_input
    hist_event_input_gap = hist_event_input.filter(pl.col('timestamp')>str(datetime(2023, 10, 1, 6, 0).strftime('%Y-%m-%dT%H:%M:%SZ')))
    preprocess_input = PreProcess(hist_event_input_gap, trigger_event_data, features_config, tags_config, plant_config)
    # Call the function that triggers the warning
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, _run_tag_config, _trigger_run_tag_config, feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    # test if the number of rows for feature_store_input are 121 at the end
    assert len(feature_store_input_all_cols) == 121
    # Check if the warning message is captured in caplog to warn about no data
    assert "Gaps found in the historical data!" in [rec.message for rec in caplog.records]
    # second scenario
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    # make a gap at the end of the hist_event_input
    hist_event_input_gap_2 = hist_event_input.filter(pl.col('timestamp')<str(datetime(2023, 10, 1, 6, 0).strftime('%Y-%m-%dT%H:%M:%SZ')))
    # Call the function that triggers the warning
    preprocess_input = PreProcess(hist_event_input_gap_2, trigger_event_data, features_config, tags_config, plant_config)
    # Call the function that triggers the warning
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, run_tag_config, _trigger_run_tag_config, feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    # test if the number of rows for feature_store_input are 121 at the end
    assert len(feature_store_input_all_cols) == 121
    # Check if the warning message is captured in caplog to warn about no data
    assert "Gaps found in the historical data!" in [rec.message for rec in caplog.records]

    # third scenario
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    # make a gap in the middle of the hist_event_input
    hist_event_input_gap = hist_event_input.filter(
        (pl.col('timestamp')>str(datetime(2023, 10, 1, 5, 0).strftime('%Y-%m-%dT%H:%M:%SZ')))
        | (pl.col('timestamp')<str(datetime(2023, 10, 1, 4, 30).strftime('%Y-%m-%dT%H:%M:%SZ'))))
    # Call the function that triggers the warning
    preprocess_input = PreProcess(hist_event_input_gap, trigger_event_data, features_config, tags_config, plant_config)
    # Call the function that triggers the warning
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, run_tag_config, _trigger_run_tag_config, feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()
    # test if the number of rows for feature_store_input are 121 at the end
    assert len(feature_store_input_all_cols) == 121
    # Check if the warning message is captured in caplog to warn about no data
    assert "Gaps found in the historical data!" in [rec.message for rec in caplog.records]

def test_format_results_inputs(setup):
    '''
    test if the error messages for inputs of format_results being None or empty are raised
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data, features_config, tags_config, plant_config)
    (feature_store_input_all_cols, trigger_feature_store_input, features_config, trigger_features_config,
    tags_config, trigger_tags_config, run_tag_config, trigger_run_tag_config, feature_tag_config,
    trigger_feature_tag_config, no_plant_conditions_tags, plant_conditions_tags, event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    no_plant_conditions_tags.append('*')
    plant_conditions_tags.append('*')

    trigger_conditions = TriggersCheck(
            trigger_feature_store_input, trigger_features_config,
            trigger_tags_config, trigger_run_tag_config,
            trigger_feature_tag_config, plant_config, eventtype2_data)
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
        eval_features = FeatureStore(feature_store_input_all_cols, features_config, tags_config, run_tag_config, feature_tag_config, plant_config)
        (feature_current_output, feature_anticipated_output) = eval_features.run_feature_store()

        process_results = Results(None, feature_anticipated_output, no_plant_conditions_tags, plant_conditions_tags, event_timestamp, flag)
        with pytest.raises(AssertionError, match="The current features is None"):
            process_results.format_results()

        process_results = Results(feature_current_output, None, no_plant_conditions_tags, plant_conditions_tags, event_timestamp, flag)
        with pytest.raises(AssertionError, match="The anticipated features is None"):
            process_results.format_results()
        # make current input empty
        feature_current_output_empty = feature_current_output.clear(n=0)
        process_results = Results(feature_current_output_empty, feature_anticipated_output,
                                   no_plant_conditions_tags, plant_conditions_tags, event_timestamp, flag)
        with pytest.raises(AssertionError, match="The current features is empty"):
            process_results.format_results()
        # make anticipated input empty
        feature_anticipated_output_empty = feature_anticipated_output.clear(n=0)
        process_results = Results(feature_current_output, feature_anticipated_output_empty,
                                   no_plant_conditions_tags, plant_conditions_tags, event_timestamp, flag)
        with pytest.raises(AssertionError, match="The anticipated features is empty"):
            process_results.format_results()


def test_fetch_input_data_empty(setup):
    '''test if the error messages for inputs of fetch_input_data being empty are raised
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup

    # make event_data empty
    trigger_event_data_empty = trigger_event_data.clear(n=0)
    preprocess_input = PreProcess(hist_event_input, trigger_event_data_empty, features_config, tags_config, plant_config)

    with pytest.raises(AssertionError, match="Event input is empty"):
        preprocess_input.run_preprocess()

    # make features_config empty
        (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    features_config_empty = features_config.clear(n=0)
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config_empty, tags_config, plant_config)
    with pytest.raises(AssertionError, match="Features config is empty"):
        preprocess_input.run_preprocess()

    # make tags_config empty
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup
    tags_config_empty = tags_config.clear(n=0)
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config_empty, plant_config)
    with pytest.raises(AssertionError, match=re.escape("ETL configuration (tags config) is empty")):
        preprocess_input.run_preprocess()


def test_plantUptime_conditions(setup, caplog):
    '''
    test if having less than 80% of plantUp raises a warning message
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup

    # make some of the plantUp values to be 0 making the plantUptime less than 80% in the last hour
    # make 30 minutes plantUp value as 0 for the last hour
    hist_event_input = hist_event_input.with_columns(pl.when(pl.col('timestamp')>'2023-10-01T06:00:00.000000Z')
                                                    .then(0).otherwise(pl.col('*'))
                                                    .alias("*"))
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, _run_tag_config, _trigger_run_tag_config, _feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    # check if the warning message is captured in caplog to warn about Plant was not up for more than 80% in the last hour
    assert "Plant was not up for more than 80% of the last hour! It has been up for 52.0% of the last hour." in [rec.message for rec in caplog.records]
    # check if plantUptime (*) is changed to 0
    assert feature_store_input_all_cols.select("*")[0,0] == 0

    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        _eventtype2_data
    ) = setup

    # make 30 minutes plantUp value as 0 for the hour before
    hist_event_input = hist_event_input.with_columns(pl.when(pl.col('timestamp')<'2023-10-01T05:00:00.000000Z')
                                                    .then(0).otherwise(pl.col('*'))
                                                    .alias("*"))
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (feature_store_input_all_cols, _trigger_feature_store_input, features_config, _trigger_features_config,
    tags_config, _trigger_tags_config, _run_tag_config, _trigger_run_tag_config, _feature_tag_config,
    _trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()
    # check if the warning message is captured in caplog to warn about Plant was not up for more than 80% in the hour before
    assert "Plant was not up for more than 80% of the hour before! It has been up for 50.0% of the hour before." in [rec.message for rec in caplog.records]
    # check if plantUptime (*) is changed to 0
    assert feature_store_input_all_cols.select("*")[0,0] == 0


def test_triggers_plantup_check(setup):
    '''
    Test plant up for triggers
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (_feature_store_input_all_cols, trigger_feature_store_input, features_config, trigger_features_config,
    tags_config, trigger_tags_config, _run_tag_config, trigger_run_tag_config, _feature_tag_config,
    trigger_feature_tag_config, no_plant_conditions_tags, plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    no_plant_conditions_tags.append('*')
    plant_conditions_tags.append('*')

    trigger_conditions = TriggersCheck(
            trigger_feature_store_input, trigger_features_config,
            trigger_tags_config, trigger_run_tag_config,
            trigger_feature_tag_config, plant_config, eventtype2_data)

    # make plantup as 0
    trigger_conditions.feature_current_output = trigger_conditions.feature_current_output.with_columns(pl.lit(0).alias('****'))
    trigger, flag = trigger_conditions.trigger_run()

    assert trigger == 0
    assert flag is None

def test_triggers_eventtype2_check(setup):
    '''
    Test eventtype2 change conditions for triggers
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (_feature_store_input_all_cols, trigger_feature_store_input, features_config, trigger_features_config,
    tags_config, trigger_tags_config, _run_tag_config, trigger_run_tag_config, _feature_tag_config,
    trigger_feature_tag_config, no_plant_conditions_tags, plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    no_plant_conditions_tags.append('*')
    plant_conditions_tags.append('*')

    # Get the current time in UTC timezone
    current_time_utc = datetime.now(timezone.utc)

    # Subtract 5 hours from the current time
    time_5_hours_ago = current_time_utc - timedelta(hours=5)

    eventtype2_data = eventtype2_data.with_columns(timestamp = time_5_hours_ago)
    eventtype2_data = eventtype2_data.with_columns(pl.col("timestamp").cast(pl.Datetime).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    trigger_conditions = TriggersCheck(
            trigger_feature_store_input, trigger_features_config,
            trigger_tags_config, trigger_run_tag_config,
            trigger_feature_tag_config, plant_config, eventtype2_data)
    trigger_conditions.plantup_check()
    # if there is feature store output (trigger_enabled is still 1, go to the next check)
    if trigger_conditions.trigger_enabled == 1:
        # check the eventtype2 change
        trigger_conditions.eventtype2_check()
    assert trigger_conditions.trigger_enabled == 1
    assert trigger_conditions.flag is None

    # Subtract 3 hours from the current time
    time_3_hours_ago = current_time_utc - timedelta(hours=3)

    eventtype2_data = eventtype2_data.with_columns(timestamp = time_3_hours_ago)
    eventtype2_data = eventtype2_data.with_columns(pl.col("timestamp").cast(pl.Datetime).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    trigger_conditions = TriggersCheck(
            trigger_feature_store_input, trigger_features_config,
            trigger_tags_config, trigger_run_tag_config,
            trigger_feature_tag_config, plant_config, eventtype2_data)
    trigger_conditions.plantup_check()
    # if there is feature store output (trigger_enabled is still 1, go to the next check)
    if trigger_conditions.trigger_enabled == 1:
        # check the eventtype2 change
        trigger_conditions.eventtype2_check()
    assert trigger_conditions.trigger_enabled == 1
    assert trigger_conditions.flag is True

    # Subtract 1 hour from the current time
    time_1_hour_ago = current_time_utc - timedelta(hours=1)

    eventtype2_data = eventtype2_data.with_columns(timestamp = time_1_hour_ago)
    eventtype2_data = eventtype2_data.with_columns(pl.col("timestamp").cast(pl.Datetime).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    trigger_conditions = TriggersCheck(
            trigger_feature_store_input, trigger_features_config,
            trigger_tags_config, trigger_run_tag_config,
            trigger_feature_tag_config, plant_config, eventtype2_data)
    trigger_conditions.plantup_check()
    # if there is feature store output (trigger_enabled is still 1, go to the next check)
    if trigger_conditions.trigger_enabled == 1:
        # check the eventtype2 change
        trigger_conditions.eventtype2_check()
    assert trigger_conditions.trigger_enabled == 0
    assert trigger_conditions.flag is None


def test_triggers_switch_check(setup):
    '''
    Test eventtype2 change conditions for triggers
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (_feature_store_input_all_cols, trigger_feature_store_input, features_config, trigger_features_config,
    tags_config, trigger_tags_config, _run_tag_config, trigger_run_tag_config, _feature_tag_config,
    trigger_feature_tag_config, no_plant_conditions_tags, plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    no_plant_conditions_tags.append('*')
    plant_conditions_tags.append('*')

    # Get the current time in UTC timezone
    current_time_utc = datetime.now(timezone.utc)

    # Subtract 5 hours from the current time
    time_5_hours_ago = current_time_utc - timedelta(hours=5)

    eventtype2_data = eventtype2_data.with_columns(timestamp = time_5_hours_ago)
    eventtype2_data = eventtype2_data.with_columns(pl.col("timestamp").cast(pl.Datetime).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    trigger_conditions = TriggersCheck(
            trigger_feature_store_input, trigger_features_config,
            trigger_tags_config, trigger_run_tag_config,
            trigger_feature_tag_config, plant_config, eventtype2_data)

    # make plantup_switch as 1
    trigger_conditions.feature_current_output = trigger_conditions.feature_current_output.with_columns(pl.lit(1).alias('*'))

    trigger, flag = trigger_conditions.trigger_run()

    assert trigger == 1
    assert flag is True

def test_triggers_moisture_check(setup):
    '''
    Test eventtype2 change conditions for triggers
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (_feature_store_input_all_cols, trigger_feature_store_input, features_config, trigger_features_config,
    tags_config, trigger_tags_config, _run_tag_config, trigger_run_tag_config, _feature_tag_config,
    trigger_feature_tag_config, no_plant_conditions_tags, plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    no_plant_conditions_tags.append('*')
    plant_conditions_tags.append('*')

    # Get the current time in UTC timezone
    current_time_utc = datetime.now(timezone.utc)

    # Subtract 5 hours from the current time
    time_5_hours_ago = current_time_utc - timedelta(hours=5)

    eventtype2_data = eventtype2_data.with_columns(timestamp = time_5_hours_ago)
    eventtype2_data = eventtype2_data.with_columns(pl.col("timestamp").cast(pl.Datetime).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    trigger_conditions = TriggersCheck(
            trigger_feature_store_input, trigger_features_config,
            trigger_tags_config, trigger_run_tag_config,
            trigger_feature_tag_config, plant_config, eventtype2_data)

    # make plantup_switch as 0
    trigger_conditions.feature_current_output = trigger_conditions.feature_current_output.with_columns(pl.lit(0).alias('****'))
    # have moisture difference of more than 0.3 pp
    trigger_conditions.feature_current_output = trigger_conditions.feature_current_output.with_columns(pl.lit(8.1).alias('****'))
    trigger_conditions.feature_current_output = trigger_conditions.feature_current_output.with_columns(pl.lit(7.1).alias('****'))

    trigger, flag = trigger_conditions.trigger_run()

    assert trigger == 1
    assert flag is True


def test_triggers_moisture_check_2(setup):
    '''
    Test eventtype2 change conditions for triggers
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (_feature_store_input_all_cols, trigger_feature_store_input, features_config, trigger_features_config,
    tags_config, trigger_tags_config, _run_tag_config, trigger_run_tag_config, _feature_tag_config,
    trigger_feature_tag_config, no_plant_conditions_tags, plant_conditions_tags, _event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    no_plant_conditions_tags.append('*')
    plant_conditions_tags.append('*')

    # Get the current time in UTC timezone
    current_time_utc = datetime.now(timezone.utc)

    # Subtract 5 hours from the current time
    time_5_hours_ago = current_time_utc - timedelta(hours=5)

    eventtype2_data = eventtype2_data.with_columns(timestamp = time_5_hours_ago)
    eventtype2_data = eventtype2_data.with_columns(pl.col("timestamp").cast(pl.Datetime).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    trigger_conditions = TriggersCheck(
            trigger_feature_store_input, trigger_features_config,
            trigger_tags_config, trigger_run_tag_config,
            trigger_feature_tag_config, plant_config, eventtype2_data)

    # make plantup_switch as 0
    trigger_conditions.feature_current_output = trigger_conditions.feature_current_output.with_columns(pl.lit(0).alias('****'))
    # have moisture difference of less than 0.3 pp
    trigger_conditions.feature_current_output = trigger_conditions.feature_current_output.with_columns(pl.lit(8.1).alias('****'))
    trigger_conditions.feature_current_output = trigger_conditions.feature_current_output.with_columns(pl.lit(8.0).alias('****'))

    trigger, flag = trigger_conditions.trigger_run()

    assert trigger == 1
    assert flag is False



def test_null_threshold(setup, caplog):
    '''test if having nulls more than the threshold raises error
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (feature_store_input_all_cols, trigger_feature_store_input, features_config, trigger_features_config,
    tags_config, trigger_tags_config, run_tag_config, trigger_run_tag_config, feature_tag_config,
    trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    trigger_conditions = TriggersCheck(
            trigger_feature_store_input, trigger_features_config,
            trigger_tags_config, trigger_run_tag_config,
            trigger_feature_tag_config, plant_config, eventtype2_data)
    trigger, _flag = trigger_conditions.trigger_run()
    # if trigger conditions are not met, do not proceed
    if trigger == 0:
        logger.info("Trigger conditions are not met.")
        logger.info("Stopping feature store pipeline")
    # if trigger conditions are met, proceed to the next steps
    else:
        logger.info("Trigger conditions are succesfully met.")
        # Evaluate feature store output
        logger.info("Starting Generating Features")
        eval_features = FeatureStore(feature_store_input_all_cols, features_config, tags_config, run_tag_config, feature_tag_config, plant_config)
        feature_function_output = eval_features.evaluate_feature_function(
        feature_store_input_all_cols, feature_tag_config
        )
        # make some data points null for features
        # make a feature null for 30 minutes (this shouldn't raise an exception since it results in 25% nulls)
        offset_half_hr = event_timestamp - timedelta(hours=0.5)
        feature_function_output = feature_function_output.with_columns(pl.when(pl.col('timestamp')>offset_half_hr)
                                                    .then(None).otherwise(pl.col('*'))
                                                    .alias('*'))

        # make 2 features null for 2 hours (this should raise an exception since it results in more than 60% nulls)
        feature_function_output = feature_function_output.with_columns(pl.when(pl.col('timestamp')<offset_half_hr)
                                                    .then(None).otherwise(pl.col('**'))
                                                    .alias('**'))
        feature_function_output = feature_function_output.with_columns(pl.when(pl.col('timestamp')<offset_half_hr)
                                                    .then(None).otherwise(pl.col('**'))
                                                    .alias('**'))

        _output = eval_features.evaluate_null_threshold(
                feature_function_output, feature_tag_config
            )

        for rec in caplog.records:
            print(rec.message)
    assert "There are features with nulls more than the null threshold: ['**', '**']" in [rec.message for rec in caplog.records]


def test_null_threshold_trigger_tag(setup, caplog):
    '''test if having nulls more than the threshold raises error for trigger tags
    '''
    (
        features_config,
        tags_config,
        hist_event_input,
        trigger_event_data,
        _hist_event_input_two_days,
        _trigger_event_data_two_days,
        plant_config,
        eventtype2_data
    ) = setup
    preprocess_input = PreProcess(hist_event_input, trigger_event_data,  features_config, tags_config, plant_config)
    (_feature_store_input_all_cols, trigger_feature_store_input, features_config, trigger_features_config,
    tags_config, trigger_tags_config, _run_tag_config, trigger_run_tag_config, _feature_tag_config,
    trigger_feature_tag_config, _no_plant_conditions_tags, _plant_conditions_tags, event_timestamp,
    _two_day_feature_store_input, _two_day_features_config, _two_day_feature_tag_config,
    _two_day_tags_config, _two_day_run_tag_config
    ) = preprocess_input.run_preprocess()

    offset_half_hr = event_timestamp - timedelta(hours=0.5)
    # make a trigger feature null for 2 hours
    trigger_feature_store_input = trigger_feature_store_input.with_columns(
        pl.when(pl.col('timestamp')<offset_half_hr)
        .then(None).otherwise(pl.col('****'))
        .alias('****'))

    trigger_conditions = TriggersCheck(
            trigger_feature_store_input, trigger_features_config,
            trigger_tags_config, trigger_run_tag_config,
            trigger_feature_tag_config, plant_config, eventtype2_data)
    _trigger, _flag = trigger_conditions.trigger_run()

    assert "Trigger_check: There are features with nulls more than the null threshold: ['****']" in [rec.message for rec in caplog.records]
