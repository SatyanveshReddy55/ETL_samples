"""
This module contains classes and methods for connecting and sending data to an eventhub
"""
from azure.eventhub import EventData, EventHubProducerClient
from azure.eventhub.exceptions import ConnectError
from function_app.commons.logger import logger


class EventHubProducer:
    def __init__(self, eventhub_connection_string, feature_store_output_eh):
        self.producer_client = EventHubProducerClient.from_connection_string(
            conn_str=eventhub_connection_string,
            eventhub_name=feature_store_output_eh,  # EventHub name should be specified if it doesn't show up in connection string.
        )

    def process_data_to_eventhub(self, data):
        try:
            event_data_batch = self.producer_client.create_batch()
            event_data_batch.add(EventData(data.encode("utf-8")))
            self.producer_client.send_batch(event_data_batch)
        except ConnectError as eh_error:
            logger.error(f"EventHubError: Failed to connect to Event Hub - {eh_error}")
            raise eh_error
        except ValueError as value_error:
            logger.error(f"ValueError: Invalid data format - {value_error}")
            raise value_error
        except Exception as exception:
            logger.error(f"Error while sending event to eventhub - {exception}")
            raise exception
        finally:
            # Close down the producer handler.
            self.producer_client.close()
