import json
import polars as pl
from azure.storage.blob import BlobServiceClient
from function_app.commons.logger import logger


class BlobWrite:
    def __init__(self, connection_string, container_name) -> None:
        self.connection_string = connection_string
        self.container_name = container_name

        # BlobServiceClient connection
        self.blob_service_client = BlobServiceClient.from_connection_string(conn_str=self.connection_string)

        self.container_client = self.blob_service_client.get_container_client(self.container_name)

    def blob_client(self, blob_name: str):
        """Blob client for a specific blob in container
        Args:
            blob_name (str): Name of the blob for which client is needed
        Returns:
            blobClient - client connection to the blob
        """
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, blob=blob_name)
        return blob_client

    def list_blobs_in_container(self):
        """List of blobs in a container
        Args: None
        Returns:
            list: list of blobs present in the container
        """
        return self.container_client.list_blobs()

    def create_append_blob(self, blob_name: str):
        """Create a blob of type AppendBlob
        Args:
            blob_name (str): name of append blob to create
        Returns:
            None
        """
        try:
            blob_client = self.blob_client(blob_name=blob_name)
            logger.info("Creating append blob...")
            blob_client.create_append_blob()
        except Exception as exc:
            logger.error(f"Error creating blob - {exc}")
            raise exc

    def append_data(self, data: str, blob_name: str):
        """Append data in the blob, if blob doesn't exist: first create the blob
        and then add the data.
        Args:
            data (str): Data to append
            blob_name (str): Blob to append data into
        Returns:
            None
        """
        try:
            blob_client = self.blob_client(blob_name=blob_name)
            # if blob doesn't exist
            if not blob_client.exists():
                # Create append blob before adding data
                self.create_append_blob(blob_name=blob_name)

            blob_properties = blob_client.get_blob_properties()

            # if blob is not empty, append in new line
            if blob_properties.size != 0:
                data = f"\n{data}"

            blob_client.append_block(data, length=len(data))
            logger.info("Data appended to the blob.")
        except Exception as exc:
            logger.error(f"Error appending to blob - {exc}")
            raise exc

    def read_blob(self, blob_name: str):
        """Reads the blob using client and provides data as Polars Dataframe
        Args:
            blob_name (str): blob to read data from
        Returns:
            dataframe: dataframe with blob data
        """
        try:
            blob_client = self.blob_client(blob_name=blob_name)
            # Set if exists condition
            logger.info("Reading data from blob.")

            blob_properties = blob_client.get_blob_properties()

            # if blob is not empty, read data else none
            if blob_properties.size != 0:
                # return blob data as string
                blob_text = blob_client.download_blob().readall().decode()

                # Split the string by '\n' and parse each JSON object
                json_objects = blob_text.strip().split("\n")

                # Convert JSON objects to dictionaries
                data = [json.loads(obj) for obj in json_objects]

                # Create DataFrame from the list of dictionaries
                df_data = pl.DataFrame(data)
            else:
                df_data = None

            return df_data
        except Exception as exc:
            logger.error(f"Error reading from blob {blob_name} - {exc}")
            raise exc

    def read_all_blobs(self):
        """Reads all the blobs available in the container
        Args: None
        Returns:
            dict: with key as blob name
                value being data in the blob as polars dataframe
        """
        data = {}
        try:
            # list all blobs
            blob_list = self.list_blobs_in_container()
            for blob in blob_list:
                blob_data = self.read_blob(blob_name=blob.name)
                data[blob.name] = blob_data
            return data
        except Exception as exc:
            logger.error(f"Error reading blobs data - {exc}")
            raise exc
