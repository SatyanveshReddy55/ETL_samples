import azure.functions as func
import json
import polars as pl

class EventManipulation():
    def __init__(self, event: func.EventHubEvent) -> None:
        self.event = event

    # get event as bytes
    def get_event_as_bytes(self):
        return self.event.get_body()
    
    # return as dataframe
    def get_event_as_dataframe(self):
        return pl.read_json(self.get_event_as_bytes())
        
    # return as string
    def get_event_as_string(self):
        return self.get_event_as_bytes().decode()
    
    #return json loads
    def get_event_as_json_dict(self):
        return json.loads(self.get_event_as_string())
