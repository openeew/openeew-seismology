# import modules
from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey
from cloudant.database import CloudantDatabase

from dotenv import dotenv_values
import time
import pandas as pd

ibm_cred = dotenv_values()

SERVICE_USERNAME = ibm_cred["SERVICE_USERNAME"]
SERVICE_PASSWORD = ibm_cred["SERVICE_PASSWORD"]
SERVICE_URL = ibm_cred["SERVICE_URL"]


class GetDevices:
    """This class gets the devices from Cloudant"""

    def __init__(self, devices, params) -> None:
        super().__init__()
        self.devices = devices
        self.params = params

    def get_devices(self):
        # Establish a connection with the service instance.
        client = Cloudant(SERVICE_USERNAME, SERVICE_PASSWORD, url=SERVICE_URL)
        client.connect()

        database_name = self.params["db_name"]
        my_database = client[database_name]

        all_devices = Result(my_database.all_docs, include_docs=True)

        # create empty device df to replace the old one
        new_device_table = pd.DataFrame()

        for device in all_devices:

            if device["doc"]["status"] == "Connect":

                try:
                    device_id = device["doc"]["DeviceID"]
                    latitude = device["doc"]["latitude"]
                    longitude = device["doc"]["longitude"]

                    dev = pd.DataFrame(
                        {
                            "device_id": device_id,
                            "latitude": latitude,
                            "longitude": longitude,
                        },
                        index=[0],
                    )

                    new_device_table = new_device_table.append(dev, ignore_index=True)

                except:
                    pass

        self.devices.data = new_device_table

    def run(self):
        # run loop indefinitely
        while True:
            self.get_devices()
            time.sleep(self.params["sleep_time_devices"])
