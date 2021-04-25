# import modules
from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey
from cloudant.database import CloudantDatabase

from dotenv import dotenv_values
import time
import pandas as pd
import json
import os


class GetDevices:
    """This class gets the devices from Cloudant"""

    def __init__(self, devices, params) -> None:
        super().__init__()
        self.devices = devices
        self.params = params

    def get_devices(self):
        # Establish a connection with the service instance.
        client = Cloudant(
            os.environ["CLOUDANT_USERNAME"],
            os.environ["CLOUDANT_PASSWORD"],
            url=os.environ["CLOUDANT_URL"],
        )
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

    def get_devices_local(self):

        device_local_path = self.params["device_local_path"]

        with open(device_local_path, "r") as devices:

            devices = json.load(devices)
            for device in devices:

                try:
                    device_id = device["device_id"]
                    latitude = device["latitude"]
                    longitude = device["longitude"]

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

    def run(self):
        # run loop indefinitely
        while True:

            try:
                # try to get devices from cloud
                self.get_devices()
                print("✅ Received devices from the cloudant database.")
            except:
                # get devices from json file locally
                self.get_devices_local()
                print("✅ Received devices from a local file.")

            time.sleep(self.params["sleep_time_devices"])
