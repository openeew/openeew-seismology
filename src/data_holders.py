from dataclasses import dataclass
import pandas as pd
import json
import numpy as np


@dataclass
class RawData:
    """This dataclass holds a reference to the RawData DF in memory."""

    data: pd.DataFrame = pd.DataFrame()

    def update(self, data):

        data = json.loads(data)

        # create cloud_time vector and replicate device_id
        number_of_entires = len(data["x"])
        sr = data["sr"]
        data["device_id"] = [data["device_id"]] * number_of_entires
        data["cloud_t"] = list(
            data["cloud_t"] - np.arange(0, number_of_entires)[::-1] / sr
        )

        # create a df
        df_new = pd.DataFrame(data)

        # append to the data
        self.data = self.data.append(df_new, ignore_index=True)


@dataclass
class Detections:
    """This dataclass holds a reference to the current DF in memory.
    This is necessary if you do operations without in-place modification of
    the DataFrame, since you will need replace the whole object.
    """

    data: pd.DataFrame = pd.DataFrame(
        columns=[
            "detection_id",
            "device_id",
            "cloud_t",
            "mag1",
            "mag2",
            "mag3",
            "mag4",
            "mag5",
            "mag6",
            "mag7",
            "mag8",
            "mag9",
            "event_id",
        ]
    )

    def update(self, data):
        self.data = self.data.append(data, ignore_index=True)


@dataclass
class Devices:
    """This dataclass holds a reference to the current DF in memory.
    This is necessary if you do operations without in-place modification of
    the DataFrame, since you will need replace the whole object.
    """

    data: pd.DataFrame = pd.DataFrame()

    def update(self, data):

        data = json.loads(data)

        # create a df
        df_new = pd.DataFrame(data, index=[0])

        # append to the data
        self.data = self.data.append(df_new, ignore_index=True)


@dataclass
class Events:
    """This dataclass holds a reference to the current DF in memory.
    This is necessary if you do operations without in-place modification of
    the DataFrame, since you will need replace the whole object.
    """

    data: pd.DataFrame = pd.DataFrame(
        columns=[
            "event_id",
            "cloud_t",
            "orig_time",
            "lat",
            "lon",
            "dep",
            "mag",
            "num_assoc",
        ]
    )

    def update(self, data):

        # create a df
        df_new = pd.DataFrame(data, index=[0])

        # append to the data
        self.data = self.data.append(df_new, ignore_index=True)
