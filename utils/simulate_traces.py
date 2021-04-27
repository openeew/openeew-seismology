"""Simulates traces to an MQTT Server. Takes a .JSONL file and publishes each line to MQTT"""

import json
import glob
from argparse import ArgumentParser
from paho.mqtt.client import Client as MqttClient

import pandas as pd
import time
from datetime import datetime

import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from params import params


def run(datapath):
    """Main method that creates client and executes the rest of the script"""

    if params["MQTT"] == "IBM":
        # create a client
        client = create_client(
            host=os.environ["MQTT_HOST"],
            port=os.environ["MQTT_PORT"],
            username=os.environ["MQTT_USERNAME"],
            password=os.environ["MQTT_PASSWORD"],
            clientid=os.environ["MQTT_CLIENTID"] + "m",
        )

    elif params["MQTT"] == "local":
        # create a client
        client = create_client(
            host="localhost",
            port=1883,
            username="NA",
            password="NA",
            clientid=os.environ["MQTT_CLIENTID"] + "m",
        )

    elif params["MQTT"] == "custom":
        # create a client
        client = create_client(
            host=os.environ["CUS_MQTT_HOST"],
            port=int(os.environ["CUS_MQTT_PORT"]),
            username=os.environ["CUS_MQTT_USERNAME"],
            password=os.environ["CUS_MQTT_PASSWORD"],
            clientid=os.environ["CUS_MQTT_CLIENTID"] + "sim",
        )

    topic = "iot-2/type/OpenEEW/id/000000000000/evt/trace/fmt/json"

    publish_jsonl(datapath, client, topic)


def create_client(host, port, username, password, clientid):
    """Creating an MQTT Client Object"""
    client = MqttClient(clientid)

    if username and password:
        client.username_pw_set(username=username, password=password)

    client.connect(host=host, port=port)
    return client


def publish_jsonl(data_path, client, topic):
    """Publish each line of a jsonl given a directory"""

    # dataframe that will keep all data
    data = pd.DataFrame()

    # loop over all *.jsonl files in a folder
    for filepath in glob.iglob(data_path + "/*/*.jsonl"):

        print("Processing:" + filepath)

        with open(filepath, "r") as json_file:
            json_array = list(json_file)
            data = data.append([json.loads(line) for line in json_array])

    # create a vector of 'deplays' that will make the data chunks come at the right time
    data.sort_values(by=["cloud_t"], inplace=True)
    timediff = data["cloud_t"].diff()
    timediff = timediff.iloc[1:].append(pd.Series([0])) / 1

    # loop over all json elements in the json array and publish to MQTT
    for i in range(len(data)):

        json_str = data[["device_id", "x", "y", "z", "sr"]].iloc[i].to_json()
        client.publish(topic, json.dumps(json_str))

        time.sleep(timediff.iloc[i])

        print(
            datetime.utcfromtimestamp(data["cloud_t"].iloc[i]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )


eqs = [
    "2017_12_15",
    "2017_12_16",
    "2017_12_25",
    "2018_1_8",
    "2018_1_29",
    "2018_2_16",
    "2018_8_12",
    "2018_8_22",
    "2018_9_25",
    "2019_3_9",
    "2020_1_11",
    "2020_1_24",
    "2020_1_29",
    "2020_1_30",
    "2020_3_30",
    "2020_6_23",
    "2020_7_2",
]

for eq in eqs:

    hist_data_path = "../data/" + eq
    run(hist_data_path)

    time.sleep(200)
