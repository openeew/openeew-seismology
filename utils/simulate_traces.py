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


def run():
    """Main method that creates client and executes the rest of the script"""

    if params["MQTT"] == "IBM":
        # create a client
        client = create_client(
            host=os.environ["MQTT_HOST"],
            port=1883,
            username=os.environ["MQTT_USERNAME"],
            password=os.environ["MQTT_PASSWORD"],
            clientid=os.environ["MQTT_CLIENTID"] + "m"
        )

    elif params["MQTT"] == "local":
        # create a client
        client = create_client(
            host="localhost",
            port=1883,
            username="NA",
            password="NA",
            clientid=os.environ["MQTT_CLIENTID"] + "m"
        )

    # topic = "iot-2/type/OpenEEW/id/" + params["region"] + "/evt/trace/fmt/json"
    topic = "iot-2/type/OpenEEW/id/000000000000/evt/status/fmt/json"

    publish_jsonl(params["hist_data_path"], client, topic)


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
        print(topic)
        json_str = data[["device_id", "x", "y", "z", "sr"]].iloc[i].to_json()
        client.publish(topic, json.dumps(json_str))
        time.sleep(timediff.iloc[i])

        print(
            datetime.utcfromtimestamp(data["cloud_t"].iloc[i]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )


run()
