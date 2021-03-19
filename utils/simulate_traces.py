"""Simulates traces to an MQTT Server. Takes a .JSONL file and publishes each line to MQTT"""

import json
import glob
from argparse import ArgumentParser
from paho.mqtt.client import Client as MqttClient


def run():
    """Main method that parses command options and executes the rest of the script"""
    parser = ArgumentParser()
    parser.add_argument(
        "--host", help="An MQTT host", nargs="?", const="localhost", default="localhost"
    )
    parser.add_argument(
        "--port", help="An MQTT port", nargs="?", type=int, const=1883, default=1883
    )
    parser.add_argument(
        "--directory",
        help="A directory containing *.JSONL files",
        nargs="?",
        default="../data/2020_7_2",
    )
    parser.add_argument("--clientid", help="MQTT clientID", default="simulator_traces")

    # If MQTT has username and password authentication on
    parser.add_argument("--username", help="A username for the MQTT Server")
    parser.add_argument("--password", help="A password for the MQTT server")

    arguments = parser.parse_args()

    client = create_client(
        arguments.host, arguments.port, arguments.username, arguments.password
    )
    publish_jsonl(arguments.directory, client, "iot-2/type/OpenEEW/id/000000000000/evt/status/fmt/json")


def create_client(host, port, username, password):
    """Creating an MQTT Client Object"""
    client = MqttClient()

    if username and password:
        client.username_pw_set(username=username, password=password)

    client.connect(host=host, port=port)
    return client


def publish_jsonl(data_path, client, topic):
    """Publish each line of a jsonl given a directory"""

    # loop over all *.jsonl files in a folder
    for filepath in glob.iglob(data_path + "/*/*.jsonl"):

        print("Processing:" + filepath)

        with open(filepath, "r") as json_file:
            json_array = list(json_file)

        # loop over all json elements in the json array and publish to MQTT
        for json_str in json_array:
            client.publish(topic, json.dumps(json_str))


run()
