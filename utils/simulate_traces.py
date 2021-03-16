import json
import glob
from argparse import ArgumentParser
from paho.mqtt.client import Client as MqttClient


def run():
    parser = ArgumentParser()
    parser.add_argument(
        "--host", help="MQTT host", nargs="?", const="localhost", default="localhost"
    )
    parser.add_argument(
        "--port", help="MQTT port", nargs="?", type=int, const=1883, default=1883
    )
    parser.add_argument("--file", nargs="?", default="../data/2020_7_2")

    # If MQTT has username and password authentication on
    parser.add_argument("--username", help="MQTT username")
    parser.add_argument("--password", help="MQTT password")

    arguments = parser.parse_args()

    client = create_client(
        arguments.host, arguments.port, arguments.username, arguments.password
    )
    publish_jsonl(arguments.file, client)


def create_client(host, port, username, password):
    client = MqttClient()

    if username and password:
        client.username_pw_set(username=username, password=password)

    client.connect(host=host, port=port)
    return client


def publish_jsonl(data_path, client):
    # loop over all *.jsonl files in a folder
    for filepath in glob.iglob(data_path + "/*/*.jsonl"):

        print("Processing:" + filepath)

        with open(filepath, "r") as json_file:
            json_array = list(json_file)

        # loop over all json elements in the json array and publish to MQTT
        for json_str in json_array:
            client.publish("/traces-test", json.dumps(json_str))


run()
