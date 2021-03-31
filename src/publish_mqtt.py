"""Simulate devices by sending device data to an MQTT Server"""
import json
from time import sleep
from argparse import ArgumentParser
from paho.mqtt.client import Client as MqttClient


def run(topic, json_data):
    """Main method that parses command options and executes the rest of the script"""
    parser = ArgumentParser()
    parser.add_argument(
        "--host", help="An MQTT host", nargs="?", const="localhost", default="localhost"
    )
    parser.add_argument(
        "--port", help="An MQTT port", nargs="?", type=int, const=1883, default=1883
    )
    parser.add_argument(
        "--file",
        help="A file containing list of devices in *.JSON ",
        nargs="?",
        default="../data/devices/device_locations.json",
    )

    parser.add_argument("--clientid", help="MQTT clientID", default="simulate_devices")

    # If MQTT has username and password authentication on
    parser.add_argument("--username", help="A username for the MQTT Server")
    parser.add_argument("--password", help="A password for the MQTT server")

    arguments = parser.parse_args()

    client = create_client(
        arguments.host, arguments.port, arguments.username, arguments.password
    )

    topic = "iot-2/type/OpenEEW/id/000000000000/" + topic

    publish_json(client, topic, json_data)

    client.disconnect()


def publish_json(client, topic, data):
    """Publish each JSON to the given topic"""

    json_obj = json.dumps(data)
    # print(f"Sending device {json_obj}")
    client.publish(topic, json.dumps(json_obj))


def create_client(host, port, username, password):
    """Creating an MQTT Client Object"""
    client = MqttClient()

    if username and password:
        client.username_pw_set(username=username, password=password)

    client.connect(host=host, port=port)
    return client