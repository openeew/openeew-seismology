"""Simulate devices by sending device data to an MQTT Server"""
import json
from time import sleep
from paho.mqtt.client import Client as MqttClient
import os


def run(region, topic, json_data, params):
    """
    Main method that creates client and executes the rest of the script

    MQTT variable in params (params["MQTT"]) define whether local, or IBM MQTT is used
    """

    if params["MQTT"] == "IBM":
        # create a client
        client = create_client(
            host=os.environ["MQTT_HOST"],
            port=os.environ["MQTT_PORT"],
            username=os.environ["MQTT_USERNAME"],
            password=os.environ["MQTT_PASSWORD"],
        )

    elif params["MQTT"] == "local":
        # create a client
        client = create_client(
            host="localhost",
            port=1883,
            username="NA",
            password="NA",
        )

    elif params["MQTT"] == "custom":
        # create a client
        client = create_client(
            host=os.environ["CUS_MQTT_HOST"],
            port=int(os.environ["CUS_MQTT_PORT"]),
            username=os.environ["CUS_MQTT_USERNAME"],
            password=os.environ["CUS_MQTT_PASSWORD"],
            cafile=os.environ["CUS_MQTT_CERT"],
        )

    topic = "iot-2/type/OpenEEW/id/" + region + "/evt/" + topic + "/fmt/json"

    publish_json(client, topic, json_data)

    client.disconnect()


def publish_json(client, topic, data):
    """Publish each JSON to a given topic"""

    json_obj = json.dumps(data)

    client.publish(topic, json_obj)


def create_client(host, port, username, password, cafile=None):
    """Creating an MQTT Client Object"""
    client = MqttClient()

    if username and password:
        client.username_pw_set(username=username, password=password)

    if cafile:
        client.tls_set(ca_certs=cafile)

    client.connect(host=host, port=port)
    return client
