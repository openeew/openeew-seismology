"""Simulate devices by sending device data to an MQTT Server"""
import json
from time import sleep
from argparse import ArgumentParser
from paho.mqtt.client import Client as MqttClient
import os


def run(region, topic, json_data, params):
    """Main method that creates client and executes the rest of the script"""

    if params["MQTT"]=="IBM":
        # create a client
        client = create_client(
            host=os.environ["MQTT_HOST"],
            port=1883,
            username=os.environ["MQTT_USERNAME"],
            password=os.environ["MQTT_PASSWORD"],
        )

    elif params["MQTT"]=="local":
        # create a client
        client = create_client(
            host="localhost",
            port=1883,
            username="NA",
            password="NA",
        )

    topic = "iot-2/type/OpenEEW/id/"+ region +"/evt/" + topic + "/fmt/json"

    publish_json(client, topic, json_data)

    client.disconnect()


def publish_json(client, topic, data):
    """Publish each JSON to a given topic"""

    json_obj = json.dumps(data)

    client.publish(topic, json_obj)


def create_client(host, port, username, password):
    """Creating an MQTT Client Object"""
    client = MqttClient()

    if username and password:
        client.username_pw_set(username=username, password=password)

    client.connect(host=host, port=port)
    return client
