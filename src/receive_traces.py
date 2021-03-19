"""This script receives trace data from MQTT by subscribing to the /traces topic"""
import json
from argparse import ArgumentParser
from paho.mqtt.client import Client as MqttClient


def run():
    """Main method that parses command options and executes the rest of the script"""
    parser = ArgumentParser()
    parser.add_argument("--username", help="MQTT username")
    parser.add_argument("--password", help="MQTT password")
    parser.add_argument("--clientid", help="MQTT clientID", default="simulator")
    parser.add_argument(
        "--host", help="MQTT host", nargs="?", const="localhost", default="localhost"
    )
    parser.add_argument(
        "--port", help="MQTT port", nargs="?", type=int, const=1883, default=1883
    )
    arguments = parser.parse_args()

    client = create_client(
        arguments.host, arguments.port, arguments.username, arguments.password, arguments.clientid
    )
    client.loop_forever()


def create_client(host, port, username, password, clientid):
    """Creating an MQTT Client Object"""
    client = MqttClient(clientid)

    if username and password:
        client.username_pw_set(username=username, password=password)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host=host, port=port)
    return client


def on_connect(client, userdata, flags, resultcode):
    """Upon connecting to an MQTT server, subscribe to the /traces topic"""
    """The production topic is 'iot-2/type/OpenEEW/id/+/evt/status/fmt/json'"""
    topic = "/traces"
    topic = "iot-2/type/OpenEEW/id/+/evt/status/fmt/json"
    print(f"✅ Connected with result code {resultcode}")
    client.subscribe(topic)


def on_message(client, userdata, message):
    """When a message is sent to a subscribed topic,
    decode the message and send it to another method"""
    try:
        decoded_message = str(message.payload.decode("utf-8", "ignore"))
        data = json.loads(decoded_message)
        print(f"Received data: {data}")
        # Pass information here
    except BaseException as exception:
        print(exception)


run()
