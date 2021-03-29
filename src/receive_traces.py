"""This script receives trace data from MQTT by subscribing to a topic"""
import json
from argparse import ArgumentParser
from paho.mqtt.client import Client as MqttClient


class DataReceiver:
    """This class subscribes to the MQTT and receivces raw data"""


    def __init__(self, df_holder) -> None:
        """Initializes the DataReceiver object"""
        super().__init__()
        self.df_holder = df_holder


    def run(self):
        """Main method that parses command options and executes the rest of the script"""
        parser = ArgumentParser()
        parser.add_argument("--username", help="MQTT username")
        parser.add_argument("--password", help="MQTT password")
        parser.add_argument(
            "--clientid", help="MQTT clientID", default="recieve_traces_simulator"
        )
        parser.add_argument(
            "--host", help="MQTT host", nargs="?", const="localhost", default="localhost"
        )
        parser.add_argument(
            "--port", help="MQTT port", nargs="?", type=int, const=1883, default=1883
        )
        arguments = parser.parse_args()

        client = self.create_client(
            arguments.host,
            arguments.port,
            arguments.username,
            arguments.password,
            arguments.clientid,
        )
        client.loop_forever()


    def create_client(self, host, port, username, password, clientid):
        """Creating an MQTT Client Object"""
        client = MqttClient(clientid)

        if username and password:
            client.username_pw_set(username=username, password=password)

        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(host=host, port=port)
        return client


    def on_connect(self, client, userdata, flags, resultcode):
        """Upon connecting to an MQTT server, subscribe to the topic
        The production topic is 'iot-2/type/OpenEEW/id/+/evt/status/fmt/json'"""

        topic = "iot-2/type/OpenEEW/id/+/evt/status/fmt/json"
        print(f"✅ Connected with result code {resultcode}")
        client.subscribe(topic)


    def on_message(self, client, userdata, message):
        """When a message is sent to a subscribed topic,
        decode the message and send it to another method"""
        try:
            decoded_message = str(message.payload.decode("utf-8", "ignore"))
            data = json.loads(decoded_message)
            
            self.df_holder.update(data)
        except BaseException as exception:
            print(exception)
