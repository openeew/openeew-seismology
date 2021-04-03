"""This script receives device data from MQTT by subscribing to the iot-2/type/OpenEEW/id/+/mon"""
import json
from argparse import ArgumentParser
from paho.mqtt.client import Client as MqttClient


class DeviceReceiver:
    """This class subscribes to the MQTT and receivces raw data"""

    def __init__(self, df_receivers) -> None:
        """Initializes the DataReceiver object"""
        super().__init__()
        self.df_receivers = df_receivers

    def run(self):
        """Main method that parses command options and executes the rest of the script"""
        parser = ArgumentParser()
        parser.add_argument("--username", help="MQTT username")
        parser.add_argument("--password", help="MQTT password")
        parser.add_argument(
            "--clientid", help="MQTT clientID", default="recieve_devices_simulator"
        )
        parser.add_argument(
            "--host",
            help="MQTT host",
            nargs="?",
            const="localhost",
            default="localhost",
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
        """Upon connecting to an MQTT server, subscribe to a topic
        the production topic is 'iot-2/type/OpenEEW/id/+/mon'"""

        topic = "iot-2/type/OpenEEW/id/+/mon"
        print(f"✅ Subscribed to devices with result code {resultcode}")
        client.subscribe(topic)

    def on_message(self, client, userdata, message):
        """When a message is sent to a subscribed topic,
        decode the message and send it to another method"""
        try:
            decoded_message = str(message.payload.decode("utf-8", "ignore"))
            data = json.loads(decoded_message)

            self.df_receivers.update(data)
        except BaseException as exception:
            print(exception)
