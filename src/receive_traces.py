"""This script receives trace data from MQTT by subscribing to a topic"""
import json
from argparse import ArgumentParser
from paho.mqtt.client import Client as MqttClient
import datetime
import os


class DataReceiver:
    """This class subscribes to the MQTT and receivces raw data"""

    def __init__(self, df_holder, params) -> None:
        """
        Initializes the DataReceiver object
        
        MQTT variable in params (params["MQTT"]) define whether local, or IBM MQTT is used
        """
        super().__init__()
        self.df_holder = df_holder
        self.params = params

    def run(self):
        """Main method that creates client and executes the rest of the script"""

        if self.params["MQTT"] == "IBM":
            # create a client
            client = self.create_client(
                host=os.environ["MQTT_HOST"],
                port=os.environ["MQTT_PORT"],
                username=os.environ["MQTT_USERNAME"],
                password=os.environ["MQTT_PASSWORD"],
                clientid=os.environ["MQTT_CLIENTID"] + "trace",
            )

        elif self.params["MQTT"] == "local":
            # create a client
            client = self.create_client(
                host="localhost",
                port=1883,
                username="NA",
                password="NA",
                clientid="NA:" + "trace",
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
        The production topic is 'iot-2/type/OpenEEW/id/+/evt/trace/fmt/json'"""

        # topic = "iot-2/type/OpenEEW/id/" + self.params["region"] + "/evt/trace/fmt/json"
        topic = "iot-2/type/OpenEEW/id/000000000000/evt/status/fmt/json"

        print(f"✅ Subscribed to sensor data with result code {resultcode}")
        client.subscribe(topic)

    def on_message(self, client, userdata, message):
        """When a message is sent to a subscribed topic,
        decode the message and send it to another method"""
        try:
            decoded_message = str(message.payload.decode("utf-8", "ignore"))
            data = json.loads(decoded_message)

            # get timestamp for the received trace
            dt = datetime.datetime.now(datetime.timezone.utc)
            utc_time = dt.replace(tzinfo=datetime.timezone.utc)
            cloud_t = utc_time.timestamp()

            self.df_holder.update(data, cloud_t)
        except BaseException as exception:
            print(exception)
