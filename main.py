"""
This is the main file that runs the OpenEEW code package
"""

# import modules
import time
import pickle
from threading import Thread

from params import main_params, tt_params, det_params, ev_params
from src import (
    data_holders,
    receive_traces,
    receive_devices,
    detection,
    event,
    travel_time,
)

__author__ = "Vaclav Kuna"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Vaclav Kuna"
__email__ = "kuna.vaclav@gmail.com"
__status__ = ""


def main():

    # Pre-load/calculate travel-time tables
    travel_times = travel_time.calculate_trave_times(params=tt_params)

    # Pre-load keras detection model
    # detection_model_name = det_params["detection_model_name"]
    # detection_model = detection.load_model(detection_model_name)

    # Create a RawData DataFrame.
    raw_data = data_holders.RawData()

    # Create a Devices DataFrame.
    devices = data_holders.Devices()

    # Create a Detections DataFrame.
    detections = data_holders.Detections()

    # Create a Events DataFrame.
    events = data_holders.Events()

    # We create and start our devices update worker
    stream = receive_devices.DeviceReceiver(devices)
    receive_devices_process = Thread(target=stream.run)
    receive_devices_process.start()

    # We create and start our raw_data update worker
    stream = receive_traces.DataReceiver(raw_data)
    receive_data_process = Thread(target=stream.run)
    receive_data_process.start()

    # We create and start detection worker
    compute = detection.Detect(
        raw_data=raw_data, detections=detections, params=det_params
    )
    detect_process = Thread(target=compute.run)
    detect_process.start()

    # We create and start event worker
    compute = event.Event(
        devices=devices,
        detections=detections,
        events=events,
        travel_times=travel_times,
        params=ev_params,
    )
    event_process = Thread(target=compute.run)
    event_process.start()

    # We join our Threads, i.e. we wait for them to finish before continuing
    receive_devices_process.join()
    receive_data_process.join()
    detect_process.join()
    event_process.join()


if __name__ == "__main__":
    main()
