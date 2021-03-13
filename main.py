"""
This is the main file that runs the OpenEEW code package
"""

# import modules
import time
import pickle
from utils import raw_data, devices
from src import db_handle, detection, magnitude, travel_time, event
from params import db, main_params, tt_params, det_params, mag_params

__author__ = "Vaclav Kuna"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Vaclav Kuna"
__email__ = "kuna.vaclav@gmail.com"
__status__ = ""


# ---------------------------------------------
# INITIATE THE DATABASE AND POPULATE THE TABLES
# ---------------------------------------------

# create empty database
if main_params['db_init']:
    db_handle.db_init(db)

# create empty detection, device, assoc and event tables
db_handle.db_tables_init(db)

# populate the device table
devices.populate_devices(main_params['device_path'], db)

# open/calculate travel times
travel_times = travel_time.calculate_trave_times(db, params=tt_params)

# create and populate raw_data table in the database
if main_params['populate_raw']:
    raw_data.make_raw_table(main_params['data_path'], db)

# load keras detection model
detection_model_name = det_params['detection_model_name']
detection_model = detection.load_model(detection_model_name)

# initiate the first event
ev = None

# ---------------------------------------------
# MAIN CODE
# ---------------------------------------------

# get the start time
time_now = raw_data.time_start(db)+1

# create empty data dictionary
data_buffer = []

# MAIN LOOP

while True:

    # get data from the last second
    data_sec = raw_data.fetch_data(db, time_now)

    # add the data to the data buffer
    data_buffer.extend(data_sec)

    # DETECTION
    detection.detect(
        model=detection_model,
        data_buffer=data_buffer,
        db=db,
        time_now=time_now,
        params=det_params # alternatively choose 'ml'
    )

    # LOCATION
    ev = event.find_and_locate(
        ev=ev,
        db=db,
        time_now=time_now, 
        travel_times=travel_times,
        params=mag_params
    )


    # Remove old data from the buffer
    # (all data chunks with the firs element older that time_now - buffer time)
    buffer_len = main_params['buffer_len']
    data_buffer = [line for line in data_buffer if line['time'][0]>(time_now-buffer_len)]

    # UPDATE TIME AND SLEEP
    time_now += 1
    time.sleep(main_params['sleep_time']) 
