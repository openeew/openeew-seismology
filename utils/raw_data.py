"""
The raw data module handles the historic data for OpenEEW processing

The module contains 3 functions:

make_raw_table: receives data from .jsonl fiels and passes them into
    raw_data table of a specified database

raw_data_display: displays a sub-section of the raw_data table
"""

import mysql.connector
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

__author__ = "Vaclav Kuna"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Vaclav Kuna"
__email__ = "kuna.vaclav@gmail.com"
__status__ = ""


def make_raw_table(data_path, db):

    """The function loads .jsonl data from a specified folder and passes them
    into a raw_data table of a predefined database

    INPUT:
    data_path : Absolute/relative path to the data
    db_name : Database name

    OUTPUT:
    The function has no outputs
    """

    # do some printing
    print("")
    print("----------")
    print("POPULATING RAW DATA TABLE")
    print("----------")

    # connect to the database
    mydb = mysql.connector.connect(
        host=db["host"], user=db["user"], passwd=db["passwd"], database=db["db_name"]
    )

    # set the database pointer
    cur = mydb.cursor()

    # delete the data table if there is one in the database
    sql = "DROP TABLE IF EXISTS raw_data"
    cur.execute(sql)

    # create empty data table
    sql = "CREATE TABLE raw_data \
        (device_id VARCHAR(255), \
        time DOUBLE(14,4), \
        x DOUBLE(7,5), \
        y DOUBLE(7,3), \
        z DOUBLE(7,3))"
    cur.execute(sql)

    ## POPULATE THE DATA
    sql = "INSERT INTO raw_data (device_id, time, x, y, z) VALUES (%s, %s, %s, %s, %s)"

    # loop over all files in a folder
    for filepath in glob.iglob(data_path + "/*/*.jsonl"):

        print(filepath)

        with open(filepath, "r") as json_file:
            json_list = list(json_file)

        # loop over all elements in the file
        for json_str in json_list:
            result = json.loads(json_str)

            # get the data from json line
            device_id = result["device_id"]
            cloud_t = result["cloud_t"]
            x = result["x"]
            y = result["y"]
            z = result["z"]

            # loop over all irems in the json line
            for item in range(len(x)):
                entry = (device_id, cloud_t, x[item], y[item], z[item])
                cur.execute(sql, entry)

    # commit changes in the table
    mydb.commit()

    # you should close what you've opened
    mydb.close()
    cur.close()


def plot_raw_data(db):
    pass


def time_start(db):

    """The function finds the minimum time in the database

    INPUT:
    db_name : Database name

    OUTPUT:
    minimum time
    """

    # connect to the database
    mydb = mysql.connector.connect(
        host=db["host"], user=db["user"], passwd=db["passwd"], database=db["db_name"]
    )

    # set the database pointer
    cur = mydb.cursor()

    # delete the data table if there is one in the database
    sql = "SELECT MIN(time) FROM raw_data"
    cur.execute(sql)

    # fetch and round down the result
    time_min = cur.fetchall()
    time_min = int(np.floor(np.array(time_min)))

    # you should close what you've opened
    mydb.close()
    cur.close()

    # do some printing:
    time_string = datetime.utcfromtimestamp(time_min).strftime("%Y-%m-%d %H:%M:%S")

    print("")
    print("----------")
    print("Start time: {}".format(time_string))
    print("----------")

    # return start timestamp
    return time_min


def fetch_data(db, timestamp):

    """The function fetches all data from a second given by timestamp
    from a given database

    INPUT:
    db_name : Database name
    timestamp : Bring in this second

    OUTPUT:
    data : All data in raw_data table of a given second
    """

    # signal rate
    sr = 31.25

    # connect to the database
    mydb = mysql.connector.connect(
        host=db["host"], user=db["user"], passwd=db["passwd"], database=db["db_name"]
    )

    # set the database pointer
    cur = mydb.cursor()

    # delete the data table if there is one in the database
    sql = (
        "SELECT device_id, time, x, y, z FROM raw_data WHERE time BETWEEN "
        + str(timestamp - 1)
        + " AND "
        + str(timestamp)
    )
    cur.execute(sql)

    # fetch the result
    data = cur.fetchall()

    # get the values
    data_sta = [n[0] for n in data]
    data_time = [n[1] for n in data]
    data_x = [n[2] for n in data]
    data_y = [n[3] for n in data]
    data_z = [n[4] for n in data]

    # unique stations
    unique_sta = set(data_sta)

    # create an empty list that will contain a list of dictionaries
    data_list = []

    # iterate over stations and paste data in a dictionary
    for sta in unique_sta:

        # find indexes of entries from a particular station
        sta_index = [i for i, e in enumerate(data_sta) if e == sta]

        # find entries for the station
        x = [data_x[i] for i in sta_index]
        y = [data_y[i] for i in sta_index]
        z = [data_z[i] for i in sta_index]

        # find the time (primary dictionary key)
        time_start = data_time[sta_index[0]]
        time = list(time_start - np.arange(0, len(x))[::-1] / sr)

        # add into the disctionary
        data_list.append({"sta": sta, "time": time, "x": x, "y": y, "z": z})

    # you should close what you've opened
    mydb.close()
    cur.close()

    # do some printing:
    time_string = datetime.utcfromtimestamp(time_start).strftime("%Y-%m-%d %H:%M:%S")

    print("Time: {}".format(time_string))

    return data_list
