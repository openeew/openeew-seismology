"""

"""

import mysql.connector
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv

__author__ = "Vaclav Kuna"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Vaclav Kuna"
__email__ = "kuna.vaclav@gmail.com"
__status__ = ""


def populate_devices(data_path, db):

    """The function loads .csv data from a specified folder and passes them
    into a devices table of a predefined database

    INPUT:
    data_path : Absolute/relative path to the station csv
    db_name : Database name

    OUTPUT:
    The function has no outputs
    """
    print("")
    print("----------")
    print("POPULATING DEVICE TABLE")
    print("----------")

    # connect to the database
    mydb = mysql.connector.connect(
        host=db["host"], user=db["user"], passwd=db["passwd"], database=db["db_name"]
    )

    # set the database pointer
    cur = mydb.cursor()

    # delete the data table if there is one in the database
    sql = "DROP TABLE IF EXISTS devices"
    cur.execute(sql)

    # create table devices
    sql = "CREATE TABLE devices \
            (device_id VARCHAR(255), \
            latitude DOUBLE(7,3), \
            longitude DOUBLE(7,3), \
            elev DOUBLE(7,3), \
            firmware_version DOUBLE(5,2), \
            device_type VARCHAR(255), \
            time_entered DOUBLE(13,3))"
    cur.execute(sql)

    ## POPULATE THE DATA
    sql = "INSERT INTO devices (\
        device_id, \
        latitude, \
        longitude, \
        elev, \
        firmware_version, \
        device_type, \
        time_entered \
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)"

    # open csv file with station locations
    with open(data_path) as csvfile:
        data = csv.reader(csvfile, delimiter=",")
        for row in data:
            # read station specs
            device_id = row[0]
            latitude = float(row[1])
            longitude = float(row[2])
            elev = 0
            firmware_version = 1.0
            device_type = "OpenEEW 2.0"
            time_entered = 0

            # enter in the database
            entry = (
                device_id,
                latitude,
                longitude,
                elev,
                firmware_version,
                device_type,
                time_entered,
            )
            cur.execute(sql, entry)

    # commit changes in the table
    mydb.commit()

    # you should close what you've opened
    mydb.close()
    cur.close()


def plot_devices(db):
    pass
