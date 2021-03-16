"""

"""

import mysql.connector
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Vaclav Kuna"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Vaclav Kuna"
__email__ = "kuna.vaclav@gmail.com"
__status__ = ""


def db_init(db):

    """The function cretes an empty database via mysql

    INPUT:
    db_name : Database name

    OUTPUT:
    The function has no outputs
    """

    # connect to the database
    mydb = mysql.connector.connect(
        host=db["host"], user=db["user"], passwd=db["passwd"]
    )

    # set the database pointer
    cur = mydb.cursor()

    # delete an old database if exists
    sql = "DROP DATABASE IF EXISTS " + db["db_name"]
    cur.execute(sql)

    # create a new database
    sql = "CREATE DATABASE " + db["db_name"]
    cur.execute(sql)

    # close db and cursor
    mydb.close()
    cur.close()


def db_tables_init(db):

    """The function populates a db with empty tables

    INPUT:
    db_name : Database name

    OUTPUT:
    The function has no outputs
    """

    # connect to the database
    mydb = mysql.connector.connect(
        host=db["host"], user=db["user"], passwd=db["passwd"], database=db["db_name"]
    )

    # set the database pointer
    cur = mydb.cursor()

    # delete tables if they exist
    sql = "DROP TABLE IF EXISTS devices"
    cur.execute(sql)

    sql = "DROP TABLE IF EXISTS detections"
    cur.execute(sql)

    sql = "DROP TABLE IF EXISTS assoc"
    cur.execute(sql)

    sql = "DROP TABLE IF EXISTS event"
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

    # create a detection table
    sql = "CREATE TABLE detections \
        (detection_id INT AUTO_INCREMENT PRIMARY KEY, \
        device_id VARCHAR(255), \
        time DOUBLE(13,3), \
        mag1 DOUBLE(8,3), \
        mag2 DOUBLE(8,3), \
        mag3 DOUBLE(8,3), \
        mag4 DOUBLE(8,3), \
        mag5 DOUBLE(8,3), \
        mag6 DOUBLE(8,3), \
        mag7 DOUBLE(8,3), \
        mag8 DOUBLE(8,3), \
        mag9 DOUBLE(8,3), \
        mag10 DOUBLE(8,3))"
    cur.execute(sql)

    # create an association table
    sql = "CREATE TABLE assoc \
            (device_id VARCHAR(255), \
            det_id VARCHAR(255), \
            event_id VARCHAR(255))"
    cur.execute(sql)

    # create an event table
    sql = "CREATE TABLE event \
        (event_id VARCHAR(255), \
        orig_time DOUBLE(8,3), \
        latitude DOUBLE(8,3), \
        longitude DOUBLE(8,3), \
        depth DOUBLE(8,3), \
        magnitude DOUBLE(8,3))"
    cur.execute(sql)

    # you should close what you've opened
    mydb.close()
    cur.close()
