"""

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


def plot_data(db, station):

    """"""

    # signal rate
    sr = 31.25

    # connect to the database
    mydb = mysql.connector.connect(
        host=db["host"], user=db["user"], passwd=db["passwd"], database=db["db_name"]
    )

    # set the database pointer
    cur = mydb.cursor()

    # select records from the database
    sql = (
        "SELECT time, x, y, z FROM raw_data WHERE device_id="
        + station
        + " ORDER BY time"
    )
    cur.execute(sql)

    # fetch the result
    data = cur.fetchall()

    # get unique times
    unique_time = list(set(n[0] for n in data))

    # plot data
    for t in unique_time:
        # get the values
        x = [n[1] for n in data if n[0] == t]
        y = [n[2] for n in data if n[0] == t]
        z = [n[3] for n in data if n[0] == t]

        time = list(np.arange(0, len(x)) / sr + t)

        plt.plot(
            time, np.array(x) / 10, linewidth=0.5, color=[0.5, 0.5, 0.5]
        )  # x component
        plt.plot(
            time, np.array(y) / 10 + 1, linewidth=0.5, color=[0.5, 0.5, 0.5]
        )  # y component
        plt.plot(
            time, np.array(z) / 10 + 2, linewidth=0.5, color=[0.5, 0.5, 0.5]
        )  # z component

    # get the detections
    sql = "SELECT time FROM detections WHERE device_id=" + station
    cur.execute(sql)

    # fetch the result
    data = cur.fetchall()

    # get the value
    time = [n[0] for n in data]

    for n in range(len(time)):

        plt.scatter(time[n], 0, facecolor="none", edgecolor="b")
        plt.scatter(time[n], 1, facecolor="none", edgecolor="b")
        plt.scatter(time[n], 2, facecolor="none", edgecolor="b")

    plt.show()

    # you should close what you've opened
    mydb.close()
    cur.close()


# choose db name
db_name = "openeew"
host = "localhost"
user = "root"
passwd = "a0a975770495"
db = {"db_name": db_name, "host": host, "user": user, "passwd": passwd}

plot_data(db, "014")
