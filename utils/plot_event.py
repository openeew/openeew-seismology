"""

"""

import mysql.connector
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import math

import sys, os

sys.path.append(os.path.dirname(sys.path[0]))
from src import event

__author__ = "Vaclav Kuna"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Vaclav Kuna"
__email__ = "kuna.vaclav@gmail.com"
__status__ = ""


def globe_distance(lat1, lon1, lat2, lon2):

    # approximate radius of earth in km
    R = 6373.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    return distance


def plot_event(ev):

    """"""

    ev_name = ev[0]
    ev_time = ev[1]
    ev_lat = ev[2]
    ev_lon = ev[3]
    ev_mag = ev[4]

    # open the event
    with open("obj/events/" + ev_name + ".pkl", "rb") as f:
        event = pickle.load(f)

    # plot location
    lat = event.location["grid"]["lat"]
    lon = event.location["grid"]["lon"]

    prob = event.location["prob"]

    mine_lat = event.location["location"][-1]["lat"]
    mine_lon = event.location["location"][-1]["lon"]

    lat_min = lat.min()
    lat_max = lat.max()
    lon_min = lon.min()
    lon_max = lon.max()

    # set up the plot
    plt.close("all")
    fig = plt.figure(figsize=(8, 8))
    title_str = (
        "Event: "
        + ev_name.split("_")[0]
        + "/"
        + ev_name.split("_")[1]
        + "/"
        + ev_name.split("_")[2]
        + " M"
        + str(ev_mag)
    )
    fig.suptitle(title_str, fontsize=16)

    ax1 = plt.subplot2grid(
        (3, 3), (0, 0), colspan=3, rowspan=2, projection=ccrs.PlateCarree()
    )
    ax2 = plt.subplot2grid((3, 3), (2, 0))
    ax3 = plt.subplot2grid((3, 3), (2, 1))
    ax4 = plt.subplot2grid((3, 3), (2, 2))

    # ---------------------
    # AX 1 - EPICENTRAL MAP
    # ---------------------

    # plot map and probability
    ax1.imshow(
        prob, extent=[lon_min, lon_max, lat_min, lat_max], cmap="turbo", alpha=0.7
    )
    ax1.coastlines()
    ax1.plot(
        mine_lon,
        mine_lat,
        marker=(5, 1),
        markersize=12,
        markerfacecolor=[0.8500, 0.3250, 0.0980],
        markeredgecolor=[0.1, 0.1, 0.1],
    )
    # ax1.text(lon[prob==prob.max()]+1, lat[prob==prob.max()]+1, str(lon[prob==prob.max()]) + ' ' + str(lat[prob==prob.max()]))
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # stations and true epicenter
    detected_sta = [n["device_id"] for n in event.detections.values()]
    f = open("data/devices/devices_locations.csv", "r")
    for line in f:
        sta = line.split(",")[0]
        lat = float(line.split(",")[1])
        lon = float(line.split(",")[2])

        if sta in detected_sta:
            ax1.plot(
                lon,
                lat,
                "^",
                markersize=6,
                markerfacecolor=[0.8500, 0.3250, 0.0980],
                markeredgecolor=[0.1, 0.1, 0.1],
            )
        else:
            ax1.plot(
                lon,
                lat,
                "^",
                markersize=6,
                markerfacecolor=[1, 1, 1],
                markeredgecolor=[0.1, 0.1, 0.1],
            )

    ax1.plot(
        ev_lon,
        ev_lat,
        marker=(5, 1),
        markersize=12,
        markerfacecolor=[1, 1, 1],
        markeredgecolor=[0.1, 0.1, 0.1],
    )

    # ---------------------
    # AX 2 - DETECTIONS
    # ---------------------
    det = event.detections

    time = np.array([n["time"] for n in det.values()]) - ev_time
    y = np.arange(1, time.size + 1, 1)

    ax2.plot(time, y, "-", color=[0.8, 0.8, 0.8])
    ax2.plot(
        time,
        y,
        "o",
        markerfacecolor=[0, 0.4470, 0.7410],
        markeredgecolor=[0.2, 0.2, 0.2],
        markersize=5,
    )
    ax2.set_ylim((0, 10))
    ax2.set_xlim((0, 60))
    ax2.set_xlabel("Time since eq origin [s]")
    ax2.set_ylabel("Number of detections")

    # ---------------------
    # AX 3 - MAGNITUDE
    # ---------------------
    m = event.magnitude["magnitude"]

    mag = np.array([n["mag"] for n in m])

    time = np.array([n["time"] for n in m])[mag > 0] - ev_time
    mag2 = np.array([n["mag_conf2"] for n in m])[mag > 0]
    mag16 = np.array([n["mag_conf16"] for n in m])[mag > 0]
    mag86 = np.array([n["mag_conf84"] for n in m])[mag > 0]
    mag98 = np.array([n["mag_conf98"] for n in m])[mag > 0]
    mag = mag[mag > 0]

    # Visualize the result
    ax3.plot([0, 60], [ev_mag, ev_mag], "-", color="gray")
    ax3.plot(time, mag, "o", markersize=2, markerfacecolor=[0, 0.4470, 0.7410])
    ax3.fill_between(time, mag2, mag98, color="gray", alpha=0.2, edgecolor=None)
    ax3.fill_between(time, mag16, mag86, color="gray", alpha=0.2, edgecolor=None)
    ax3.set_ylim((3, 9))
    ax3.set_xlim((0, 60))
    ax3.set_xlabel("Time since eq origin [s]")
    ax3.set_ylabel("Magnitude")

    # ---------------------
    # AX 4 - MISLOCATION
    # ---------------------
    loc = event.location["location"]

    time = np.array([n["time"] for n in loc]) - ev_time
    # lat = [n['lat'] for n in loc if len(n['lat'])==1]
    # lon = [n['lon'] for n in loc if len(n['lon'])==1]
    lat = [n["lat"] for n in loc]
    lon = [n["lon"] for n in loc]
    coord = list(zip(lat, lon))

    dist = np.array([globe_distance(ev_lat, ev_lon, n[0], n[1]) for n in coord])
    time = np.array(time[-len(dist) :])

    # Visualize the result
    ax4.plot(time, dist, "-", color=[0.8, 0.8, 0.8])
    ax4.plot(time, dist, "o", markersize=2, markerfacecolor=[0, 0.4470, 0.7410])
    ax4.set_ylim((0, 200))
    ax4.set_xlim((0, 60))
    ax4.set_xlabel("Time since eq origin [s]")
    ax4.set_ylabel("Epicentral error [km]")

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None
    )

    fig.savefig(ev_name + ".pdf", bbox_inches="tight")

    # print mislocation, time and magnitude error
    time = event.origin_time[-1]["origin_time"] - ev_time
    # print((dist[-1], mag[-1]-ev_mag, time))

    plt.show()


events = [
    ("2017_12_15_23_13", 1513379623, 17.382, -101.35, 4.6),
    ("2017_12_16_4_7", 1513397250, 17.592, -101.41, 4.1),
    ("2017_12_25_20_23", 1514233391, 16.986, -99.845, 5.0),
    ("2018_1_8_17_1", 1515430863, 16.578, -99.26, 4.7),
    ("2018_1_29_17_41", 1517247716, 17.414, -101.63, 4.6),
    ("2018_2_16_23_39", 1518824379, 16.218, -98.013, 7.2),
    ("2018_8_12_14_42", 1534084929, 17.112, -100.84, 5.2),
    ("2018_8_22_18_3", 1534960988, 16.534, -98.745, 5.3),
    ("2018_9_25_2_22", 1537842139, 16.47, -99.078, 5.2),
    ("2019_3_9_14_0", 1552140049, 17.26, -100.67, 5.1),
    ("2020_1_11_14_22", 1578752522, 16.25, -98.318, 5.1),
    ("2020_1_24_10_47", 1579862869, 16.002, -97.178, 5.2),
    ("2020_1_29_23_17", 1580339868, 16.787, -100.14, 5.1),
    ("2020_1_30_6_47", 1580366842, 16.831, -100.1, 5.3),
    ("2020_3_30_5_8", 1585544901, 16.46, -98.881, 5.1),
    ("2020_6_23_15_29", 1592926143, 15.784, -96.12, 7.4),
    ("2020_7_2_16_17", 1593706676, 16.21, -98.02, 5.2),
]


# for ev in events:
plot_event(events[16])
