"""
File description
"""

# import modules
import numpy as np
import pickle
from obspy.taup.tau import TauPyModel
import mysql.connector
import math

import sys
import time

__author__ = "Vaclav Kuna"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Vaclav Kuna"
__email__ = "kuna.vaclav@gmail.com"
__status__ = ""


def get_travel_time_vector(params):
    """"""

    # set params from params
    lat_min = params["lat_min"]
    lat_max = params["lat_max"]
    lon_min = params["lon_min"]
    lon_max = params["lon_max"]
    step = params["step"]
    vel_model = params["vel_model"]
    eq_depth = params["eq_depth"]

    # define maxiumum distance
    max_dist = ((lat_max - lat_min) ** 2 + (lon_max - lon_min) ** 2) ** (1 / 2)

    # define velocity model
    model = TauPyModel(model=vel_model)

    # has to be <= 1/10 of grid step
    step = step / 10

    # define distances and empty array for results
    dist = np.arange(start=0, stop=max_dist, step=step)
    tt = np.zeros_like(dist)

    # loop over all distances
    for i, distance_in_degree in enumerate(dist):

        # do the calculation
        time_out = model.get_travel_times(
            source_depth_in_km=eq_depth,
            distance_in_degree=distance_in_degree,
            phase_list=["p", "P"],
        )
        tt[i] = time_out[0].time

        # print progress percent
        progress = str(int((i / len(dist)) * 100))
        sys.stdout.write("\r  Calculating new velocity vector: " + progress + "%")

    # make a dictionary out of it
    tt_precalc = {"dist": dist, "travel_time": tt}

    return tt_precalc


def get_lat_lon_grid(params):

    lat_min = params["lat_min"]
    lat_max = params["lat_max"]
    lon_min = params["lon_min"]
    lon_max = params["lon_max"]
    step = params["step"]

    lat = np.arange(start=lat_min, stop=lat_max, step=step)
    lon = np.arange(start=lon_min, stop=lon_max, step=step)

    xv, yv = np.meshgrid(lat, lon, sparse=False, indexing="ij")

    return (xv, yv)


def get_travel_time_grid(tt_precalc, params):

    # get width of the grid
    lat_width = params["lat_max"] - params["lat_min"]
    lon_width = params["lon_max"] - params["lon_min"]

    # get grid
    lat = np.arange(start=-lat_width, stop=lat_width, step=params["step"])
    lon = np.arange(start=-lon_width, stop=lon_width, step=params["step"])

    xv, yv = np.meshgrid(lat, lon, sparse=False, indexing="ij")

    nx = xv.shape[0]
    ny = xv.shape[1]

    tt = np.zeros_like(xv)

    for i in range(nx):
        for j in range(ny):

            point_lat = xv[i, j]
            point_lon = yv[i, j]

            # using mine
            distance_in_degree = globe_distance(point_lat, point_lon, 0, 0)

            # find the closest time from the tt_precalc and place it in the grid
            tt[i, j] = tt_precalc["travel_time"][
                np.argmin(np.abs(tt_precalc["dist"] - distance_in_degree))
            ]

        # print progress percent
        progress = str(int((i / nx) * 100))
        sys.stdout.write("\r     Calculating new velocity grid: " + progress + "%")

    return tt


def globe_distance(lat1, lon1, lat2, lon2):

    # approximate radius of earth in km
    R = 6373.0
    deg2km = 111.3

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

    distance = R * c / deg2km

    return distance


def get_travel_time(params):

    tt_path = params["tt_path"]

    # try to load pre-computed travel time vector from file
    try:

        with open(tt_path + "/travel_times.pkl", "rb") as f:
            travel_times = pickle.load(f)

        if all(
            [
                travel_times["params"]["lat_min"] == params["lat_min"],
                travel_times["params"]["lat_max"] == params["lat_max"],
                travel_times["params"]["lon_min"] == params["lon_min"],
                travel_times["params"]["lon_max"] == params["lon_max"],
                travel_times["params"]["eq_depth"] == params["eq_depth"],
                travel_times["params"]["step"] == params["step"],
                travel_times["params"]["vel_model"] == params["vel_model"],
            ]
        ):

            travel_time_fit = True
            print("  Travel time table successfully loaded.")

        else:
            travel_time_fit = False
            print(
                "  Saved travel time table does not match the parameters given in the parameter file."
            )

    except:

        travel_time_fit = False

    if travel_time_fit == False:

        # calculate travel_time vector
        tt_vector = get_travel_time_vector(params)

        # calculate lat lon grid
        grid_lat, grid_lon = get_lat_lon_grid(params)

        # calculate travel_time grid
        tt_grid = get_travel_time_grid(tt_vector, params)

        # params to save
        params2save = {
            "lat_min": params["lat_min"],
            "lat_max": params["lat_max"],
            "lon_min": params["lon_min"],
            "lon_max": params["lon_max"],
            "step": params["step"],
            "eq_depth": params["eq_depth"],
            "vel_model": params["vel_model"],
        }

        # Create and save travel time dictionary
        travel_times = {
            "tt_vector": tt_vector,
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "tt_grid": tt_grid,
            "params": params2save,
        }

        # save to
        with open(tt_path + "/travel_times.pkl", "wb") as f:
            pickle.dump(travel_times, f, pickle.HIGHEST_PROTOCOL)

    # Return the dictionary
    return travel_times
