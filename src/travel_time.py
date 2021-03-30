"""
File description
"""

# import modules
import numpy as np
import pickle
from obspy.taup.tau import TauPyModel
import mysql.connector
import math

__author__ = "Vaclav Kuna"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Vaclav Kuna"
__email__ = "kuna.vaclav@gmail.com"
__status__ = ""


def make_grid(lat_min, lat_max, lon_min, lon_max, step):

    lat = np.arange(start=lat_min, stop=lat_max, step=step)
    lon = np.arange(start=lon_min, stop=lon_max, step=step)

    xv, yv = np.meshgrid(lat, lon, sparse=False, indexing="ij")

    return (xv, yv)


def calculate_tt(tt_precalc, grid, sta_lat, sta_lon, eq_depth):

    xv = grid[0]
    yv = grid[1]

    nx = grid[0].shape[0]
    ny = grid[0].shape[1]

    tt = np.zeros_like(xv)

    for i in range(nx):
        for j in range(ny):

            point_lat = xv[i, j]
            point_lon = yv[i, j]

            # using mine
            distance_in_degree = globe_distance(point_lat, point_lon, sta_lat, sta_lon)

            # find the closest time from the tt_precalc and place it in the grid
            tt[i, j] = tt_precalc["travel_time"][
                np.argmin(np.abs(tt_precalc["dist"] - distance_in_degree))
            ]

            # time_out = model.get_travel_times(source_depth_in_km=eq_depth, distance_in_degree=distance_in_degree, phase_list=["p","P"])

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


def precalculate_times(max_dist, step, eq_depth, model):

    # to ensure small tt errors the step for precalculation of travel times
    # has to be <= 1/10 of grid step
    step = step / 10

    dist = np.arange(start=0, stop=max_dist, step=step)
    tt = np.zeros_like(dist)

    for i, distance_in_degree in enumerate(dist):

        time_out = model.get_travel_times(
            source_depth_in_km=eq_depth,
            distance_in_degree=distance_in_degree,
            phase_list=["p", "P"],
        )
        tt[i] = time_out[0].time

    print(tt)

    tt_precalc = {"dist": dist, "travel_time": tt}

    return tt_precalc


def run(params):

    # set params from params
    lat_min = params["lat_min"]
    lat_max = params["lat_max"]
    lon_min = params["lon_min"]
    lon_max = params["lon_max"]
    step = params["step"]
    calculate_open = params["calculate_open"]
    vel_model = params["vel_model"]
    eq_depth = params["eq_depth"]

    max_dist = ((lat_max - lat_min) ** 2 + (lon_max - lon_min) ** 2) ** (1 / 2)

    model_name = "travel_time_d" + str(eq_depth)

    if calculate_open == "calculate":

        # This needs to receive the device dataframe,
        # but probs does not need to run all the time...
        # So it needs some more work

        print("")
        print("----------")
        print("CALCULATING TRAVEL TIMES")
        print("----------")

        print("Precalculationg tt")
        # define velocity model
        model = TauPyModel(model=vel_model)
        # precalculate times
        tt_precalc = precalculate_times(max_dist, step, eq_depth, model)

        # Here get the devices
        # list of [(device_id, latitude, longitude)]
        # devices = ....

        grid = make_grid(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            step=step,
        )

        travel_time = {"grid_lat": grid[0], "grid_lon": grid[1], "vector": tt_precalc}

        print("Calculating for stations")
        for device in devices:

            sta_name = device[0]
            print("Station: {}".format(sta_name))

            sta_lat = device[1]
            sta_lon = device[2]

            travel_time[sta_name] = calculate_tt(
                tt_precalc, grid, sta_lat, sta_lon, eq_depth
            )

        with open("obj/travel_time/" + model_name + ".pkl", "wb") as f:
            pickle.dump(travel_time, f, pickle.HIGHEST_PROTOCOL)

        return travel_time

    elif calculate_open == "open":

        with open("obj/travel_time/" + model_name + ".pkl", "rb") as f:
            travel_time = pickle.load(f)

        return travel_time


run(params)
