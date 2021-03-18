"""
Event module
"""

# Import modules
import mysql.connector
import numpy as np
import scipy
import matplotlib.pyplot as plt
from inpoly import inpoly2
import pickle
from datetime import datetime
import math
import time

__author__ = "Vaclav Kuna"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Vaclav Kuna"
__email__ = "kuna.vaclav@gmail.com"
__status__ = ""


def find_and_locate(ev, db, time_now, travel_times, params):

    # 1. Get  new detections
    new_detections = get_detections(db=db, time_now=time_now, window=2)

    # 2. Establish new event or return None
    # if there are no new detections and no event is initiated
    if (len(new_detections) == 0) and (ev == None):
        return None

    # if there are new detections and no event is initiated
    if (len(new_detections) > 0) and (ev == None):
        # set new event
        ev = Event(travel_times, time_now, params)

    # 3. Calculate new location
    ev.get_location(
        new_detections=new_detections,
        travel_times=travel_times,
        db=db,
        time_now=time_now,
        params=params,
    )

    # 6. Calculate magnitude using associated phases
    ev.get_magnitude(db=db, params=params, time_now=time_now)

    # 7. Save or return the event
    # time since the last detection
    tsl = ev.time_since_last(time_now)

    # number of associated detections
    number_of_detections = ev.num_of_dets()

    tsl_max = params["tsl_max"]
    ndef_min = params["ndef_min"]

    if tsl > tsl_max:
        if number_of_detections >= ndef_min:
            # save event to events
            ev.save_event()

        return None

    # print('Detection {}:'.format(tdet-ts))
    # print('Probs {}:'.format(tprob-tdeta))
    # print('Assoc {}:'.format(tassoc-tprob))
    # print('Loc2 {}:'.format(tloc-tassoc))
    # print('Mag {}:'.format(tmag-tloc))

    return ev


def get_detections(db, time_now, window):

    """"""

    # connect to the database
    mydb = mysql.connector.connect(
        host=db["host"], user=db["user"], passwd=db["passwd"], database=db["db_name"]
    )

    # set the database pointer
    cur = mydb.cursor()

    # get the detections less than 10 s old
    sql = (
        "SELECT \
        detection_id, device_id, time \
        FROM detections WHERE time>"
        + str(time_now - window)
        + " && time<"
        + str(time_now)
    )
    cur.execute(sql)

    # fetch the result
    detections = cur.fetchall()

    # new detections
    new_detections = {}

    for det in detections:
        new_entry = {"device_id": det[1], "time": det[2]}
        new_detections[det[0]] = new_entry

    return new_detections


class Event:
    """
    This class handles computation of earthquake magnitude and location.

    The magnitude calculation is probabilistic and based on the Beyes Theorem.
    It starts wits an initial probability distribution of earhquake magnitudes,
    the magnitude range and b Guttenberg-Richter b-value can be specified.
    """

    def __init__(self, travel_times, time_now, params):
        """
        The __init__ function sets the magnitude space and the initial probability
        distribution for magnitude
        """

        # DETECTIONS
        # detections is a dictionary with the detection
        # ID as a primary key. Every detection has 3 secondary
        # keys: device_id, time and assoc
        self.detections = {}

        # LOCATION
        # location is a dictionary with the following keys
        # grid: with secondary keys grid_lat, grid_lon: lat, lon mesh grids
        # loc_prob: probability in the shape of mesh grids
        # location: list of tuples in each time steps
        #    (time, lat, lon, depth)
        prior = self.prior_loc(travel_times=travel_times)

        grid = {"lat": travel_times["grid_lat"], "lon": travel_times["grid_lon"]}
        # loc_prob = {'prior': prior}

        self.location = {"grid": grid, "prob": prior, "location": []}

        # MAGNITUDE
        # magnitude is a dictionary with the following keys
        # mag_grid: bins of magnitude
        # mag_prob: probability of magnitude
        # magnitude: list of tuples in each time steps
        #    (time, mag, mag_lower, mag_upper)
        mag_bins, mag_prior = self.prior_mag(params=params)

        self.magnitude = {"grid": mag_bins, "prob": mag_prior, "magnitude": []}

        # ORIGIN TIME
        # list of origin times in each time step
        # list of tuples
        #    (time, origin_time)
        self.origin_time = []

    # ---------------------
    # LOCATION FUNCTIONS
    # ---------------------

    def get_location(self, new_detections, travel_times, db, time_now, params):

        # Loop over all new detections
        for detection in new_detections.items():

            # Get the number of already associated detections
            num_of_det_start = len(list(self.detections.values()))

            # FOR THE FIRST DETECTION
            if num_of_det_start == 0:

                # Get probabilities for not-yet-arrived stations
                self.get_probs_not_yet_arrived(
                    new_detection=detection,
                    travel_times=travel_times,
                    db=db,
                    time_now=time_now,
                    params=params,
                )

            # FOR ALL FOLLOWING DETECTIONS
            elif num_of_det_start > 0:

                # Get probabilities and associate the new detection
                self.get_probs_and_associate(
                    new_detection=detection,
                    travel_times=travel_times,
                    db=db,
                    time_now=time_now,
                    params=params,
                )

                # Get the number of already associated detections
                num_of_det_stop = len(list(self.detections.values()))

                # if  detections were added to the event
                if num_of_det_stop > num_of_det_start:

                    # Get best location
                    self.get_best_location(
                        time_now=time_now,
                        travel_times=travel_times,
                        params=params,
                        save=True,
                    )

    def get_probs_and_associate(
        self, new_detection, travel_times, db, time_now, params
    ):
        """
        Calculate probabilities and associate new event
        """
        # get detection variables
        new_detection_id = new_detection[0]
        new_detection = new_detection[1]

        detected_devices = list(set([n["device_id"] for n in self.detections.values()]))

        # get the new device id and detection time
        new_device = new_detection["device_id"]
        new_time = new_detection["time"]

        # set a new list of new probabilities
        new_prob = np.zeros_like(self.location["prob"])

        if new_device not in detected_devices:

            # loop over all associated detections
            for detection in self.detections.values():

                # get device ID and detection time
                det_device = detection["device_id"]
                det_time = detection["time"]

                print(
                    "New device: {}, detection device {}".format(new_device, det_device)
                )

                # get sigma
                sigma = self.get_sigma(new_device, det_device, db, params)

                # calculate probability curve
                tt_prob = np.exp(
                    -(
                        (
                            travel_times[det_device]
                            - travel_times[new_device]
                            - det_time
                            + new_time
                        )
                        ** 2
                    )
                    / (2 * sigma ** 2)
                )

                # and add the probability the rest
                new_prob = new_prob + tt_prob

            # ASSOCIATE THE NEW DETECTION

            # get updated potential location of the eq epicenter
            best_lat, best_lon, _ = self.get_best_location(
                time_now, travel_times, params=params, save=False, add_prob=new_prob
            )

            # test the RMS of mispics
            tt_precalc = travel_times["vector"]
            misfit = []

            # get the new location

            for detection in self.detections.values():

                det_device_old = detection["device_id"]
                det_time_old = detection["time"]

                epic_dist_old = self.get_sta_delta(
                    db, det_device_old, eq_lat=best_lat, eq_lon=best_lon
                )
                epic_dist_new = self.get_sta_delta(
                    db, new_device, eq_lat=best_lat, eq_lon=best_lon
                )

                # find the closest time from the tt_precalc and place it in the grid
                tt_old = tt_precalc["travel_time"][
                    np.argmin(np.abs(tt_precalc["dist"] - epic_dist_old / 111.3))
                ]
                tt_new = tt_precalc["travel_time"][
                    np.argmin(np.abs(tt_precalc["dist"] - epic_dist_new / 111.3))
                ]

                misfit.append(((tt_old - tt_new) - (det_time_old - new_time)) ** 2)

            misfit_mean = np.sqrt(np.sum(np.array(misfit)) / len(misfit))

            print(misfit_mean)
            assoc_win = params["assoc_win"]

            if misfit_mean < assoc_win:

                # if associated, append the probabbilities
                self.location["prob"] = self.location["prob"] + new_prob

                # add new detection to detections
                self.add_detection(
                    new_detection=new_detection, new_detection_id=new_detection_id
                )

    def get_best_location(self, time_now, travel_times, params, save=False, add_prob=0):

        # GET BEST LOCATION and ORIGIN TIME
        lat = self.location["grid"]["lat"]
        lon = self.location["grid"]["lon"]

        # initial probability is equal to the prior
        loc_prob = self.location["prob"]

        # add aditional probability (for calling the function by the associator)
        loc_prob = loc_prob + add_prob

        # get best location
        best_lat = lat[loc_prob == loc_prob.max()][0]
        best_lon = lon[loc_prob == loc_prob.max()][0]
        best_depth = params["eq_depth"]  # depth is fixed for all

        # get origin time based on the location and the first detection
        first_det = list(self.detections.values())[0]
        first_sta = first_det["device_id"]
        first_time = first_det["time"]
        sta_travel_time = travel_times[first_sta][loc_prob == loc_prob.max()]
        best_orig_time = first_time - sta_travel_time[0]

        # if save option is selected
        if save:

            # append the location
            to_append = {
                "time": time_now,
                "lat": best_lat,
                "lon": best_lon,
                "depth": best_depth,
            }
            self.location["location"].append(to_append)

            # append the origin time
            to_append = {"time": time_now, "origin_time": best_orig_time}
            self.origin_time.append(to_append)

        # if not, output the lat, lon and origin time
        else:

            return best_lat, best_lon, best_orig_time

    def get_sigma(self, new_device, det_device, db, params):
        """
        Get sigma from distances between the detections  and easrthquakes
        """

        # if constant  sigma is chosen
        if params["sigma_type"] == "const":

            sigma = params["sigma_const"]

        # if sigma is computed from the sigmoid function
        elif params["sigma_type"] == "linear":

            try:
                dist1 = self.get_sta_delta(db, new_device)
                dist2 = self.get_sta_delta(db, det_device)

                dist_ave = (dist1 + dist2) / 2

                sigma = dist_ave * 0.05 + 1
                if sigma > 8:
                    sigma = 8

            except:
                sigma = params["sigma_const"]

        return sigma

    def get_probs_not_yet_arrived(
        self, new_detection, travel_times, db, time_now, params
    ):
        """
        Updates location
        """
        # get detection variables
        new_detection_id = new_detection[0]
        new_detection = new_detection[1]

        # get list of active stations
        self.active_devices = self.get_active_devices(db=db, time_now=time_now)

        # get the station with the first detection
        first_device = new_detection["device_id"]

        # get all the not-yet arrived devices
        nya_devices = list(set(self.active_devices) ^ set([first_device]))

        # get location of all the devices
        device_loc = self.get_all_sta_loc(db)

        # get location of all the not-yet arrived devices
        loc_nya = [(device_loc[n]["lon"], device_loc[n]["lat"]) for n in nya_devices]

        # get location of all the detected device
        loc_det = [(device_loc[first_device]["lon"], device_loc[first_device]["lat"])]

        # append the loc_det at the beginning
        loc_all = loc_det + loc_nya

        # compute the Voronoi cells
        vor = scipy.spatial.Voronoi(loc_all)
        regions, vertices = self.voronoi_finite_polygons_2d(vor)

        # get the lat and lon grid
        lat_grid = self.location["grid"]["lat"]
        lon_grid = self.location["grid"]["lon"]

        # get the polygon aroud the device with detection
        polygon = vertices[regions[0]]

        # get the points in the polygon
        points = np.concatenate(
            (
                np.reshape(lon_grid, (lon_grid.size, 1)),
                np.reshape(lat_grid, (lat_grid.size, 1)),
            ),
            axis=1,
        )
        inside, onedge = inpoly2(points, polygon)

        # change the points in the polygons to 1 and out of the polygon to 0
        inside = inside.reshape(lon_grid.shape)
        inside[inside == True] = 1
        inside[inside == False] = 0

        # get the best prob
        best_prob = self.location["prob"] + inside

        # and replace the prob with the  best prob
        self.location["prob"] = best_prob

        # add detections somewhere
        self.add_detection(
            new_detection=new_detection, new_detection_id=new_detection_id
        )

        # set the best location to the station location
        best_lat = loc_det[0][1]
        best_lon = loc_det[0][0]
        best_depth = params["eq_depth"]  # depth is fixed for all
        best_orig_time = (
            new_detection["time"] - travel_times["vector"]["travel_time"][0]
        )

        # append the location
        to_append = {
            "time": time_now,
            "lat": best_lat,
            "lon": best_lon,
            "depth": best_depth,
        }
        self.location["location"].append(to_append)

        # append the origin time
        to_append = {"time": time_now, "origin_time": best_orig_time}
        self.origin_time.append(to_append)

    def prior_loc(self, travel_times):
        """
        This function sets the prior probability distribution for earthquake location
        """

        loc_prob = np.zeros_like(travel_times["grid_lat"])

        return loc_prob

    # -------------------
    # MAGNITUDE FUNCTIONS
    # -------------------

    def get_magnitude(self, db, params, time_now):
        """
        This function uses the station magnitude estimation and calculates
        the probability distribution for the magnitude.
        It also updates the most likely magnitude and the 68 and 96 percent
        probability intervals
        """

        mag_prob = self.magnitude["prob"]

        for det_id, det in self.detections.items():

            det_sta = det["device_id"]
            pd_all = self.get_pd_detection_id(db=db, det_id=det_id)[0]
            pd = [n for n in pd_all if n is not None]

            try:
                pd_type = "mag" + str(len(pd))
                pd = pd[-1]

                a = params[pd_type][0]
                b = params[pd_type][1]
                c = params[pd_type][2]
                std = params[pd_type][3]

                # Normalize the displacement for the epicentral distance of 1 km
                dist = self.get_sta_delta(db=db, sta=det_sta)
                pd = np.log10(pd) + c * np.log10(dist + 1)

                # Calculate station magnitude from pd given the linear function with a, b, c
                sta_mag_mu = a * pd + b

                # generate the probability distribution for the station magnitude
                p_m_pd = scipy.stats.norm(sta_mag_mu, std).pdf(self.magnitude["grid"])

                # multiply the prior and the current measurement (the Bayes happens in here)
                mag_prob = np.multiply(mag_prob, p_m_pd)

            except:
                pass

        # normalize the mag_prob
        mag_prob = mag_prob / max(np.cumsum(mag_prob))

        # append to the list of probability distributions
        self.magnitude["prob"] = mag_prob

        # get magnitude and confidence
        magnitude = self.magnitude["grid"][np.argmax(mag_prob)]

        cum_prob = np.cumsum(mag_prob)
        conf2 = self.magnitude["grid"][np.argmin(abs(cum_prob - 0.02))]
        conf16 = self.magnitude["grid"][np.argmin(abs(cum_prob - 0.16))]
        conf84 = self.magnitude["grid"][np.argmin(abs(cum_prob - 0.84))]
        conf98 = self.magnitude["grid"][np.argmin(abs(cum_prob - 0.98))]

        # form the tuple and append to the list of magnitudes
        mag_tuple = {
            "time": time_now,
            "mag": magnitude,
            "mag_conf2": conf2,
            "mag_conf16": conf16,
            "mag_conf84": conf84,
            "mag_conf98": conf98,
        }
        self.magnitude["magnitude"].append(mag_tuple)

    def prior_mag(self, params):
        """
        This function sets the prior probability distribution for magnitude
        It uses the concept of magnitude of completeness and exponential
        decay of probability with increasing magnitude

        The prior probability distribution is a lineary increasing function
        (in a log10 space) from the Mc-2 to Mc (Mc is the magnitude of completenes).
        It peaks at the Mc and decreases to 0 at magnitude 10 with the slope of
        b_value (set to 1 by default)
        """

        prior_type = params["prior_type"]
        mc = params["mc"]
        b_value = params["b_value"]

        # set limits on magnitude and the discretization step
        mag_step = 0.01
        mag_min = 0
        mag_max = 10
        mag_bins = np.arange(mag_min, mag_max, mag_step)

        if prior_type == "gutenberg":

            # create an array with zero probability everywhere
            mag_prob = np.zeros(len(mag_bins))
            mag_step = mag_bins[1] - mag_bins[0]

            # the index of Mc
            peak_index = int(mc * 1 / mag_step)

            # the linear decrease with the b_value
            max_value = (10 - mc) * b_value
            num_of_steps = (10 - mc) * (1 / mag_step)
            mag_prob[peak_index:] = max_value - np.arange(
                0, max_value, max_value / num_of_steps
            )

            # the linear increase to the Mc
            num_of_steps = int(2 * (1 / mag_step))
            mag_prob[peak_index - num_of_steps : peak_index] = np.arange(
                0, max_value, max_value / num_of_steps
            )

            mag_prob = np.ones(len(mag_bins))

            # transform from linear to exponential
            mag_prob = 10 ** mag_prob

        elif prior_type == "constant":

            mag_prob = np.ones(len(mag_bins))

        # normalize probability density function
        mag_prob = mag_prob / max(np.cumsum(mag_prob))

        # return the probability function
        return mag_bins, mag_prob

    def get_pd_detection_id(self, db, det_id):

        # connect to the database
        mydb = mysql.connector.connect(
            host=db["host"],
            user=db["user"],
            passwd=db["passwd"],
            database=db["db_name"],
        )

        # set the database pointer
        cur = mydb.cursor()

        # get the detections less than 10 s old
        sql = (
            "SELECT \
            mag1, mag2, mag3, mag4, mag5, mag6, mag7, mag8, mag9 \
            FROM detections WHERE detection_id="
            + str(det_id)
        )
        cur.execute(sql)

        # fetch the result
        pd = cur.fetchall()

        return pd

    # -----------------
    # UTILITY FUNCTIONS
    # -----------------

    def save_event(self):

        time = self.origin_time[-1]["origin_time"]

        yr = str(datetime.utcfromtimestamp(time).year)
        mo = str(datetime.utcfromtimestamp(time).month)
        day = str(datetime.utcfromtimestamp(time).day)
        hr = str(datetime.utcfromtimestamp(time).hour)
        mi = str(datetime.utcfromtimestamp(time).minute)
        name = yr + "_" + mo + "_" + day + "_" + hr + "_" + mi

        print("Saving event. name:{}".format(name))

        with open("obj/events/" + name + ".pkl", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def add_detection(self, new_detection, new_detection_id):

        # only the first detection from each station will be added
        unique_device = list(
            set([dev["device_id"] for dev in self.detections.values()])
        )

        new_device = new_detection["device_id"]
        new_time = new_detection["time"]

        # if the station did not detect yet
        if new_device not in unique_device:

            # form a detection entry and add it to detections
            new_entry = {"device_id": new_device, "time": new_time, "assoc": True}

            # append to detections
            self.detections[new_detection_id] = new_entry

    def get_active_devices(self, db, time_now):
        """"""

        # connect to the database
        mydb = mysql.connector.connect(
            host=db["host"],
            user=db["user"],
            passwd=db["passwd"],
            database=db["db_name"],
        )

        # set the database pointer
        cur = mydb.cursor()

        # get the detections less than 10 s old
        sql = (
            "SELECT device_id FROM raw_data WHERE time>"
            + str(time_now - 2)
            + " && time<"
            + str(time_now)
        )
        cur.execute(sql)

        # fetch the result
        device_id = cur.fetchall()
        device_id = list(set([dev[0] for dev in device_id]))

        return device_id

    def globe_distance(self, lat1, lon1, lat2, lon2):

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

    def get_sta_delta(self, db, sta, **kwargs):

        # connect to the database
        mydb = mysql.connector.connect(
            host=db["host"],
            user=db["user"],
            passwd=db["passwd"],
            database=db["db_name"],
        )

        # set the database pointer
        cur = mydb.cursor()

        # get the detections from the particular station
        sql = (
            "SELECT \
            latitude, longitude \
            FROM devices WHERE device_id="
            + sta
        )
        cur.execute(sql)

        # fetch the result
        sta_coord = cur.fetchall()

        sta_lat = sta_coord[0][0]
        sta_lon = sta_coord[0][1]

        if "eq_lat" in kwargs.keys():
            eq_lat = kwargs["eq_lat"]
        else:
            eq_lat = self.location["location"][-1]["lat"]

        if "eq_lon" in kwargs.keys():
            eq_lon = kwargs["eq_lon"]
        else:
            eq_lon = self.location["location"][-1]["lon"]

        epic_dist = self.globe_distance(sta_lat, sta_lon, eq_lat, eq_lon)

        return epic_dist

    def get_all_sta_loc(self, db):

        # connect to the database
        mydb = mysql.connector.connect(
            host=db["host"],
            user=db["user"],
            passwd=db["passwd"],
            database=db["db_name"],
        )

        # set the database pointer
        cur = mydb.cursor()

        # get the detections from the particular station
        sql = "SELECT \
            device_id, latitude, longitude \
            FROM devices"
        cur.execute(sql)

        # fetch the result
        database_out = cur.fetchall()

        devices = {}

        for dev in database_out:
            devices[dev[0]] = {"lat": dev[1], "lon": dev[2]}

        return devices

    def time_since_last(self, time_now):
        """
        Get time elapsed since the last detection
        """

        last_det_time = time_now - max([n["time"] for n in self.detections.values()])

        return last_det_time

    def num_of_dets(self):
        """
        Get the number of associated detections
        """

        number_of_detections = len(
            [n["assoc"] for n in self.detections.values() if n["assoc"] == True]
        )

        return number_of_detections

    def voronoi_finite_polygons_2d(self, vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        Credit: Pauli Virtanen, github: pv
        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max() * 2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)
