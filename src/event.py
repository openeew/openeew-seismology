"""
Event module
"""

# Import modules
import numpy as np
import scipy
from inpoly import inpoly2
import pickle
import datetime
import math
import time


class Event:
    """This class handles all the detection procedures"""

    def __init__(self, devices, detections, events, travel_times, params) -> None:
        super().__init__()
        self.devices = devices
        self.detections = detections
        self.params = params
        self.events = events
        self.travel_times = travel_times
        self.active_events = {}

    def find_and_locate(self):

        # 1. Get  new detections
        new_detections = self.get_detections()

        # 2. Associate new detections with events
        # for each new detection
        for new_index, new_detection in new_detections.iterrows():

            # initially, the detection is not associated
            det_assoc = False

            for event_id in self.active_events.keys():

                # while not associate, continue trying
                det_assoc = self.associate(event_id, new_index, new_detection)

                if det_assoc == True:
                    break

            if det_assoc == False:

                # if it could not be associated with an existing event, create a new one
                self.set_new_event(new_index, new_detection)

            print("⭐ New detection at the device " + new_detection["device_id"] + ".")
            print(
                "     Associated with event id: "
                + str(self.detections.data["event_id"].iloc[-1])
            )

        # 3. Update location and magnitude of each event
        for event_id in list(self.active_events.keys()):

            # time since the last detection
            tsl = self.time_since_last(event_id)

            # Delete event if it is too old
            if tsl > self.params["tsl_max"]:
                del self.active_events[event_id]

            # Or update location, magnitude, and origin time, publish to mqtt
            else:
                self.update_events(event_id)
                self.events.publish_event(self.params, event_id=event_id)

        # 4. Drop detections and events that are older than det_ev_buffer
        self.detections.drop(self.params)
        self.events.drop(self.params)

    def get_detections(self):
        """Get new detections from the detection table"""

        # Get new detections
        new_detections = self.detections.data[self.detections.data["event_id"].isnull()]

        return new_detections

    def set_new_event(self, new_index, new_detection):
        """This sets a new event in the class"""

        # Get event ID
        try:
            event_id = self.events.data["event_id"].to_list()
            event_id.append(max(self.active_events.keys()))

            event_id = max(event_id) + 1
        except:
            event_id = 1

        self.active_events[event_id] = {}

        # Get location and magnitude based on the first detection
        self.get_loc_not_yet_arrived(event_id, new_detection)

        # Associate detection with event
        self.detections.data.loc[new_index, "event_id"] = event_id

    def prior_loc(self):
        """
        This function sets the prior probability distribution for earthquake location

        The function is rather a placeholder for a more sophisticated initial distrubution
        given by historical seismicity etc.
        """

        loc_prob = np.zeros_like(self.travel_times.grid_lat)

        return loc_prob

    def prior_mag(self):
        """
        This function sets the prior probability distribution for magnitude
        It uses the concept of magnitude of completeness and exponential
        decay of probability with increasing magnitude

        The prior probability distribution is a lineary increasing function
        (in a log10 space) from the Mc-2 to Mc (Mc is the magnitude of completenes).
        It peaks at the Mc and decreases to 0 at magnitude 10 with the slope of
        b_value (set to 1 by default)
        """

        prior_type = self.params["prior_type"]
        mc = self.params["mc"]
        b_value = self.params["b_value"]

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
        return mag_prob, mag_bins

    def get_loc_not_yet_arrived(self, event_id, new_detection):
        """
        Updates location for a new event
        """

        # get list of active stations
        active_devices = self.get_active_devices()

        # get the station with the first detection
        first_device = new_detection["device_id"]

        # get all the not-yet arrived devices
        nya_devices = list(set(active_devices) ^ set([first_device]))

        # device loc
        device_loc = self.devices.data

        # get location of all the not-yet arrived devices
        loc_nya = [
            (
                device_loc[device_loc["device_id"] == n]["longitude"].to_list()[0],
                device_loc[device_loc["device_id"] == n]["latitude"].to_list()[0],
            )
            for n in nya_devices
        ]

        # get location of all the detected device
        loc_det = [
            (
                device_loc[device_loc["device_id"] == first_device][
                    "longitude"
                ].to_list()[0],
                device_loc[device_loc["device_id"] == first_device][
                    "latitude"
                ].to_list()[0],
            )
        ]

        # append the loc_det at the beginning
        loc_all = loc_det + loc_nya

        # compute the Voronoi cells
        vor = scipy.spatial.Voronoi(loc_all)
        regions, vertices = self.voronoi_finite_polygons_2d(vor)

        # get the lat and lon grid
        lat_grid = self.travel_times.grid_lat
        lon_grid = self.travel_times.grid_lon

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
        inside, _ = inpoly2(points, polygon)

        # change the points in the polygons to 1 and out of the polygon to 0
        inside = inside.reshape(lon_grid.shape)
        inside[inside == True] = 1
        inside[inside == False] = 0

        # get the best prob
        loc_prior = self.prior_loc()
        best_prob = loc_prior + inside

        # and replace the prob with the  best prob
        self.active_events[event_id] = {"loc_prob": best_prob}

    def get_active_devices(self):
        """Grabs all the devices that are sending data

        This functions as a placeholder for more sophisticated function that
        would grab active devices from some device SOH info
        """

        try:
            device_id = self.devices.data["device_id"]
        except:
            device_id = None

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

    def get_sta_delta(self, event_id, sta, **kwargs):

        sta_lat = self.devices.data[self.devices.data["device_id"] == sta]["latitude"]
        sta_lon = self.devices.data[self.devices.data["device_id"] == sta]["longitude"]

        if "eq_lat" in kwargs.keys():
            eq_lat = kwargs["eq_lat"]
        else:
            eq_lat = self.events.data[self.events.data["event_id"] == event_id][
                "lat"
            ].iloc[-1]

        if "eq_lon" in kwargs.keys():
            eq_lon = kwargs["eq_lon"]
        else:
            eq_lon = self.events.data[self.events.data["event_id"] == event_id][
                "lon"
            ].iloc[-1]

        epic_dist = self.globe_distance(sta_lat, sta_lon, eq_lat, eq_lon)

        return epic_dist

    def time_since_last(self, event_id):
        """
        Get time elapsed since the last detection
        """
        # get timestamp for the received trace
        dt = datetime.datetime.now(datetime.timezone.utc)
        utc_time = dt.replace(tzinfo=datetime.timezone.utc)
        cloud_t = utc_time.timestamp()

        last_detection = self.detections.data[
            self.detections.data["event_id"] == event_id
        ]["cloud_t"].iloc[-1]
        last_det_time = cloud_t - last_detection

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

    def get_best_location(self, event_id, add_prob=0):

        lat = self.travel_times.grid_lat
        lon = self.travel_times.grid_lon

        # initial probability is equal to the prior
        loc_prob = self.active_events[event_id]["loc_prob"]

        # add aditional probability (for calling the function by the associator)
        loc_prob = loc_prob + add_prob

        # get best location
        best_lat = lat[loc_prob == loc_prob.max()][0]
        best_lon = lon[loc_prob == loc_prob.max()][0]
        best_depth = self.params["eq_depth"]  # depth is fixed for all

        # get first detection
        first_det = self.detections.data[
            self.detections.data["event_id"] == event_id
        ].iloc[0]

        # get origin time based on the location and the first detection
        first_sta = first_det["device_id"]
        first_time = first_det["cloud_t"]
        device_grid = self.get_device_tt_grid(first_sta, self.params)

        sta_travel_time = device_grid[loc_prob == loc_prob.max()]
        best_orig_time = first_time - sta_travel_time[0]

        return best_lat, best_lon, best_depth, best_orig_time

    def get_device_tt_grid(self, device_id, params):

        # get device latitude and longitude
        dev_lat = self.devices.data[self.devices.data["device_id"] == device_id][
            "latitude"
        ]
        dev_lon = self.devices.data[self.devices.data["device_id"] == device_id][
            "longitude"
        ]

        # get grid limits
        lat_min = params["lat_min"]
        lat_max = params["lat_max"]
        lon_min = params["lon_min"]
        lon_max = params["lon_max"]
        step = params["step"]

        # get first and last samples
        first_sample_lat = int(
            np.round(((lat_max - lat_min) - (dev_lat - lat_min)) * (1 / step))
        )
        last_sample_lat = first_sample_lat + self.travel_times.grid_lat.shape[0]
        first_sample_lon = int(
            np.round(((lon_max - lon_min) - (dev_lon - lon_min)) * (1 / step))
        )
        last_sample_lon = first_sample_lon + self.travel_times.grid_lat.shape[1]

        # get the device grid
        dev_grid = self.travel_times.tt_grid[
            first_sample_lat:last_sample_lat, first_sample_lon:last_sample_lon
        ]

        return dev_grid

    def get_magnitude(self, event_id, best_lat, best_lon):
        """
        This function uses the station magnitude estimation and calculates
        the probability distribution for the magnitude.
        It also updates the most likely magnitude and the 68 and 96 percent
        probability intervals
        """

        # get magnitude bins and prior
        mag_prob, mag_bins = self.prior_mag()

        # get all detections
        detections = self.detections.data[self.detections.data["event_id"] == event_id]

        for _, det in detections.iterrows():

            det_sta = det["device_id"]
            pd_all = det[
                ["mag1", "mag2", "mag3", "mag4", "mag5", "mag6", "mag7", "mag8", "mag9"]
            ]
            pd = [n for n in pd_all if n is not None]

            try:
                pd_type = "mag" + str(len(pd))
                pd = pd[-1]

                a = self.params[pd_type][0]
                b = self.params[pd_type][1]
                c = self.params[pd_type][2]
                std = self.params[pd_type][3]

                # Normalize the displacement for the epicentral distance of 1 km
                dist = self.get_sta_delta(
                    event_id, sta=det_sta, eq_lat=best_lat, eq_lon=best_lon
                )
                pd = np.log10(pd) + c * np.log10(dist + 1)

                # Calculate station magnitude from pd given the linear function with a, b, c
                sta_mag_mu = a * pd + b

                # generate the probability distribution for the station magnitude
                p_m_pd = scipy.stats.norm(sta_mag_mu, std).pdf(mag_bins)

                # multiply the prior and the current measurement (the Bayes happens in here)
                mag_prob = np.multiply(mag_prob, p_m_pd)

            except:
                pass

        # normalize the mag_prob
        mag_prob = mag_prob / max(np.cumsum(mag_prob))

        # get magnitude and confidence
        magnitude = mag_bins[np.argmax(mag_prob)]

        cum_prob = np.cumsum(mag_prob)
        conf2 = mag_bins[np.argmin(abs(cum_prob - 0.02))]
        conf16 = mag_bins[np.argmin(abs(cum_prob - 0.16))]
        conf84 = mag_bins[np.argmin(abs(cum_prob - 0.84))]
        conf98 = mag_bins[np.argmin(abs(cum_prob - 0.98))]

        # set initial magnitude and confidence intervals
        # (just a rough estimate)
        if magnitude == 0:
            magnitude = 4
            conf2 = 2
            conf16 = 3
            conf84 = 5.5
            conf98 = 8

        return magnitude, conf2, conf16, conf84, conf98

    def update_events(self, event_id):

        # get timestamp for the event trace
        dt = datetime.datetime.now(datetime.timezone.utc)
        utc_time = dt.replace(tzinfo=datetime.timezone.utc)
        cloud_t = utc_time.timestamp()

        # Update location
        best_lat, best_lon, best_depth, best_orig_time = self.get_best_location(
            event_id
        )

        # Update magnitude
        magnitude, mconf2, mconf16, mconf84, mconf98 = self.get_magnitude(
            event_id, best_lat, best_lon
        )

        # Number of associated phases
        num_assoc = len(
            self.detections.data[self.detections.data["event_id"] == event_id]
        )

        # Add line in events
        new_event = {
            "event_id": event_id,
            "cloud_t": cloud_t,
            "orig_time": best_orig_time,
            "lat": best_lat,
            "lon": best_lon,
            "dep": best_depth,
            "mag": magnitude,
            "mconf2": mconf2,
            "mconf16": mconf16,
            "mconf84": mconf84,
            "mconf98": mconf98,
            "num_assoc": num_assoc,
        }
        self.events.update(new_event)

        print("🔥 Event id " + str(event_id) + " in progress:")
        print(
            "     Magnitude: "
            + str(magnitude)
            + ", Lat: "
            + str(best_lat)
            + ", Lon: "
            + str(best_lon)
            + ", Associated detections: "
            + str(num_assoc)
            + "."
        )

    def associate(self, event_id, new_index, new_detection):
        """
        Calculate probabilities and associate new event
        """

        # get all detections of the event
        all_detections = self.detections.data[
            self.detections.data["event_id"] == event_id
        ]

        # get all detected devices
        detected_devices = all_detections["device_id"]

        # get the new device id and detection time
        new_detection_id = new_detection["detection_id"]
        new_device = new_detection["device_id"]
        new_time = new_detection["cloud_t"]

        # set a new list of new probabilities
        new_prob = np.zeros_like(self.travel_times.grid_lat)

        if new_device not in detected_devices:

            # loop over all associated detections
            for _, detection in all_detections.iterrows():

                # get device ID and detection time
                det_device = detection["device_id"]
                det_time = detection["cloud_t"]

                # get sigma
                sigma = self.get_sigma(event_id, new_device, det_device)

                # calculate probability curve
                grid_device_old = self.get_device_tt_grid(det_device, self.params)
                grid_device_new = self.get_device_tt_grid(new_device, self.params)

                tt_prob = np.exp(
                    -((grid_device_old - grid_device_new - det_time + new_time) ** 2)
                    / (2 * sigma ** 2)
                )

                # and add the probability the rest
                new_prob = new_prob + tt_prob

            # ASSOCIATE THE NEW DETECTION

            # get updated potential location of the eq epicenter
            best_lat, best_lon, _, _ = self.get_best_location(
                event_id, add_prob=new_prob
            )

            # test the RMS of mispics
            tt_precalc = self.travel_times.tt_vector
            misfit = []

            # get the new location

            for _, detection in all_detections.iterrows():

                det_device_old = detection["device_id"]
                det_time_old = detection["cloud_t"]

                epic_dist_old = self.get_sta_delta(
                    event_id, det_device_old, eq_lat=best_lat, eq_lon=best_lon
                )
                epic_dist_new = self.get_sta_delta(
                    event_id, new_device, eq_lat=best_lat, eq_lon=best_lon
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

            assoc_win = self.params["assoc_win"]

            if misfit_mean < assoc_win:

                # if associated, append the probabbilities
                self.active_events[event_id]["loc_prob"] = (
                    self.active_events[event_id]["loc_prob"] + new_prob
                )

                # add new detection to detections
                self.detections.data.loc[new_index, "event_id"] = event_id

                return True

            else:
                return False

    def get_sigma(self, event_id, new_device, det_device):
        """
        Get sigma from distances between the detections  and easrthquakes
        """

        # if constant  sigma is chosen
        if self.params["sigma_type"] == "const":

            sigma = self.params["sigma_const"]

        # if sigma is computed from the sigmoid function
        elif self.params["sigma_type"] == "linear":

            try:
                dist1 = self.get_sta_delta(event_id, new_device)
                dist2 = self.get_sta_delta(event_id, det_device)

                dist_ave = (dist1 + dist2) / 2

                sigma = dist_ave * 0.05 + 1
                if sigma > 8:
                    sigma = 8

            except:
                sigma = self.params["sigma_const"]

        return sigma

    def run(self):
        # run loop indefinitely
        while True:
            self.find_and_locate()
            time.sleep(self.params["sleep_time"])
