"""
This file sets parameters used in real-time OpenEEW algorithm
"""

# BUFFER AND SLEEP
sleep_time = 0  # sleep_time = 1 s to simulate real time observations
samp_rate = 31.25  # sample rate
buffer_len = 14  # buffer_len*samp_rate must be longer than array_samp

# TRAVEL TIME GRID AND CALCULATION
lat_min = 13  # minimum latitude
lat_max = 23  # maximum latitude
lon_min = -106  # minimum longitude
lon_max = -90  # maximum longitude
step = 0.01  # step in degrees
eq_depth = 20  # earthquake depth
calculate_open = "open"  # 'calculate' new travel times or 'open' existing
vel_model = "iasp91"  # velocity model from obspy list

# DETECTION
det_type = "stalta"  # 'stalta' or 'ml' for machine learning
detection_model_name = "detection_model.model"  # name of the ml model
STA_len = 32  # STA length in samples
LTA_len = 320  # LTA length in samples
array_samp = 352  # must be >= STA_len+LTA_len for 'stalta', or 300 for 'ml'
STALTA_thresh = 3  # threshold for STA/LTA
no_det_win = 60  # window without new detections after a detection
vert_chan = "x"  # which channel is oriented in the vertical direction

# LOCATION AND MAGNITUDE REGRESSION PARAMS
tsl_max = 20  # save/discard event after this many seconds without a new detection
assoc_win = 1  # window for associated phases
ndef_min = 4  # minimum number of station detections defining an event
sigma_type = "const"  # either 'const' sigma or 'linear' function
sigma_const = 3  # overall time error (travel time + pick + cloud_time)
nya_weight = 1  # how much to weight not-yet-arrived information
nya_nos = 1  # use not-yet-arrived information for this number of seconds after the first arrival
prior_type = (
    "constant"  # 'constant' or 'gutenberg' if you like to start with GR distribution
)
mc = 3  # magnitude of completeness for GR distribution
b_value = 1  # b-value for GR distribution

mag1 = (
    1.67,
    5.68,
    1,
    0.85,
)  # a, b, c, std params in M = a*pd + b, c is distance normalization, std is pd scatter
mag2 = (1.56, 5.47, 1, 0.74)
mag3 = (1.44, 5.35, 1, 0.66)
mag4 = (1.41, 5.32, 1, 0.59)
mag5 = (1.41, 5.29, 1, 0.57)
mag6 = (1.35, 5.22, 1, 0.51)
mag7 = (1.45, 5.24, 1, 0.57)
mag8 = (1.39, 5.21, 1, 0.52)
mag9 = (1.32, 5.19, 1, 0.47)


main_params = {
    "sleep_time": sleep_time,
    "buffer_len": buffer_len
}

tt_params = {
    "lat_min": lat_min,
    "lat_max": lat_max,
    "lon_min": lon_min,
    "lon_max": lon_max,
    "step": step,
    "calculate_open": calculate_open,
    "vel_model": vel_model,
    "eq_depth": eq_depth,
}

det_params = {
    "det_type": det_type,
    "STA_len": STA_len,
    "LTA_len": LTA_len,
    "STALTA_thresh": STALTA_thresh,
    "no_det_win": no_det_win,
    "samp_rate": samp_rate,
    "vert_chan": vert_chan,
    "array_samp": array_samp,
    "detection_model_name": detection_model_name,
}

ev_params = {
    "mag1": mag1,
    "mag2": mag2,
    "mag3": mag3,
    "mag4": mag4,
    "mag5": mag5,
    "mag6": mag6,
    "mag7": mag7,
    "mag8": mag8,
    "mag9": mag9,
    "tsl_max": tsl_max,
    "ndef_min": ndef_min,
    "sigma_const": sigma_const,
    "sigma_type": sigma_type,
    "nya_weight": nya_weight,
    "nya_nos": nya_nos,
    "prior_type": prior_type,
    "mc": mc,
    "b_value": b_value,
    "assoc_win": assoc_win,
    "eq_depth": eq_depth,
}
