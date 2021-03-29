"""
Detection module
"""

# import modules
import tensorflow as tf
import numpy as np
from scipy import signal, integrate
import itertools
import mysql.connector
import time
import pandas as pd

import matplotlib.pyplot as plt
from datetime import datetime

__author__ = "Vaclav Kuna"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Vaclav Kuna"
__email__ = "kuna.vaclav@gmail.com"
__status__ = ""


class Detect:
    """This class handles all the detection procedures"""

    def __init__(self, raw_data, detections, params) -> None:
        super().__init__()
        self.raw_data = raw_data
        self.detections = detections
        self.params = params


    def detect(self):

        if self.params["det_type"] == "stalta":
            # do STA/LTA detection
            self.detect_stalta()

        # elif params["det_type"] == "ml":
        #     # do ML detection
        #     self.detect_ml(
        #         model=model, data_array=data_array, db=db, time_now=time_now, params=params
        #     )

        # do station magnitude
        self.station_magnitude()


    def detect_stalta(self):

        STA_len = self.params["STA_len"]
        LTA_len = self.params["LTA_len"]
        STALTA_thresh = self.params["STALTA_thresh"]
        det_off_win = self.params["no_det_win"]

        try:
            devices = self.raw_data.data['device_id'].unique()
        except:
            devices = []
            
        for device in devices:

            det_time = []

            for channel in ['x', 'y', 'z']:

                trace = self.raw_data.data[self.raw_data.data['device_id']==device][channel].iloc[-(LTA_len+STA_len):]
                time = self.raw_data.data[self.raw_data.data['device_id']==device]['cloud_t'].iloc[-(LTA_len+STA_len):]

                try:
                    STALTA = self.standard_STALTA(trace)
                    (ind,) = np.where(STALTA > STALTA_thresh)
                    det_time.append(time.iloc[LTA_len + ind[0]])
                except:
                    pass

            if len(det_time) > 0:

                past_detections = self.detections.data[(self.detections.data['device_id']==device) & 
                    (self.detections.data['cloud_t']-max(det_time)+det_off_win)>0]
                
                if past_detections.shape[0]==0:

                    try:
                        detection_id = self.detections.data['detection_id'].iloc[-1] + 1
                    except:
                        detection_id = 1

                    new_detection = pd.DataFrame({'detection_id': detection_id,'device_id': device, 'cloud_t': min(det_time),
                        'mag1': None, 'mag2': None, 'mag3': None, 'mag4': None, 'mag5': None,
                        'mag6': None, 'mag7': None, 'mag8': None, 'mag9': None, 'event_id': None
                        }, index=[0])

                    self.detections.update(new_detection)

    
    def standard_STALTA(self, trace):

        STA_len = self.params["STA_len"]
        LTA_len = self.params["LTA_len"]

        # check if there is enough data for calculation,
        # otherwise return False
        if len(trace) < (STA_len + LTA_len):
            return False

        # demean detrend do something
        trace = trace - np.mean(trace)
        trace = abs(trace)

        # calculate first LTA and STA
        LTA_first = np.mean(trace[0:LTA_len])
        STA_first = np.mean(trace[LTA_len - STA_len : LTA_len])

        # the beginning of the LTA
        LTA_beg_vector = trace[0:STA_len]  # take the start of the array
        LTA_beg_mat = np.tile(LTA_beg_vector, (STA_len, 1))  # repeat the vector in matrix
        LTA_beg_mat = np.tril(LTA_beg_mat, k=-1)  # do the lower triangular transformation

        # the end of the LTA and STA (they  are the same)
        STALTA_end_vector = trace[
            LTA_len : LTA_len + STA_len
        ]  # take the start of the array
        STALTA_end_mat = np.tile(
            STALTA_end_vector, (STA_len, 1)
        )  # repeat the vector in matrix
        STALTA_end_mat = np.tril(
            STALTA_end_mat, k=-1
        )  # do the lower triangular transformation

        # the beginning of the STA
        STA_beg_vector = trace[LTA_len - STA_len : LTA_len]  # take the start of the array
        STA_beg_mat = np.tile(STA_beg_vector, (STA_len, 1))  # repeat the vector in matrix
        STA_beg_mat = np.tril(STA_beg_mat, k=-1)  # do the lower triangular transformation

        # calculate LTA vector
        LTA = (
            LTA_first
            + (np.sum(STALTA_end_mat, axis=1) - np.sum(LTA_beg_mat, axis=1)) / LTA_len
        )

        # calculate STA vector
        STA = (
            STA_first
            + (np.sum(STALTA_end_mat, axis=1) - np.sum(STA_beg_mat, axis=1)) / STA_len
        )

        # calculate STA/LTA
        STALTA = STA / LTA

        return STALTA

    
    def get_pd(self, trace, time, det_time):

        """
        The function receives a trace and calculates the
        peak ground displacement
        """

        # define variables
        sr = self.params["samp_rate"]  # definition of sampling frequency

        # double integration of stream in displacement
        trace = np.cumsum(trace * 1 / sr)
        trace = np.cumsum(trace * 1 / sr)

        # filrtation of the signal
        sos = signal.butter(4, (0.2, 3), "bandpass", fs=sr, output="sos")
        trace = signal.sosfilt(sos, trace)

        # reshape time
        time = np.reshape(time, (time.size,))

        # get the hilbert envelope of the signal
        hilb = signal.hilbert(trace)
        hilb = np.abs(hilb)

        # create a trace long enough to contain 10-s of signal
        eval_trace = np.empty((300,))
        eval_trace[:] = np.nan

        # and fill in the values from the hilbert trace
        after_det = hilb[time > det_time]
        eval_trace[0 : len(after_det)] = after_det

        # get the maximum of the displacement within the window
        pd_len = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        pd_max = [eval_trace[0 : int(n * sr)].max() for n in pd_len]

        # Return the peak ground displacement
        return pd_max


    def station_magnitude(self):

        """This function updates the detection table with
        peak ground displacements

        INPUT:
        data_buffer : data buffer
        db_name : Database name
        timestamp : Bring in this second
        """

        # what is the vertical channel?
        vert_chan = self.params["vert_chan"]
        
        # what the time is
        try:
            time_now = self.raw_data.data['cloud_t'].iloc[-1]
        except:
            return

        # get the detections less than 10 s old
        detections = self.detections.data[(self.detections.data['cloud_t']-time_now+10)>0]

        # for each detection, do
        for index, detection in detections.iterrows():

            try:
                device_id = detection['device_id']
                det_cloud_t = detection['cloud_t']

                trace = self.raw_data.data[(self.raw_data.data['device_id']==device_id) & (self.raw_data.data['cloud_t']>det_cloud_t)][vert_chan]
                time = self.raw_data.data[(self.raw_data.data['device_id']==device_id) & (self.raw_data.data['cloud_t']>det_cloud_t)]['cloud_t']

                # get the peak ground displacement for [1,2,3,4,5,6,7,8,9] s windows
                pd_max = self.get_pd(trace, time, det_cloud_t)

                entry = tuple([None if np.isnan(n) else n for n in pd_max])

                self.detections.data.loc[index, 'mag1'] = entry[0]
                self.detections.data.loc[index, 'mag2'] = entry[1]
                self.detections.data.loc[index, 'mag3'] = entry[2]
                self.detections.data.loc[index, 'mag4'] = entry[3]
                self.detections.data.loc[index, 'mag5'] = entry[4]
                self.detections.data.loc[index, 'mag6'] = entry[5]
                self.detections.data.loc[index, 'mag7'] = entry[6]
                self.detections.data.loc[index, 'mag8'] = entry[7]
                self.detections.data.loc[index, 'mag9'] = entry[8]

            except:
                pass


    def run(self):
        # run loop indefinitely
        while True:
            self.detect()
            time.sleep(.5)



# # FUNCTIONS FOR ML IMPLEMENTATION


# def detect_ml(model, data_array, db, time_now, params):

#     data = data_array["data"]
#     sta = data_array["sta"]
#     chan = data_array["chan"]
#     time = data_array["time"]

#     # if the data is not empty, run the detector
#     if data.shape[0] > 0:

#         # detrend
#         data = signal.detrend(data, axis=-1, type="constant")

#         # and normalize
#         roow_max = np.max(np.abs(data), axis=1)
#         roow_max = roow_max.reshape((len(roow_max), 1))
#         data = data / roow_max

#         # See how things went
#         batch_out = data_generator(data)

#         # GET PREDICTIONS
#         predictions = model.predict(batch_out)

#         # set limits for detection
#         det_thresh = 0.2
#         det_delay = 320

#         # no detections in first 1 second
#         cut = 1
#         cut_edges = np.hstack(
#             (
#                 np.zeros((cut * 32,)),
#                 np.ones((len(data[0, :]) - (2 * cut * 32),)),
#                 np.zeros((cut * 32,)),
#             )
#         )

#         for wf in range(predictions.shape[0]):

#             pred_curr = predictions[wf] * cut_edges
#             time_curr = time[wf, :]

#             # Finding signal detections withing the predictions
#             peaks_pred, _ = signal.find_peaks(
#                 pred_curr, height=det_thresh, distance=det_delay
#             )

#             # if there are detections
#             if peaks_pred.shape[0] > 0:

#                 # if sta[wf]=='000':
#                 #     plt.plot(time[wf,:], pred_curr)
#                 #     plt.plot(time[wf,:], data[wf,:])
#                 #     plt.show()

#                 # add each detection in a detection list
#                 for det in peaks_pred:
#                     device_id = sta[wf]
#                     channel = chan[wf]
#                     det_time = time_curr[det]

#                     # and add the array in the detection db
#                     detection2db(db, device_id, det_time, params)


# def load_model(model_path):

#     # load model from a folder with a specified name
#     model = tf.keras.models.load_model("src/" + model_path)

#     return model


# def data_generator(batch):

#     # just a little offset
#     epsilon = 1e-6

#     # does feature log
#     batch_sign = np.sign(batch)
#     batch_val = np.log(np.abs(batch) + epsilon)

#     batch_out = []
#     for ii in range(batch.shape[0]):
#         batch_out.append(
#             np.hstack(
#                 [batch_val[ii, :].reshape(-1, 1), batch_sign[ii, :].reshape(-1, 1)]
#             )
#         )
#     batch_out = np.array(batch_out)

#     return batch_out