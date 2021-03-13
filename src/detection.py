"""
Detection module
"""

# import modules
import tensorflow as tf
import numpy as np
from scipy import signal, integrate
import itertools
import mysql.connector

import matplotlib.pyplot as plt
from datetime import datetime

__author__ = "Vaclav Kuna"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Vaclav Kuna"
__email__ = "kuna.vaclav@gmail.com"
__status__ = ""



def load_model(model_path):

    # load model from a folder with a specified name
    model = tf.keras.models.load_model('src/' + model_path)

    return model


def data_generator(batch):

    # just a little offset
    epsilon=1e-6
            
    # does feature log
    batch_sign = np.sign(batch)
    batch_val = np.log(np.abs(batch)+epsilon)
    
    batch_out=[]
    for ii in range(batch.shape[0]):
        batch_out.append(np.hstack([batch_val[ii,:].reshape(-1,1), batch_sign[ii,:].reshape(-1,1)]))
    batch_out=np.array(batch_out)
    
    return batch_out


def buffer2array(data_buffer, samp_out):

    # get list of unique stations in the buffer
    unique_sta  = list(set([n['sta'] for n in data_buffer]))

    # create empty dictionary for data
    data_out = {}
    data_out['data'] = np.empty((0,samp_out))
    data_out['time'] = np.empty((0,samp_out))
    data_out['sta'] = []
    data_out['chan'] = []

    # iterate over unique seconds
    for sta in unique_sta:
        
        # create data arrays for x, y, z
        x = [n['x'] for n in data_buffer if n['sta']==sta]
        x = list(itertools.chain.from_iterable(x))

        y = [n['y'] for n in data_buffer if n['sta']==sta]
        y = list(itertools.chain.from_iterable(y))

        z = [n['z'] for n in data_buffer if n['sta']==sta]
        z = list(itertools.chain.from_iterable(z))

        time = [n['time'] for n in data_buffer if n['sta']==sta]
        time = list(itertools.chain.from_iterable(time))

        # if the data array is longer than samp_out samples
        if len(time)>samp_out:
            # crop the arrays to the last samp_out samples (depends on the CNN)
            x = np.array(x[-samp_out:])
            y = np.array(y[-samp_out:])
            z = np.array(z[-samp_out:])
            time = np.array(time[-samp_out:])

            # add the array to the big array and append the specs
            data_out['data'] = np.vstack((data_out['data'], x))
            data_out['data'] = np.vstack((data_out['data'], y))
            data_out['data'] = np.vstack((data_out['data'], z))
            
            data_out['time'] = np.vstack((data_out['time'], time))
            data_out['time'] = np.vstack((data_out['time'], time))
            data_out['time'] = np.vstack((data_out['time'], time))

            # append station name and channel in the list
            data_out['sta'].append(sta)
            data_out['sta'].append(sta)
            data_out['sta'].append(sta)

            data_out['chan'].append('x')
            data_out['chan'].append('y')
            data_out['chan'].append('z')

    return data_out


def detection2db(db, device_id, det_time, params):

    """The function checks whether there are detections within the past number
    of seconds specified by win. If not, it adds detections into the detection table
    """

    # connect to the database
    mydb = mysql.connector.connect(
        host=db['host'],
        user=db['user'],
        passwd=db['passwd'], 
        database=db['db_name']
    ) 

    # set the database pointer
    cur = mydb.cursor()

    # GET THE PAST DETECTIONS
    # number of seconds without new detections
    det_off_win = params['no_det_win']

    # get the detections less than 10 s old
    sql = "SELECT device_id, time FROM detections WHERE time>" \
        + str(det_time-det_off_win) + " && time<" + str(det_time+det_off_win) \
        + " && device_id=" + device_id
    cur.execute(sql)

    # fetch the result
    old_detections = cur.fetchall()

    if not old_detections:

        # make sql command
        sql = "INSERT INTO detections (device_id, time) VALUES (%s, %s)"

        entry = (device_id, det_time)
        cur.execute(sql, entry)
    
        time_string = datetime.utcfromtimestamp(det_time).strftime('%Y-%m-%d %H:%M:%S')
        print('Station {}, Detection time: {}'.format(device_id, time_string))

        # commit changes in the table
        mydb.commit()

        # you should close what you've opened
        mydb.close()
        cur.close()
    
    else:
        pass


def detect_ml(model, data_array, db, time_now, params):

    data = data_array['data']
    sta = data_array['sta']
    chan = data_array['chan']
    time = data_array['time']

    # if the data is not empty, run the detector
    if data.shape[0]>0:

        # detrend
        data = signal.detrend(data, axis=-1, type='constant')

        # and normalize
        roow_max = np.max(np.abs(data), axis=1)
        roow_max = roow_max.reshape((len(roow_max),1))
        data = data/roow_max

        # See how things went
        batch_out = data_generator(data)

        # GET PREDICTIONS
        predictions = model.predict(batch_out)

        # set limits for detection
        det_thresh = .2
        det_delay = 320

        # no detections in first 1 second
        cut = 1
        cut_edges = np.hstack((np.zeros((cut*32,)), np.ones((len(data[0,:])-(2*cut*32),)), np.zeros((cut*32,))))

        for wf in range(predictions.shape[0]):

            pred_curr = predictions[wf] * cut_edges
            time_curr = time[wf,:]

            # Finding signal detections withing the predictions
            peaks_pred, _ = signal.find_peaks(pred_curr, height=det_thresh, distance=det_delay)

            # if there are detections
            if peaks_pred.shape[0]>0:

                # if sta[wf]=='000':
                #     plt.plot(time[wf,:], pred_curr)
                #     plt.plot(time[wf,:], data[wf,:])
                #     plt.show()

                # add each detection in a detection list
                for det in peaks_pred:
                    device_id = sta[wf]
                    channel = chan[wf]
                    det_time = time_curr[det]

                    # and add the array in the detection db
                    detection2db(db, device_id, det_time, params)


def standard_STALTA(trace, STA_len, LTA_len):

    # check if there is enough data for calculation,
    # otherwise return False
    if len(trace)<(STA_len + LTA_len):
        return False

    # demean detrend do something
    trace = trace - np.mean(trace)
    trace = abs(trace)

    # calculate first LTA and STA
    LTA_first = np.mean(trace[0:LTA_len])
    STA_first = np.mean(trace[LTA_len-STA_len:LTA_len])

    # the beginning of the LTA
    LTA_beg_vector = trace[0:STA_len] # take the start of the array
    LTA_beg_mat = np.tile(LTA_beg_vector, (STA_len, 1)) # repeat the vector in matrix
    LTA_beg_mat = np.tril(LTA_beg_mat, k=-1) # do the lower triangular transformation

    # the end of the LTA and STA (they  are the same)
    STALTA_end_vector = trace[LTA_len:LTA_len+STA_len] # take the start of the array
    STALTA_end_mat = np.tile(STALTA_end_vector, (STA_len, 1)) # repeat the vector in matrix
    STALTA_end_mat = np.tril(STALTA_end_mat, k=-1) # do the lower triangular transformation

    # the beginning of the STA
    STA_beg_vector = trace[LTA_len-STA_len:LTA_len] # take the start of the array
    STA_beg_mat = np.tile(STA_beg_vector, (STA_len, 1)) # repeat the vector in matrix
    STA_beg_mat = np.tril(STA_beg_mat, k=-1) # do the lower triangular transformation
    
    # calculate LTA vector
    LTA = LTA_first + (np.sum(STALTA_end_mat, axis=1) - np.sum(LTA_beg_mat, axis=1))/LTA_len

    # calculate STA vector
    STA = STA_first + (np.sum(STALTA_end_mat, axis=1) - np.sum(STA_beg_mat, axis=1))/STA_len

    # calculate STA/LTA
    STALTA = STA/LTA
    
    return STALTA


def detect_stalta(data_array, db, time_now, params):

    STA_len = params['STA_len']
    LTA_len = params['LTA_len']
    STALTA_thresh = params['STALTA_thresh']

    # if the data is not empty, run the detector
    for line in range(data_array['data'].shape[0]):

        trace = data_array['data'][line]
        time = data_array['time'][line]
        device_id = data_array['sta'][line]

        ttest = data_array['time'][line][-1]
        # time = data_array['time'][line][-1]-np.arange(start=0, stop=len(trace))[::-1]/31.25


        STALTA = standard_STALTA(trace, STA_len, LTA_len)
        
        (ind, ) = np.where(STALTA>STALTA_thresh)

        if ind.size > 0:
            det_time = time[LTA_len+ind[0]]

            # if device_id=='014' and data_array['chan'][line]=='x':
            #     plt.plot(time, trace, marker='.')
            #     plt.plot(time[-STA_len:], STALTA)
            # plt.show()

            # and add the array in the detection db
            detection2db(db, device_id, det_time, params)


def get_pd(trace, time, det_time, params):

    '''
    The function receives a trace and calculates the
    peak ground displacement
    '''

    # define variables
    samp_freq = params['samp_rate'] # definition of sampling frequency

    # double integration of stream in displacement
    trace = np.cumsum(trace*1/samp_freq)
    trace = np.cumsum(trace*1/samp_freq)

    # filrtation of the signal
    sos = signal.butter(4, (.2, 3), 'bandpass', fs=samp_freq, output='sos')
    trace = signal.sosfilt(sos, trace)

    # reshape time
    time = np.reshape(time,(time.size,))

    # get the hilbert envelope of the signal
    hilb = signal.hilbert(trace)
    hilb = np.abs(hilb)

    # create a trace long enough to contain 10-s of signal
    eval_trace = np.empty((300,))
    eval_trace[:] = np.nan

    # and fill in the values from the hilbert trace
    after_det = hilb[time>det_time]
    eval_trace[0:len(after_det)] = after_det

    # get the maximum of the displacement within the window
    pd_len = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    pd_max = [eval_trace[0:int(n*samp_freq)].max() for n in pd_len]

    # plot
    # plt.plot(trace, linewidth=.5)
    # plt.plot(hilb, linewidth=.5)
    # plt.plot(eval_trace, linewidth=.5)
    # plt.show()

    # Return the peak ground displacement
    return pd_max


def station_magnitude(data_array, db, time_now, params):

    """This function updates the detection table with
    peak ground displacements

    INPUT:
    data_buffer : data buffer
    db_name : Database name
    timestamp : Bring in this second
    """
    
    # signal rate
    vert_chan = params['vert_chan']

    # connect to the database
    mydb = mysql.connector.connect(
        host=db['host'],
        user=db['user'],
        passwd=db['passwd'], 
        database=db['db_name']
    ) 

    # set the database pointer
    cur = mydb.cursor()

    # get the detections less than 10 s old
    sql = "SELECT detection_id, device_id, time FROM detections WHERE time>" + str(time_now-10) + " && time<" + str(time_now)
    cur.execute(sql)

    # fetch the result
    detections = cur.fetchall()

    # for each detection, do
    for det in detections:
        
        try:
            # get the index of the particular station and channel trace
            sta_ind = [index for index, elem in enumerate(data_array['sta']) if elem==det[1]]
            chan_ind = [index for index, elem in enumerate(data_array['chan']) if elem==vert_chan]
            index = list(set(sta_ind).intersection(chan_ind))
            # print(sta_ind)
            # print(chan_ind)

            # retrieve the trace and time
            trace = data_array['data'][index,:]
            time = data_array['time'][index,:]
            det_time = det[2]

            # get the peak ground displacement for [1,2,3,4,5,6,7,8,9] s wndows
            pd_max = get_pd(trace, time, det_time, params)
            entry = tuple([None if np.isnan(n) else n for n in pd_max])

            # update in the database
            sql = 'UPDATE detections SET \
                mag1 = %s, \
                mag2 = %s, \
                mag3 = %s, \
                mag4 = %s, \
                mag5 = %s, \
                mag6 = %s, \
                mag7 = %s, \
                mag8 = %s, \
                mag9 = %s \
                WHERE detection_id=' + str(det[0])
            cur.execute(sql, entry)

        except:
                pass

    mydb.commit()


def detect(model, data_buffer, db, time_now, params):

    # First of all, convert the data_buffer to array from all the stations
    # The data array is a dictionary that contains 4 keys
    #     data : (n x samp_out) array with data
    #     sta : (n x 1) array of stations
    #     chan : (n x 1) array of channels
    #     time : (n x samp_out) array with time

    # number of output samples from buffer2array
    samp_out = params['array_samp'] # has to be 300 for ml detection
    
    # convert the data_buffer to format for the detector
    data_array = buffer2array(data_buffer, samp_out)

    if params['det_type']=='ml':

        # do ML detection
        detect_ml(
            model=model,
            data_array=data_array,
            db=db,
            time_now=time_now,
            params=params
        )

    elif params['det_type']=='stalta':

        # do ML detection
        detect_stalta(
            data_array=data_array,
            db=db,
            time_now=time_now,
            params=params
        )

    # do station magnitude
    station_magnitude(
        data_array=data_array,
        db=db,
        time_now=time_now,
        params=params
    )

