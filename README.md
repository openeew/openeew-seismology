# OpenEEW seismological code

## Introduction

To provide useful earthquake alerts, an Earthquake Early Warning (EEW) system has to recognize strong earthquakes quickly and reliably. The speed of the alert, that is, the time elapsed between the origin of an earthquake and the delivery of the alert to a user, determines the time available for the user to take protective action. The alert reliability is necessary for ensuring that the system issues alerts for all strong earthquakes and does not issue alerts for small earthquakes or no earthquakes at all.

This code receives raw data from the seismic network (or uses historical data), detects and recognizes earthquake signals and evaluates the earthquake's magnitude and location. This is a short overview of the first version of the code.

## Main processing components
In general, the seismological apparatus for recognition and characterization of earthquakes need to comprise four major components:
*	Earthquake detection: An algorithm for detection of significant (earthquake) shaking at individual seismic stations.
*	Event association: An algorithm evaluating whether individual detections belong to a common source (earthquake).
*	Magnitude determination: An algorithm for magnitude estimation.
*	Location determination: An algorithm for hypocentral/epicentral location estimation.

## Overview of the code structure
There are four folders in the repo:
*	**data:** contains historical earthquake data and csv file with location of devices
*	**obj:** this is where events and travel time tables are saved
*	**src:** main codes that will run on the server
*	**utils:** helper codes for preparation of historical data, plotting events and detections etc.

## Using historical data
For the testing purposes, the algorithm is set up to use historical data. The data from 16 intermediate-to-major earthquakes is stores in data/ folder.

## Major code components

The diagram shows the basic flow of the program.

![seismology_roadmap_v2_server](https://user-images.githubusercontent.com/37088604/111028616-08efe680-83f8-11eb-8cce-367c873da914.png)

*	### main.py
The program is run by running main.py. In its current form, it takes the historical data and supplies them to the algorithms. The loop runs once a second.

*	### params.py
All the parameters used by the program are defined in here.

*	### detection.py

The detection.py handles the detection of earthquake primary (P) earthquake waves as well as the station magnitude calculation.

**Input:** Sensor data - three component records of ground motion acceleration, data arrive in 1 second increments
**Output:** Detections with station magnitude in the detections table in the db
**Detection:** The detection is using one of the following methods: STA/LTA - taking a ratio of short-term vs. long-term average of the signal and ML - a convolutional neural network that recognizes P waves. Currently, I recommend using STA/LTA as the ML model still needs some more work.
**Station magnitude:** Is based on the peak ground displacement (e.g. Lancieri and Zollo, 2008; Li et al., 2017; Wu and Zhao, 2006) (doing a double integration of the first 2-4 seconds of the P wave).

* ### event.py
The event.py handles association, location and magnitude calculation of earthquakes.

**Input:** Uses detections from the detections table in db
**Output:** Earthquake events saved in obj/events

The event.py creates an event instance of a class that stores all the important information about the event. You can see the structure in the diagram below.

![event_scheme](https://user-images.githubusercontent.com/37088604/111028535-8404cd00-83f7-11eb-86ab-3e96e73ad175.png)

**Event association:** Event association is a process that gathers individual phase picks into events and throws away picks that are not associated to earthquakes (and are probably just noise).

**Event location:** We adopted real-time Bayesian evolutionary algorithm (e.g. Satriano et al., 2008). It is a grid search that tries to minimize misfits of P wave travel-time differences to different stations, i.e. it is independent of the origin time. It uses the concept of 'not yet arrived' data, which means that when you receive data at the first station, you already can have a very rough idea about the earthquake location.

**Magnitude determination:** We used Bayesian approach to calculate the magnitude from the peak ground displacement in the window from 1  to 9 s. The parameters for the calculation were determined from ~1000 earthquake records recorded at Mexico network. This saturates at about M7.5.

## Quick start

This seismology repo utilizes MQTT to receive data. In order to utilize this repo we'll need to set up an MQTT server. If you already have one set up feel free to skip this step.

To set up a local MQTT server you can follow the instructions on https://hub.docker.com/_/eclipse-mosquitto or simply

```
docker run -it -p 9001:9001 -p 1883:1883 --detach eclipse-mosquitto:latest mosquitto -c /mosquitto-no-auth.conf
```

### Simulate sensor data

This repository contains sample data that can be used to feed the seismology algorithms. To simulate this data we have included
several data simulators. Each of these simulators communicate with an MQTT server. The simulaters accept several options that can be listed by running ```python3 <simulator_script>.py -h```. If no arguments are passed, the simulators will assume MQTT is on localhost, port 1883. The simulators by default, will utilize the data contained in the data directory of this repository.

To execute the device simulator:
```
cd utils
python3 simulate_devices.py
```

To execute the traces simulator:
```
cd utils
python3 simulate_traces.py
```

