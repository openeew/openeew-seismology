import numpy as np
import pandas as pd
import os
from subprocess import Popen, PIPE
import glob


def get_trace(device_id, year, month, day, hour, minutes):

    eq_name = str(year) + "_" + str(month) + "_" + str(day)

    for m in minutes:

        # copy all files from the hour to folder temp
        rel_path = "records/country_code=mx/device_id={}/year={}/month={:02.0f}/day={:02.0f}/hour={:02.0f}/{:02.0f}.jsonl".format(
            device_id, year, month, day, hour, m
        )

        cmd = "aws s3 cp s3://grillo-openeew/{} data/{}/{}/{:02.0f}.jsonl --no-sign-request".format(
            rel_path, eq_name, sta, m
        )
        print(cmd)
        os.system(cmd)


sta_list = [
    "000",
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "019",
    "020",
    "021",
    "022",
    "023",
    "024",
    "025",
    "026",
    "027",
    "029",
    "099",
    "d11",
    "094",
    "033",
]

events = [
    (2017, 12, 15, 23, [10]),
    (2018, 1, 29, 17, [40]),
    (2018, 2, 16, 23, [35, 40]),
    (2017, 12, 16, 4, [5]),
    (2017, 12, 25, 20, [20]),
    (2018, 1, 8, 17, [0]),
]

for event in events:

    yr = event[0]
    mo = event[1]
    day = event[2]
    hr = event[3]
    mi = event[4]

    for sta in sta_list:

        try:
            get_trace(sta, yr, mo, day, hr, mi)
        except:
            pass
