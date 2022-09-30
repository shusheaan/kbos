# import os
import numpy as np
import pandas as pd
# import urllib.request
# from bs4 import BeautifulSoup

url = r'https://flightaware.com/live/airport/KBOS/enroute'
enroute = pd.read_html(url)[2]
basics  = enroute.iloc[:, [0, 1, 5]]
basics.columns = ['no', 'type', 'eta']
eta = basics.eta.str.slice(start=4, stop=11)

mask_bef = pd.to_datetime(eta) > pd.Timestamp.now()
mask_30m = (pd.to_datetime(eta) - pd.Timestamp.now()) < pd.Timedelta('15 min')
mask_sub = np.logical_and(mask_bef, mask_30m)
no_sub = basics.no[mask_sub].reset_index(drop=True)
print('Suggested Flights: ', list(no_sub))

# PENDING: live info update for photo naming, pay for flightaware API
for no in no_sub:
    url_flight = "https://flightaware.com/live/flight/" + no #.lower()
    print(url_flight)

    # os.system("wget "+url_flight)
    # tables = pd.read_html(url_flight) # not supported
    # html = BeautifulSoup(url_flight) # not supported
    # urllib.request.urlretrieve(url_flight, 'text.txt') # not supported

