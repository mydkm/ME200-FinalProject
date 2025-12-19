import pandas as pd
import numpy as np

df = pd.read_excel('C:/Users/jaine/Downloads/analysed.xlsx')

df.columns = df.columns.str.strip()

time = df['time'].values
altitude = df['altitude'].values
v_vert = df['velocity'].values

 
mask = time > 45  
#based on model time, only values corresponding to t>45
#are considered
time_d = time[mask]
altitude_d = altitude[mask]
v_vert_d = v_vert[mask]

# Compute change in vertical velocity
dv = np.diff(v_vert_d)

# Landing burn: where dowwnward velocity starts decreasing in magnitude
# (dv > 0 for negative downward velocity)
landing_burn_index = None
for i in range(len(dv)):
    if dv[i] > 0:
        landing_burn_index = i
        break

if landing_burn_index is not None:
    landing_burn_time = time_d[landing_burn_index]
    landing_burn_altitude = altitude_d[landing_burn_index]
    print(f"Landing burn starts at t = {landing_burn_time:.1f} s, altitude = {landing_burn_altitude:.2f} km")
else:
    print("Landing burn not detected")