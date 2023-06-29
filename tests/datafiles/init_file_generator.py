"""Script generates exemplar Agent init condition .json files."""
# %% Imports
# Standard Library Imports
import json

# Third Party Imports
from numpy import array, concatenate, pi, shape, sqrt, zeros
from numpy.linalg import norm

# Punch Clock Imports
from scheduler_testbed.common.constants import getConstants

# %% Setup
consts = getConstants()
RE = consts["earth_radius"]
mu = consts["mu"]
# %% Satellites
sat_r0_eci = array([[RE + 400, 0, 0], [RE + 600, 0, 0], [RE + 800, 0, 0]])
sat_ids = [1, 2, 3]
num_sats = shape(sat_r0_eci)[1]
sat_v0_eci = zeros(shape(sat_r0_eci))
for i, r0 in enumerate(sat_r0_eci):
    v_mag = sqrt(mu / norm(r0))
    if i == 0:
        sat_v0_eci[i, 1] = v_mag
    elif i == 1:
        sat_v0_eci[i][1:] = v_mag / sqrt(2)
    elif i == 2:
        sat_v0_eci[i][1:] = -v_mag / sqrt(2)
    print(i, r0, sat_v0_eci[i])

sat_x0 = concatenate((sat_r0_eci, sat_v0_eci), axis=1)

sat_agents = []
for i, id in enumerate(sat_ids):
    sat_agents.append({"sat_num": id, "sat_name": id, "init_eci": list(sat_x0[i, :])})
# %% Ground stations
ground_ids = ["A", "B"]
lats = [0, 0]
lons = [0, 20 * pi / 180]
alts = [0, 0]
ground_agents = []
for id, lat, lon, alt in zip(ground_ids, lats, lons, alts):
    print(id, lat, lon, alt)
    ground_agents.append({"id": id, "lat": lat, "lon": lon, "alt": alt})
# %% Combine and save
agents = sat_agents + ground_agents
path_name = "tests/datafiles/"
file_name = "exemplar_agents_multi_inclined"
with open(path_name + file_name + ".json", "w") as final:
    json.dump(agents, final)
# %%
print("done")
