import numpy as np
from io import StringIO
from astropy.constants import c
import matplotlib.pyplot as plt
import itertools
from astropy.coordinates import Angle
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, ITRS, CartesianRepresentation

def _earthlocation_to_altaz(location, reference_location):
    # See
    # https://docs.astropy.org/en/stable/coordinates/common_errors.html#altaz-calculations-for-earth-based-objects
    # for why this is necessary and we cannot just do
    # `get_itrs().transform_to(AltAz())`
    itrs_cart = location.get_itrs().cartesian
    itrs_ref_cart = reference_location.get_itrs().cartesian
    local_itrs = ITRS(itrs_cart - itrs_ref_cart, location=reference_location)
    return local_itrs.transform_to(AltAz(location=reference_location))

def earth_location_to_local_enu(location, reference_location):
    altaz = _earthlocation_to_altaz(location, reference_location)
    ned_coords =  altaz.cartesian.xyz
    enu_coords = ned_coords[1], ned_coords[0], -ned_coords[2]
    return enu_coords

antenna_config_path = "/opt/casa-6.6.0-20-py3.8.el7/data/alma/simmos/"
antenna_config_file = antenna_config_path + "vla.d.cfg"
dtype=[('x','f4'),('y','f4'),('z','f4'),('D','f4'),('id','S5')]
data = np.loadtxt(antenna_config_file, dtype=dtype)
local_xyz = EarthLocation.from_geocentric(data["x"], data["y"], data["z"], u.m)
telescope_center = EarthLocation.of_site("vla")
enu_coords = np.array(earth_location_to_local_enu(local_xyz, telescope_center))