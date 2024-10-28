"""
Laboratorio Interferometria
Autor: David Valero Croma ;
Laboratorio de interferometria en cual se simula covertura uv, se utiliza gridding y se obtiene la imagen de la fuente.
"""
import numpy as np
from astropy.constants import c
from astropy.coordinates import EarthLocation, AltAz, ITRS
from astropy import units as u
import matplotlib.pyplot as plt

def uvw_coordinates(local_xyz, H0, delta0):
    # Matriz de rotación basada en H0 y delta0
    rotation_matrix = np.array([
        [np.sin(H0), np.cos(H0), 0],
        [-np.sin(delta0) * np.cos(H0), np.sin(delta0) * np.sin(H0), np.cos(delta0)],
        [np.cos(delta0) * np.cos(H0), -np.cos(delta0) * np.sin(H0), np.sin(delta0)]
    ])
    return np.dot(rotation_matrix, local_xyz.T).T  # Coordenadas u, v, w


# Constantes
frequency = 1.4e9  # Frecuencia en Hz (ejemplo), cambiar según sea necesario
wavelength = c.value / frequency  # Longitud de onda

# Cargar posiciones de antenas desde el archivo de configuración
dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('D', 'f4'), ('id', 'S5')]
antenna_data = np.loadtxt('alma.out02.cfg', dtype=dtype)

# Convertir posiciones de antenas a coordenadas locales ENU
local_xyz = EarthLocation.from_geocentric(antenna_data["x"], antenna_data["y"], antenna_data["z"], u.m)
reference_location = EarthLocation.of_site("vla")  # Cambiar al centro de referencia si es necesario


# Ejemplo de aplicación con valores de H0 y delta0 (definir según datos)
H0 = 0.5  # Radianes, ejemplo
delta0 = 0.25  # Radianes, ejemplo
uvw_coords = uvw_coordinates(local_xyz, H0, delta0)

# Conversión a número de longitudes de onda
u_lambda = uvw_coords[:, 0] * frequency / c.value
v_lambda = uvw_coords[:, 1] * frequency / c.value
w_lambda = uvw_coords[:, 2] * frequency / c.value


#----------Parte 2 Gridding----------------
N = 256  # Tamaño de la grilla
delta_u = 1.0  # Tasa de muestreo en el espacio u (ajustar según necesidad)
delta_v = 1.0  # Tasa de muestreo en el espacio v (ajustar según necesidad)

VG = np.zeros((N, N), dtype=complex)  # Matriz de visibilidades
WG = np.zeros((N, N))  # Matriz de pesos

# Función de gridding
for u, v in zip(u_lambda, v_lambda):
    i = int(round(u / delta_u) + N // 2)
    j = int(round(v / delta_v) + N // 2)
    if 0 <= i < N and 0 <= j < N:
        VG[i, j] += 1  # Visibilidad acumulada (cambiar según visibilidad calculada)
        WG[i, j] += 1  # Peso (ajustar según el peso ω)

# Normalización de VG
VG /= WG

