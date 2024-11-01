"""
Proyecto de interferometria de antenas
nombre: David Valero Croma
fecha: 30/10/2024
proyceto que simula la interferometria de antenas en base a un archivo de configuracion de antenas en ENU, genera el uv coverage y la grilla de la misma.
"""
import numpy as np
from astropy.constants import c
from astropy.coordinates import EarthLocation, AltAz, ITRS
from astropy.time import Time
from astropy import units as u
import matplotlib.pyplot as plt
import math


#Funcion para leer archivo de configuracion de antenas ENU
def read_antenna_data(file_path):
    antennas = []
    with open(file_path, 'r') as file:
        for line in file:
            # Ignorar líneas de comentarios
            if line.startswith('#'):
                continue
            # Parsear las columnas: E, N, U
            columns = line.split()
            if len(columns) >= 3:
                x, y, z = map(float, columns[:3])  # Obtener las tres primeras columnas como float
                antennas.append({'x': x, 'y': y, 'z': z})
    return antennas

#Funcion para convertir ENU a Azimuth y Altitud
def enu_to_altaz(antenna_data):
    """
    Converts ENU coordinates to azimuth and elevation angles.

    Parameters:
    antenna_data (List[Dict[str, float]]): List of dictionaries with 'x', 'y', 'z' coordinates.

    Returns:
    List[Dict[str, float]]: List of dictionaries with azimuth and elevation angles for each antenna.
    """
    az_el_data = []
    for data in antenna_data:
        x = data['x']
        y = data['y']
        z = data['z']
        #Calculamos baseline
        baseline = np.sqrt(x**2 + y**2 + z**2)
        # Calculate azimuth
        azimuth = math.atan2(x, y)
        if azimuth < 0:
            azimuth += 2*np.pi  # Ensure azimuth is in the range [0, 360]
        # Calculate elevation
        elevation = np.arcsin(z/baseline)
        az_el_data.append({
            'azimuth': azimuth,
            'elevation': elevation,
            'baseline': baseline,
        })
    return az_el_data



#Funcion para convertir Azimuth y Altitud a coordenadas Ecuatoriales y XYZ
def az_el_to_equatorial_xyz(az_el_data, lat):
    # Convert latitude and HSL to radians
    lat_rad = math.radians(lat)
    equatorial_xyz_data = []
    for data in az_el_data:
        # Convert azimuth and elevation to radians
        azimuth_rad = data['azimuth']
        elevation_rad = data['elevation']

        equatorial_xyz_data.append({    # Declination in degrees
            'x': data['baseline']*(math.sin(elevation_rad) * math.cos(lat_rad) - math.cos(elevation_rad) * math.sin(lat_rad) * math.cos(azimuth_rad)),
            'y': data['baseline']*(math.sin(azimuth_rad)*math.cos(elevation_rad)),
            'z': data['baseline']*(math.sin(elevation_rad) * math.sin(lat_rad) + math.cos(elevation_rad) * math.cos(lat_rad) * math.cos(azimuth_rad)),
        })
    return equatorial_xyz_data

#Funcion para convertir coordenadas XYZ a UV
def xyz_to_uv(equatorial_xyz_data, H0, delta):
    uvw_coords = []
    for data in equatorial_xyz_data:
        # Convertir a coordenadas UV
        u = data['x']*math.sin(H0) + data['y']*math.cos(H0)
        v = -data['x']*math.sin(delta)*math.cos(H0) + data['y']*math.sin(delta)*math.sin(H0) + data['z']*math.cos(delta)
        w = data['x']*math.cos(delta)*math.cos(H0) - data['y']*math.cos(delta)*math.sin(H0) + data['z']*math.sin(delta)
        uvw_coords.append((u, v, w))
    return uvw_coords

def gaussian_2d_movable(shape, center, sigma):
    # Create a symmetric grid centered around (0, 0)
    y, x = np.ogrid[-shape[0]//2:shape[0]//2, -shape[1]//2:shape[1]//2]
    y0, x0 = center
    # Calculate the Gaussian based on the specified center
    gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return gaussian

#PARTE1: uv coverage

#----------------Convertir Coordenadas
# Leer archivo de configuración de antenas
antenas = read_antenna_data('C:/Users/david/OneDrive/Desktop/Universidad/Semestre 12/Interferometria/LAB/Lab_Interferometria/alma.cycle8.6.cfg')
# Convertir coordenadas ENU a Azimuth y Altitud
az_el_data = enu_to_altaz(antenas)
# Definir la ubicación de ALMA
Alma = EarthLocation.of_site('ALMA')

# Convertir Azimuth y Altitud a coordenadas Ecuatoriales y XYZ
ec_xyz_data = az_el_to_equatorial_xyz(az_el_data, Alma.lat.deg)

# Convertir coordenadas ecuatoriales a coordenadas UV
HA= np.linspace(0, 2*np.pi, 100)
uvw_coords = []
for i in range(len(HA)):
    uvw_coords.extend(xyz_to_uv(ec_xyz_data, HA[i], np.pi/4))

# Graficar uv coverage
uvw_coords = np.array(uvw_coords)

plt.figure(figsize=(10, 10))
plt.scatter(uvw_coords[:, 0], uvw_coords[:, 1], color="blue", s=10, label="Visibilities")
plt.scatter([-u for u in uvw_coords[:, 0]], [-v for v in uvw_coords[:, 1]], color="red", s=10, label="Conjugate Visibilities")
plt.xlabel("u (meters)")
plt.ylabel("v (meters)")
plt.title("UV Coverage Plot")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()

#Definimos las frecuencias de alma en banda 4
f_c = np.array(np.linspace(125/c,163/c,2))

#converimos a visiblidades sub lambda
ulambda = []
vlambda = []
wlambda = []
for uvw in uvw_coords:
    ulambda.append((uvw[0] * f_c))
    vlambda.append(uvw[1] * f_c)
    wlambda.append(uvw[2] * f_c)

ulambda = np.array(ulambda)
vlambda = np.array(vlambda)
wlambda = np.array(wlambda)
# Graficar uv coverage

plt.figure(figsize=(10, 10))
plt.scatter(ulambda, vlambda, color="blue", s=10, label="Visibilities")
plt.scatter([-u for u in ulambda], [-v for v in vlambda], color="red", s=10, label="Conjugate Visibilities")
plt.xlabel("u (λ)")
plt.ylabel("v (λ)")
plt.title("UV Coverage Plot (λ)")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()

#PARTE 1.1: FUENTE PUNTUAL

#definimos el l y m
l = np.linspace(-1,1,100)
m = np.linspace(-1,1,100)
S0= 1


# ejemplo de uso
shape = (100, 100)       # Size of the grid
center = (0, 0)          # Offset center for the Gaussian
sigma = 10               # Standard deviation

gaussian = gaussian_2d_movable(shape, center, sigma)

# graaficar para demostrar uso
plt.imshow(gaussian, origin='lower', extent=(-shape[1]//2, shape[1]//2, -shape[0]//2, shape[0]//2), cmap='viridis')
plt.colorbar()
plt.title("Movable 2D Circular Gaussian Distribution")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

print(len(ulambda.flatten()))

#calcualmos las visbilidades
#definimos el l, m y S0
l=0
m = 0
S0= 1

#definimos la visibilidad

V = (gaussian_2d_movable((len(ulambda.flatten()),len(ulambda.flatten())),(l,m),10) * (S0/(1-l**2-m**2))) * (np.e**(2*np.pi*(ulambda.flatten()*l + vlambda.flatten()*m)))

# Graficar visibilidad
plt.figure(figsize=(10, 10))
plt.scatter(ulambda.flatten(), vlambda.flatten(), c=np.abs(V), cmap="viridis", s=10)
plt.colorbar(label="Amplitude")
plt.xlabel("u (λ)")
plt.ylabel("v (λ)")
plt.title("Visibilities for a Point Source")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.show()

#PARTE2: Grilla de la UV coverage
"""
# Definir el tamaño del grid (por ejemplo, 256x256)
grid_size = 256

# Encontrar el máximo y mínimo en u y v para definir el rango del grid
max_u = max(np.abs(u_coords).max(), np.abs(v_coords).max())
grid_spacing = (2 * max_u) / grid_size  # espaciado basado en el tamaño del grid y rango de u, v

# Crear una rejilla vacía para almacenar las visibilidades
uv_grid = np.zeros((grid_size, grid_size), dtype=np.complex128)

# Asignar visibilidades a la rejilla
for u, v in zip(u_coords, v_coords):
    u_idx, v_idx = uv_to_grid_index(u, v, grid_size, grid_spacing)
    if 0 <= u_idx < grid_size and 0 <= v_idx < grid_size:
        uv_grid[u_idx, v_idx] += 1  # Se acumula la visibilidad (o el valor complejo si tuvieras datos de fase)

# Graficar la rejilla de cobertura uv
plt.imshow(np.abs(uv_grid), extent=(-max_u, max_u, -max_u, max_u), cmap="viridis")
plt.colorbar(label="Amplitude")
plt.xlabel("u (meters)")
plt.ylabel("v (meters)")
plt.title("Gridded UV Coverage")
plt.show()

#PARTE3: Primary Beam


el primary beam es la respuesta de un telescopio a una fuente puntual en el cielo, es decir, la respuesta del telescopio a una fuente puntual en el cielo.
en este caso sera cero ya que 
"""