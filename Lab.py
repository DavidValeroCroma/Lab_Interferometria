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
    uv_coords = []
    for data in equatorial_xyz_data:
        # Convertir a coordenadas UV
        u = data['x']*math.sin(H0) + data['y']*math.cos(H0)
        v = -data['x']*math.sin(delta)*math.cos(H0) + data['y']*math.sin(delta)*math.sin(H0) + data['z']*math.cos(delta)
        uv_coords.append((u, v))
    return uv_coords

#PARTE1: uv coverage

#----------------Convertir Coordenadas
# Leer archivo de configuración de antenas
antenas = read_antenna_data('C:/Users/david/OneDrive/Desktop/Universidad/Semestre 12/Interferometria/LAB/Lab_Interferometria/alma.cycle8.6.cfg')
print(antenas)
# Convertir coordenadas ENU a Azimuth y Altitud
az_el_data = enu_to_altaz(antenas)
print(az_el_data)
# Definir la ubicación de ALMA
Alma = EarthLocation.of_site('ALMA')
print(Alma)

# Convertir Azimuth y Altitud a coordenadas Ecuatoriales y XYZ
ec_xyz_data = az_el_to_equatorial_xyz(az_el_data, Alma.lat.deg)
print(ec_xyz_data)

# Convertir coordenadas ecuatoriales a coordenadas UV
HA= np.linspace(0, 2*np.pi, 100)
uv_coords = []
for i in range(len(HA)):
    uv_coords.extend(xyz_to_uv(ec_xyz_data, HA[i], 1))

print(uv_coords)
# Graficar uv coverage
uv_coords = np.array(uv_coords)
plt.figure(figsize=(10, 10))
plt.scatter(uv_coords[:, 0], uv_coords[:, 1], color="blue", s=10, label="Visibilities")
plt.scatter([-u for u in uv_coords[:, 0]], [-v for v in uv_coords[:, 1]], color="red", s=10, label="Conjugate Visibilities")
plt.xlabel("u (meters)")
plt.ylabel("v (meters)")
plt.title("UV Coverage Plot")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()

#PARTE2: Grilla de la UV coverage

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

"""
el primary beam es la respuesta de un telescopio a una fuente puntual en el cielo, es decir, la respuesta del telescopio a una fuente puntual en el cielo.
en este caso sera cero ya que 
"""