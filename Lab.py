import numpy as np
from astropy.constants import c
from astropy.coordinates import EarthLocation, AltAz, ITRS
from astropy import units as u
import matplotlib.pyplot as plt

"""
Función: función para convertir u, v a índices de la rejilla
Entrada: Coordenadas u, v, tamaño de la rejilla, espaciado de la rejilla.
Salida: Índices de la rejilla para u, v.
"""
def uv_to_grid_index(u, v, grid_size, grid_spacing):
    u_idx = int((u + max_u) / grid_spacing)  # Desplazar u y v para que estén centrados
    v_idx = int((v + max_u) / grid_spacing)
    return u_idx, v_idx

"""
Funcion: Lee el archivo de configuración de antenas y devuelve una lista de diccionarios con coordenadas.
Entrada: Ruta al archivo de configuración de antenas.
Salida: Lista de diccionarios con las coordenadas 'x', 'y', y 'z' de cada antena.
"""
def read_antenna_data(file_path):

    antennas = []

    with open(file_path, 'r') as file:
        for line in file:
            # Ignorar líneas de comentarios
            if line.startswith('#'):
                continue

            # Parsear las columnas: UTM-X, UTM-Y, Z
            columns = line.split()
            if len(columns) >= 3:
                x, y, z = map(float, columns[:3])  # Obtener las tres primeras columnas como float
                antennas.append({'x': x, 'y': y, 'z': z})

    return antennas
"""
Funcion: Calcula las coordenadas uvw de un arreglo de antenas respecto a una fuente astronómica.

Entrada: - Lista de antenas, donde cada antena es un diccionario con:
                     'x': coordenada UTM-X de la antena,
                     'y': coordenada UTM-Y de la antena,
                     'z': coordenada en altura de la antena (metros).
    - Ascensión recta de la fuente (en radianes).
    - Declinación de la fuente (en radianes).
Salida: Lista de coordenadas uvw para cada par de antenas.
"""
def calculate_uvw(antennas, source_ra, source_dec):

    uvw_coords = []

    # Conversión de ascensión recta y declinación a seno y coseno
    sin_ra, cos_ra = np.sin(source_ra), np.cos(source_ra)
    sin_dec, cos_dec = np.sin(source_dec), np.cos(source_dec)

    # Transformación de las coordenadas UTM de cada antena
    for i, ant1 in enumerate(antennas):
        for j, ant2 in enumerate(antennas):
            if i < j:
                dx = ant2['x'] - ant1['x']
                dy = ant2['y'] - ant1['y']
                dz = ant2['z'] - ant1['z']

                # Calculando u, v, w
                u = dx * sin_ra - dy * cos_ra
                v = dx * sin_dec * cos_ra + dy * sin_dec * sin_ra - dz * cos_dec
                w = dx * cos_dec * cos_ra + dy * cos_dec * sin_ra + dz * sin_dec

                uvw_coords.append((u, v, w))

    return uvw_coords

#----------Parte 1 UV Coverage----------------
# leemos el archivo de configuración de antenas (UTM XYZ)
file_path = 'C:/Users/david/OneDrive/Desktop/Universidad/Semestre 12/Interferometria/LAB/Lab_Interferometria/alma.out02.cfg'  # Ruta al archivo
antennas = read_antenna_data(file_path)

# Definir la fuente (ejemplo con RA=0.5 y Dec=0.3 en radianes)
source_ra = 0.5
source_dec = 0.3
uvw_coords = calculate_uvw(antennas, source_ra, source_dec)

# Imprimir las coordenadas uvw
# Separamos las coordenadas u y v para graficar
u_coords = [uvw[0] for uvw in uvw_coords]
v_coords = [uvw[1] for uvw in uvw_coords]

# graficamos el uv-coverage

plt.figure(figsize=(10, 10))
plt.scatter(u_coords, v_coords, color="blue", s=10, label="Visibilities")
plt.scatter([-u for u in u_coords], [-v for v in v_coords], color="red", s=10, label="Conjugate Visibilities")
plt.xlabel("u (meters)")
plt.ylabel("v (meters)")
plt.title("UV Coverage Plot")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()

#----------Parte 2 Gridding----------------

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