from pathlib import Path
import pyproj
import shapely.geometry
from global_land_mask import globe

def save_grid(stepsize=1000, swlat=55.4355410101663, swlon=12.140848911388979, nelat=56.06417055142977, nelon=12.688363746232875):
    # Set up transformers, EPSG:3857 is metric, same as EPSG:900913
    to_proxy_transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857')
    to_original_transformer = pyproj.Transformer.from_crs('epsg:3857', 'epsg:4326')

    # Create corners of rectangle to be transformed to a grid
    sw = shapely.geometry.Point((swlon, swlat))
    ne = shapely.geometry.Point((nelon, nelat))

    # Project corners to target projection
    transformed_sw = to_proxy_transformer.transform(sw.x, sw.y) # Transform SW point to 3857
    transformed_ne = to_proxy_transformer.transform(ne.x, ne.y) # .. same for NE

    # Iterate over 2D area
    gridpoints = []
    x = transformed_sw[0]
    while x < transformed_ne[0]:
        y = transformed_sw[1]
        while y < transformed_ne[1]:
            p = shapely.geometry.Point(to_original_transformer.transform(x, y))
            if globe.is_land(p.y, p.x):
                gridpoints.append(p)
            y += stepsize
        x += stepsize

    with open(str(Path.cwd() / 'data' / 'interim' / 'area_grid.csv'), 'w') as of:
        of.write('GPS_Longitude;GPS_Latitude\n')
        for p in gridpoints:
            of.write('{:f};{:f}\n'.format(p.x, p.y))

if __name__ == "__main__":
    save_grid()