import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

def normalize_geometry(geometry):
    """
    Normalize a Shapely Polygon or MultiPolygon by centering and scaling it.
    
    Parameters:
    geometry (shapely.geometry.Polygon or shapely.geometry.MultiPolygon): The geometry to normalize.
    
    Returns:
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon: The normalized geometry.
    """
    if isinstance(geometry, Polygon):
        return normalize_polygon(geometry)
    elif isinstance(geometry, MultiPolygon):
        # Normalize each polygon in the MultiPolygon
        normalized_polygons = [normalize_polygon(p) for p in geometry]
        return MultiPolygon(normalized_polygons)

def overlapping_area_polygon(poly1, poly2):
    """
    Calculate the overlapping area between two geometries (Polygons or MultiPolygons).
    
    Parameters:
    poly1 (shapely.geometry.Polygon or shapely.geometry.MultiPolygon): The first geometry.
    poly2 (shapely.geometry.Polygon or shapely.geometry.MultiPolygon): The second geometry.
    
    Returns:
    float: The area of overlap between the two geometries.
    """
    return poly1.intersection(poly2).area

def top_n_similar_polygons(target_polygon, polygon_list, index_list, n=10):
    """
    Find the top n polygons or multipolygons with the largest overlapping area with the target polygon.
    
    Parameters:
    target_polygon (shapely.geometry.Polygon or shapely.geometry.MultiPolygon): The target geometry.
    polygon_list (list): A list of geometries (Polygon or MultiPolygon) to compare against the target.
    index_list (list): A list of indices corresponding to the geometries.
    n (int): The number of top results to return (default is 10).
    
    Returns:
    list: A list of tuples containing the geometry, its index, and the overlapping area with the target geometry.
    """
    # Normalize the target geometry (Polygon or MultiPolygon)
    target_polygon_normalized = normalize_geometry(target_polygon)
    
    # Calculate the overlapping area for each geometry in the list
    areas = [
        (polygon, index, overlapping_area_polygon(target_polygon_normalized, normalize_geometry(polygon))) 
        for polygon, index in zip(polygon_list, index_list)
    ]
    
    # Sort the polygons or multipolygons by overlapping area in descending order
    areas_sorted = sorted(areas, key=lambda x: x[2], reverse=True)
    
    # Return the top n results
    return areas_sorted[:n]


from shapely.geometry import Polygon
import numpy as np

def normalize_polygon(polygon):
    """
    Normalize and center a Shapely Polygon object around (0, 0).
    
    Parameters:
    polygon (shapely.geometry.Polygon): The input Polygon object.
    
    Returns:
    shapely.geometry.Polygon: The normalized and centered Polygon object.
    """
    # Extract the exterior coordinates of the polygon as a numpy array
    coords = np.array(polygon.exterior.coords)
    
    # Compute the centroid of the polygon (mean of the vertices)
    centroid = np.mean(coords, axis=0)
    
    # Shift the polygon to be centered at (0, 0)
    centered_coords = coords - centroid
    
    # Compute the maximum distance from the origin
    max_distance = np.max(np.linalg.norm(centered_coords, axis=1))
    
    # Normalize the polygon so that the farthest point is at distance 1
    normalized_coords = centered_coords / max_distance
    
    # Create a new Polygon object from the normalized coordinates
    normalized_polygon = Polygon(normalized_coords)
    
    return normalized_polygon
