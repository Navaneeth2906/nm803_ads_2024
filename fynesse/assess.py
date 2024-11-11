from .config import *

from . import access

import osmnx as ox

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


def plot_buildings_in_area(place_name, latitude, longitude):
  up_lat = latitude + 0.02/2.2
  lo_lat = latitude - 0.02/2.2

  up_long = longitude + 0.02/2.2
  lo_long = longitude - 0.02/2.2

  tags = {
      "addr": ["housenumber", "street", "postcode"],
      "building": True,
      "geometry": True
  }
  pois = ox.geometries_from_bbox(up_lat, lo_lat, up_long, lo_long, tags)

  # we need to convert the coordinate reference system to meters
  pois['area_m2'] = pois.to_crs(epsg=32630)['geometry'].area


  pois['has_full_address'] = pois[['addr:housenumber', 'addr:street', 'addr:postcode']].notnull().all(axis=1)

  # retrieve graph
  graph = ox.graph_from_bbox(up_lat, lo_lat, up_long, lo_long)

  # Retrieve nodes and edges
  nodes, edges = ox.graph_to_gdfs(graph)

  # Get place boundary related to the place name as a geodataframe
  area = ox.geocode_to_gdf(place_name)

  fig, ax = plt.subplots()

  # Plot the footprint
  area.plot(ax=ax, facecolor="white")

  # Plot street edges
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

  ax.set_xlim([lo_long, up_long])
  ax.set_ylim([lo_lat, up_lat])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")

  # Plot all POIs

  pois[~pois['has_full_address']].plot(ax=ax, color="red", label="Without Address", alpha=0.7, edgecolor="black", linewidth=0.5)
  pois[pois['has_full_address']].plot(ax=ax, color="blue", label="With Address", alpha=0.7, edgecolor="black", linewidth=0.5)
  plt.tight_layout()