from .config import *

from . import access

import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd 
import geopandas as gpd
import statsmodels.api as sm
import numpy as np
import seaborn as sns



"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def add_tag_flags(df, tags):
    df = df.copy()
    for tag in tags:
        column_name = tag.replace("'", '').replace(' ', '') 
        df.loc[:, column_name] = df["tags"].apply(lambda x: tag in x)
    return df


def plot_buildings_in_area(place_name, latitude, longitude):
  north = latitude + 0.02/2.2
  south = latitude - 0.02/2.2

  east = longitude + 1/64
  west = longitude - 1/64

  tags = {
      "addr": ["housenumber", "street", "postcode"],
      "building": True,
      "geometry": True
  }
  pois = ox.features_from_bbox([west, south, east, north], tags)


  # we need to convert the coordinate reference system to meters
  pois['area_m2'] = pois.to_crs(epsg=32630)['geometry'].area


  pois['has_full_address'] = pois[['addr:housenumber', 'addr:street', 'addr:postcode']].notnull().all(axis=1)

  # retrieve graph
  graph = ox.graph_from_bbox([west, south, east, north])

  # Retrieve nodes and edges
  nodes, edges = ox.graph_to_gdfs(graph)

  # Get place boundary related to the place name as a geodataframe
  area = ox.geocode_to_gdf(place_name)

  fig, ax = plt.subplots()

  # Plot the footprint
  area.plot(ax=ax, facecolor="white")

  # Plot street edges
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

  ax.set_xlim([west, east])
  ax.set_ylim([south, north])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")

  # Plot all POIs

  pois[~pois['has_full_address']].plot(ax=ax, color="red", label="Without Address", alpha=0.7, edgecolor="black", linewidth=0.5)
  pois[pois['has_full_address']].plot(ax=ax, color="blue", label="With Address", alpha=0.7, edgecolor="black", linewidth=0.5)
  plt.tight_layout()


def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:

    degrees_lat = (0.02/2.2) * distance_km  
    degrees_long = (1/64) * distance_km  
    north = latitude + degrees_lat
    south = latitude - degrees_lat
    west = longitude - degrees_long
    east = longitude + degrees_long
    pois = ox.features_from_bbox([west, south, east, north], tags)

    pois_df = pd.DataFrame(pois)

    poi_counts = {}

    for tag in tags.keys():
      if tag in pois_df.columns:
        poi_counts[tag] = pois_df[tag].notnull().sum()
      else:
        poi_counts[tag] = 0

    return poi_counts


def calculate_tag_counts(osm_tags_gdf, oa_gdf, flags_to_count):

    # perform a spatial join to match points to polygons
    joined_gdf = gpd.sjoin(osm_tags_gdf, oa_gdf, how='inner', predicate='within')
    
    # create a new dataframe with counts of each flag grouped by polygons
    counts = joined_gdf.groupby('index_right')[flags_to_count].sum()
    
    # merge the counts back into the polygons GeoDataFrame
    oa_tag_counts_gdf = oa_gdf.join(counts, how='left')
    
    oa_tag_counts_gdf[flags_to_count] = oa_tag_counts_gdf[flags_to_count].fillna(0)
    
    return oa_tag_counts_gdf


def augment_with_nearest_distance(output_gdf, osm_gdf, tag_column, new_column_name, crs="EPSG:27700"):

    # filter based on the specified tag and value
    filtered_osm = osm_gdf[osm_gdf[tag_column] == True]

    # ensure both gdf are in the same CRS for distance calculation
    filtered_osm = filtered_osm.to_crs(crs)
    output_gdf = output_gdf.to_crs(crs)

    # define a helper function to calculate the nearest distance
    def calculate_nearest_distance(output_area, points):
        return points.geometry.distance(output_area.geometry).min() / 1000  # Convert to km

    # add the new column for the nearest distance
    output_gdf[new_column_name] = output_gdf.apply(
        lambda row: calculate_nearest_distance(row, filtered_osm), axis=1
    )

    # revert the output gdf to its original CRS
    output_gdf = output_gdf.to_crs("EPSG:4326")

    return output_gdf

def plot_and_calculate_correlation(x_column, y_column, oa_sec_tag_counts_dists_gdf):

    # create the scatter plot
    plt.scatter(oa_sec_tag_counts_dists_gdf[x_column], oa_sec_tag_counts_dists_gdf[y_column])
    plt.title(f'Scatter Plot of {y_column} vs {x_column}')
    plt.ylabel(f'{y_column}')
    plt.xlabel(f'{x_column}')
    plt.show()

    # calculate and return the correlation
    correlation = oa_sec_tag_counts_dists_gdf[x_column].corr(oa_sec_tag_counts_dists_gdf[y_column])
    print(f"Correlation between {x_column} and {y_column}: {correlation}")
    return correlation


def plot_correlation_excluding_zeros(x_column, y_column, oa_tag_counts_dists_gdf):

    # filter out rows where x_column is zero
    no_zeros = oa_tag_counts_dists_gdf[oa_tag_counts_dists_gdf[x_column] != 0]

    # create the scatter plot
    plt.scatter(no_zeros[x_column], no_zeros[y_column])
    plt.title(f'Scatter Plot of {y_column} vs {x_column} (Excluding Zeros)')
    plt.ylabel(f'{y_column} in OA')
    plt.xlabel(f'Number of {x_column} in OA')
    plt.show()

    # calculate and return the correlation
    correlation = no_zeros[x_column].corr(no_zeros[y_column])
    print(f"Correlation between {x_column} and {y_column} (Excluding Zeros): {correlation}")
    return correlation


def plot_ols_regression(oa_tag_counts_dists_gdf, independent_var, dependant_var):

    # set X (independent variable) and y (dependent variable divided by total)
    X = oa_tag_counts_dists_gdf[independent_var]
    y = oa_tag_counts_dists_gdf[dependant_var]
    
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X)
    results = model.fit()
    y_pred = results.predict(X)
    
    # plot
    plt.scatter(oa_tag_counts_dists_gdf[independent_var], y, label='Data Points', color='blue')
    plt.plot(oa_tag_counts_dists_gdf[independent_var], y_pred, color='red', label='Regression Line')
    plt.xlabel(f"Number of {independent_var} tags in the area")
    plt.ylabel(f"{dependant_var} in OA")
    plt.title(f"OLS Regression: {independent_var} vs {dependant_var}")
    plt.legend()
    plt.show()
    
    print(results.summary())
    
    
def overfitting_check(y, X, alpha=None, L1_wt=None):

  k_values = range(2, 11)
  mean_train_r2 = []
  mean_test_r2 = []
  mean_train_corr = []
  mean_test_corr = []

  for k in k_values:
      kf = KFold(n_splits=k, shuffle=True, random_state=42)
      train_r2_scores = []
      test_r2_scores = []
      train_correlations = []
      test_correlations = []

      for train_index, test_index in kf.split(X):

          train_places = X.index[train_index]
          test_places = X.index[test_index]

          X_train, X_test = X.loc[train_places], X.loc[test_places]
          y_train, y_test = y.loc[train_places], y.loc[test_places]

          X_train_with_const = sm.add_constant(X_train)
          X_test_with_const = sm.add_constant(X_test)

          if alpha is not None and L1_wt is not None:
            model = sm.OLS(y_train, X_train_with_const).fit_regularized(alpha=alpha, L1_wt=L1_wt)
          else:
            model = sm.OLS(y_train, X_train_with_const).fit()

          
          y_train_pred = np.array(model.predict(X_train_with_const))
          y_test_pred = np.array(model.predict(X_test_with_const))

          ss_total_train = np.sum((y_train - np.mean(y_train)) ** 2)
          ss_residual_train = np.sum((y_train - y_train_pred) ** 2)
          r2_train = 1 - (ss_residual_train / ss_total_train)

          ss_total_test = np.sum((y_test - np.mean(y_test)) ** 2)
          ss_residual_test = np.sum((y_test - y_test_pred) ** 2)
          r2_test = 1 - (ss_residual_test / ss_total_test)

          train_r2_scores.append(r2_train)
          test_r2_scores.append(r2_test)

          train_correlation = np.corrcoef(y_train, y_train_pred)[0, 1]
          test_correlation = np.corrcoef(y_test, y_test_pred)[0, 1]

          train_correlations.append(train_correlation)
          test_correlations.append(test_correlation)

      mean_train_r2.append(np.mean(train_r2_scores))
      mean_test_r2.append(np.mean(test_r2_scores))
      mean_train_corr.append(np.mean(train_correlations))
      mean_test_corr.append(np.mean(test_correlations))

  plt.figure(figsize=(12, 6))

  # Plot R²
  plt.subplot(1, 2, 1)
  plt.plot(k_values, mean_train_r2, marker='o', label='Train R²', color='blue')
  plt.plot(k_values, mean_test_r2, marker='o', label='Test R²', color='orange')
  plt.title("Effect of k on R²")
  plt.xlabel("Number of Folds (k)")
  plt.ylabel("Mean R²")
  plt.grid()
  plt.legend()

  # Plot Correlation
  plt.subplot(1, 2, 2)
  plt.plot(k_values, mean_train_corr, marker='o', label='Train Correlation', color='blue')
  plt.plot(k_values, mean_test_corr, marker='o', label='Test Correlation', color='orange')
  plt.title("Effect of k on Correlation")
  plt.xlabel("Number of Folds (k)")
  plt.ylabel("Mean Correlation")
  plt.grid()
  plt.legend()

  plt.tight_layout()
  plt.show()
  

def draw_feature_correlations(df, features):

    correlation_matrix = df[features].corr()

    plt.figure(figsize=(12, 10))  
    sns.heatmap(
        correlation_matrix, 
        annot=True,         
        fmt=".2f",           
        cmap="coolwarm",     
        linewidths=0.5       
    )

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.title("Correlation Matrix of Features", fontsize=14)

    plt.tight_layout()
    plt.show()


def calculate_brute_force_distances(msoa_df, ways):

    distances = []

    for idx, geom in enumerate(msoa_df.geometry):
        min_distance = ways.geometry.apply(lambda x: geom.distance(x)).min()
        distances.append(min_distance)
        
        # Logs
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(msoa_df)} geometries...")

    msoa_df = msoa_df.copy()  
    msoa_df['nearest_distance'] = distances

    return msoa_df