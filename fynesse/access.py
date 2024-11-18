from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

import requests

import pymysql

import csv

import pandas as pd

import osmnx as ox


# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

def hello_world():
  print("Hello from the data science library! from Nav")


def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored 
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def housing_upload_join_data(conn, year):
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  conn.commit()
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  conn.commit()
  print('Data stored for year: ' + str(year))

def get_buildings_data_from_osm(latitude, longitude):
    # Define latitude and longitude boundaries
    up_lat = latitude + 0.02 / 2.2
    lo_lat = latitude - 0.02 / 2.2
    up_long = longitude + 1 / 64
    lo_long = longitude - 1 / 64

    # Define tags for OSM data retrieval
    tags = {
        "addr": ["housenumber", "street", "postcode"],
        "building": True,
        "geometry": True
    }

    # Retrieve points of interest (POIs) based on bounding box
    pois = ox.geometries_from_bbox(up_lat, lo_lat, up_long, lo_long, tags)

    # Convert coordinate reference system to UTM for area calculation in meters
    pois['area_m2'] = pois.to_crs(epsg=32630)['geometry'].area

    # Check if POIs have full address information
    pois['has_full_address'] = pois[['addr:housenumber', 'addr:street', 'addr:postcode']].notnull().all(axis=1)

    # Filter only POIs with full addresses
    pois = pois[pois['has_full_address']]
    pois = pois[['addr:housenumber', 'addr:street', 'addr:postcode', 'area_m2']]

    return pois

def get_buildings_data_from_prices_paid_dataset(latitude, longitude, conn):
  up_lat = latitude + 0.02/2.2
  lo_lat = latitude - 0.02/2.2

  up_long = longitude + 1/64
  lo_long = longitude - 1/64

  cur = conn.cursor()
  cur.execute(f"SELECT pp.price, pp.date_of_transfer, pp.secondary_addressable_object_name, pp.primary_addressable_object_name, pp.street, pp.postcode, po.latitude, po.longitude FROM (SELECT * FROM pp_data WHERE date_of_transfer >= '2020-01-01') AS pp INNER JOIN postcode_data AS po ON po.postcode = pp.postcode WHERE po.latitude BETWEEN {lo_lat} AND {up_lat} AND po.longitude BETWEEN {lo_long} AND {up_long}")
  results = cur.fetchall()

  # Get column names from the cursor description
  columns = [desc[0] for desc in cur.description]


  return pd.DataFrame(results, columns=columns)