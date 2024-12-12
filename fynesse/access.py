import zipfile
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


def get_uk_data_from_osm(latitude, longitude, tags,  distance_km: float = 1.0):
    north = latitude + (0.02 / 2.2) * distance_km
    south = latitude - (0.02 / 2.2) * distance_km
    east = longitude + (1 / 64) * distance_km
    west = longitude - (1 / 64) * distance_km

    pois = ox.features.features_from_bbox([west, south, east, north], tags)

    return pois.dropna()


def get_buildings_data_from_osm(latitude, longitude):
    # Define latitude and longitude boundaries
    north = latitude + 0.02 / 2.2
    south = latitude - 0.02 / 2.2
    east = longitude + 1 / 64
    west = longitude - 1 / 64

    # Define tags for OSM data retrieval
    tags = {
        "addr": ["housenumber", "street", "postcode"],
        "building": True,
        "geometry": True
    }

    # Retrieve points of interest (POIs) based on bounding box
    pois = ox.features.features_from_bbox([west, south, east, north], tags)

    # Convert coordinate reference system to UTM for area calculation in meters
    pois['area_m2'] = pois.to_crs(epsg=32630)['geometry'].area

    # Check if POIs have full address information
    pois['has_full_address'] = pois[['addr:housenumber', 'addr:street', 'addr:postcode']].notnull().all(axis=1)

    # Filter only POIs with full addresses
    pois = pois[pois['has_full_address']]
    pois = pois[['addr:housenumber', 'addr:street', 'addr:postcode', 'area_m2']]

    return pois

def get_buildings_data_from_prices_paid_dataset(latitude, longitude, conn):
  north = latitude + 0.02/2.2
  south = latitude - 0.02/2.2

  east = longitude + 1/64
  west = longitude - 1/64

  cur = conn.cursor()
  cur.execute(f"SELECT pp.price, pp.date_of_transfer, pp.secondary_addressable_object_name, pp.primary_addressable_object_name, pp.street, pp.postcode, po.latitude, po.longitude FROM (SELECT * FROM pp_data WHERE date_of_transfer >= '2020-01-01') AS pp INNER JOIN postcode_data AS po ON po.postcode = pp.postcode WHERE po.latitude BETWEEN {south} AND {north} AND po.longitude BETWEEN {west} AND {east}")
  results = cur.fetchall()

  # Get column names from the cursor description
  columns = [desc[0] for desc in cur.description]


  return pd.DataFrame(results, columns=columns)

def download_census_data(code, base_dir=''):
  url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
  extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

  if os.path.exists(extract_dir) and os.listdir(extract_dir):
    print(f"Files already exist at: {extract_dir}.")
    return

  os.makedirs(extract_dir, exist_ok=True)
  response = requests.get(url)
  response.raise_for_status()

  with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(extract_dir)

  print(f"Files extracted to: {extract_dir}")

def load_census_data(code, level='msoa'):
  return pd.read_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv')

def upload_to_table(name, engine, df):
    try:
        with engine.connect() as conn:
            df.to_sql(name=name, con=conn, if_exists='replace', index=False)
            print("Data uploaded successfully!")
    except Exception as e:
        print(f"Error while uploading data: {e}")

def upload_table_in_chunks(engine, df, table_name, chunk_size=1000):
    """This is better for uploading large dataframes to tables"""
    try:
        with engine.connect() as conn:
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                chunk.to_sql(name=table_name, con=conn, if_exists='append', index=False)
                print(f"Chunk {i} to {i+chunk_size} uploaded successfully!")

        print(f"All data uploaded successfully to table '{table_name}'")

    except Exception as e:
        print(f"Error while uploading data: {e}")
        
def data_check(conn, table_name):
    cur = conn.cursor()
    cur.execute(f"""SELECT * FROM {table_name}""")
    results = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    return pd.DataFrame(results, columns=columns)
  