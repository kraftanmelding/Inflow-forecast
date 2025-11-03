#import libararies
import PyPDF2 as pypdf
from difflib import SequenceMatcher
import re
from glob import glob
import pandas as pd
import os
import time
from pyproj import Transformer
start = time.time()

from pdf_csv import make_df 
from weights_coords import compute_coords_weights
from add_meps import update_timeseries

basin_id = 'ytre_alsaaker'#'Holmen'

#read files from NEVINA
pdf_path = f'pdfs/Nedbørfeltparam-{basin_id}.pdf'
shp_path = f'shps/{basin_id}/NedbfeltF_v4.shp'#holmen v3

#path to attributes file (all plants/catchments)
attr = pd.read_csv('Data/attributes/attributes.csv')
attr = attr[attr['basin_id'] != basin_id.lower()]


#ADD STATIC ATTRIBUTES FROM PDF
df = make_df([pdf_path], pattern_prefix="Nedbørfeltparam-")

keep = ['basin_id', 'latitude', 'longitude',
        'length_km_river', 'perc_agricul','perc_bog', 'perc_eff_lake', 'perc_lake',
       'perc_glacier', 'perc_forest', 'perc_mountain', 'perc_urban',
       'area_total','height_minimum', 'height_maximum',
       'height_hypso_10', 'height_hypso_20', 'height_hypso_30',
       'height_hypso_40', 'height_hypso_50', 'height_hypso_60',
       'height_hypso_70', 'height_hypso_80', 'height_hypso_90',
       'specific_runoff', 'annual_runoff','is_sk']


#transform UTM to lat/lon
transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)

utm_east = df.iloc[0]['utm_east_z33']      
utm_north = df.iloc[0]['utm_north_z33']    

lon, lat = transformer.transform(utm_east, utm_north)
df['latitude'] = lat
df['longitude'] = lon

df['is_sk'] = [1]

df = df[keep]

updated_attr = pd.concat([df, attr], ignore_index=True) 
updated_attr = updated_attr.drop('Unnamed: 0', axis=1, errors='ignore')
updated_attr.to_csv('Data/attributes/attributes.csv', index=False)

print('Attributes updated')


#GET COORDINATES AND WEIGHTS FOR METEOROLOGICAL FORCINGS
df2 = compute_coords_weights(shapefile_path=shp_path)
df2.to_csv(f'coords_weights/{basin_id}_coords_weights.csv', index=False)

print('Coordinates and weights saved')


#(GET DYNAMIC INPUT FROM NORA3)


#UPDATE HISTORICAL DYNAMIC INPUT WITH MEPS DATA
update_timeseries(
    nora_nc_path=f"Data/time_series/{basin_id.lower()}.nc",
    weights_csv_path=f"coords_weights/{basin_id}_coords_weights.csv",
    out_nc_path=f"Data/time_series/{basin_id.lower()}.nc",  # overwrite
    start_date="2024-10-01",
    #end_date=None -> yesterday
)

#IF HISTORICAL INFLOW DATA IS AVAILABLE: INCLUDE IN TRAINING OR FINE TUNE MODEL ...

end = time.time()
print(f'Process time: {round((end-start)/3600,1)}h')
