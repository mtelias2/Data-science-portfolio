"""
Created for the Data incubator
"""
import itertools
import numpy as np
from scipy import stats
import math
import sys
import csv
import collections
import pandas as pd
import folium as f


def DataLoader():
    """
    This section loads data from the Zillow downloaded files.
    Data is also cleaned
    """
    #importing data
    House_Prices_Uncleaned = pd.read_csv("zillow_data/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_mon.csv")
    #Cleaning house prices data

    House_Prices=pd.DataFrame(House_Prices_Uncleaned["RegionName"][House_Prices_Uncleaned["CountyName"]=="New York County"])

    House_Prices["Price"]=pd.DataFrame(House_Prices_Uncleaned["2020-09-30"])

    House_Rent_Uncleaned=  pd.read_csv("zillow_data/Zip_ZORI_AllHomesPlusMultifamily_SSA.csv")

    #Cleaning house rent data
    House_Rent=pd.DataFrame(House_Rent_Uncleaned["RegionName"])
    House_Rent["Rent"]=pd.DataFrame(House_Rent_Uncleaned["2020-09"])

    return House_Prices, House_Rent


def main():
    """
    Main function
    """


    House_Prices,House_Rent= DataLoader()
    Coloum_Names=["ZIP","Price To Rent"]
    Price_To_Rent=pd.DataFrame(columns=Coloum_Names)

    for row in House_Prices.index:
        zip=House_Prices["RegionName"][row]
        if zip in House_Rent.RegionName.values:
            rent_row=int(House_Rent[House_Rent["RegionName"]==zip].index.values)
            rent=House_Rent["Rent"][rent_row]
            if rent > 0:
                #print(rent)
                PtR_temp=float(House_Prices["Price"][row]/(12*rent))
                tempDF=pd.DataFrame({"ZIP":str(zip),"Price To Rent":PtR_temp},index=[0])
                Price_To_Rent=Price_To_Rent.append(tempDF,ignore_index=True)

    print(Price_To_Rent.info())

    #This next part is for printing on an interactive map
    zipcodes=Price_To_Rent["ZIP"]
    map = f.Map(location=[40.693943, -73.985880], default_zoom_start=18)
    f.Marker(location=[40.693943, -73.985880],
    popup='Welcome to <b>NEW YORK CITY</b>').add_to(map)

    map.choropleth(geo_data="zillow_data/nyc-zip-code-tabulation-areas-polygons.geojson",data=Price_To_Rent, #
             columns=['ZIP', 'Price To Rent'],
             key_on='feature.properties.postalCode',
             fill_color='BuPu', fill_opacity=0.7, line_opacity=0.2,
             legend_name='Price equivelant years of rent')

    map.save("Preview_Rent_To_Price.html")



if __name__ == '__main__':
    main()
