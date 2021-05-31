#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import requests
import urllib.request
import json
from io import StringIO
import plotly.express as px
import plotly.io as pio
from PIL import Image
pio.renderers.default = 'browser'


# In[3]:


s_code = pd.read_csv(r'C:/CHINMAY/Documents/s_code.csv')
gsheetid_1 = "1noqoXm0pnb61miW0HnaCCNsKPm_4lwhbzmHkREuF0ZY"
sheet_name_1 = "Vaccine"
gsheet_url_1 = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(gsheetid_1,sheet_name_1)
vacc = pd.read_csv(gsheet_url_1)
vacc = vacc[vacc["State"]!="India"]
vacc = vacc.groupby('State').last().reset_index()
vacc = vacc.drop(columns=['Updated On'])
vacc = vacc.drop(index=[33],axis=0)
vacc.replace(',','',regex=True,inplace=True)


# In[4]:

vacc["id"] = s_code["id"]


# In[5]:


states = json.load(open("states_india.geojson", "r"))
state_id_map = {}
for feature in states["features"]:
    feature["id"] = feature["properties"]["state_code"]
    state_id_map[feature["properties"]["st_nm"]] =  feature["id"]


# # Streamlit

# In[27]:


st.markdown("# COVID-19 Vaccination Detailed Analysis (India)") 
img = Image.open('C:/CHINMAY/Documents/COVID-19-vaccine.png')
if img.mode != 'RGB':
    img = img.convert('RGB')
st.image(img, caption = "Source: bioworld.com", width=500)
if st.checkbox('view_data'):
    st.subheader('Vaccination Data')
    st.write(vacc)


# In[28]:


st.sidebar.markdown("## Side Panel")
st.sidebar.markdown("Use this panel to explore our app")

st.sidebar.subheader('Visualizations')
if st.sidebar.checkbox('Individuals Vaccinated'):
    st.subheader('Number of Individuals Vaccinated by State')
    fig = px.bar(vacc, x="State", y="Total Individuals Vaccinated", height=800, width = 800)
    st.plotly_chart(fig)

if st.sidebar.checkbox('Covid-19 Vaccines in India'):
    st.subheader('Vaccines Administered by State')
    fig = px.bar(vacc, x="State", y=["Total Covaxin Administered","Total CoviShield Administered"], 
             barmode='group', height=800, width = 800)
    st.plotly_chart(fig)
    
if st.sidebar.checkbox('Gender'):
    st.subheader('Vaccines Administered by State and Gender')
    fig = px.bar(vacc, x="State", y=["Male(Individuals Vaccinated)","Female(Individuals Vaccinated)",
                                     "Transgender(Individuals Vaccinated)"], 
             barmode='group', height=800, width = 800)
    st.plotly_chart(fig)
    
if st.sidebar.checkbox('First Dose and Second Dose'):
    st.subheader('First and Second Doses Administered by State')
    fig = px.bar(vacc, x="State", y=['First Dose Administered','Second Dose Administered'], 
             barmode='group', height=800, width = 800)
    st.plotly_chart(fig)
    
if st.sidebar.checkbox('Age Group'):
    st.subheader('Age Group vaccinations by state')
    fig = px.bar(vacc, x="State", y=['18-45 years (Age)', '45-60 years (Age)', '60+ years (Age)'], 
             barmode='group', height=800, width = 800)
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig)
    
if st.sidebar.checkbox('Map'):
    st.subheader('Covid19 Vaccine Map')
    fig = px.choropleth_mapbox(
    vacc,
    locations="id",
    geojson = states,
    color="Total Individuals Vaccinated",
    hover_name="State",
    hover_data=['Total Individuals Vaccinated', 'Total Sessions Conducted',
       'Total Sites', 'First Dose Administered', 'Second Dose Administered',
       'Male(Individuals Vaccinated)', 'Female(Individuals Vaccinated)',
       'Transgender(Individuals Vaccinated)', 'Total Covaxin Administered',
       'Total CoviShield Administered'],
    title="India Covid-19 Vaccine Map",
    mapbox_style="carto-positron",
    center={"lat": 24, "lon": 78},
    zoom=3,
    opacity=0.5,
    )
    st.plotly_chart(fig)
    
