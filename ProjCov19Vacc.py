#!/usr/bin/env python
# coding: utf-8

# In[31]:


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
import pickle
from PIL import Image
pio.renderers.default = 'browser'


# # Data 

# In[32]:

s_code = [35,28,12,18,10,4,22,26,7,30,24,6,2,1,20,29,32,37,31,23,27,14,17,15,13,21,34,3,8,11,33,0,16,9,5,19]
gsheetid_1 = "1noqoXm0pnb61miW0HnaCCNsKPm_4lwhbzmHkREuF0ZY"
sheet_name_1 = "Vaccine"
gsheet_url_1 = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(gsheetid_1,sheet_name_1)
vacc = pd.read_csv(gsheet_url_1)
vacc = vacc[vacc["State"]!="India"]


# In[33]:


vacc = vacc.groupby('State').last().reset_index()
vacc = vacc.drop(columns=['Updated On'])
vacc.replace(',','',regex=True,inplace=True)
vacc["id"] = s_code


# In[35]:


states = json.load(open("states_india.geojson", "r"))
state_id_map = {}
for feature in states["features"]:
    feature["id"] = feature["properties"]["state_code"]
    state_id_map[feature["properties"]["st_nm"]] = feature["id"]


# # Model

# In[40]:


new_data= vacc.drop(['State','First Dose Administered', 'Second Dose Administered',
       'Male(Individuals Vaccinated)', 'Female(Individuals Vaccinated)',
       'Transgender(Individuals Vaccinated)', 'Total Covaxin Administered',
       'Total CoviShield Administered', 'AEFI', '18-45 years (Age)', '45-60 years (Age)', '60+ years (Age)',
       'Total Doses Administered','Total Sputnik V Administered',
       'Total Doses Administered','id'],axis=1)


# In[41]:


X = new_data.drop(['Total Individuals Vaccinated'],axis=1)
y = new_data['Total Individuals Vaccinated']


# In[42]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.30, random_state=2)


# In[43]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
model = LR.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[47]:

pickle.dump(model,open('vaccine.pkl','wb'))


# # Streamlit 

# In[20]:


st.markdown("# COVID-19 Vaccination Detailed Analysis (India)") 
img = "https://resize.indiatvnews.com/en/resize/newbucket/715_-/2021/05/covidvaccinetechnology-1620269459.jpg"
st.image(img, caption = "Source: bioworld.com", width=250)
if st.checkbox('view_data'):
    st.subheader('Vaccination Data')
    st.write(vacc)


# In[21]:


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


# In[22]:


model=pickle.load(open('vaccine.pkl','rb'))
st.sidebar.header("Model")
if st.sidebar.checkbox("Predict"):
    def predict(sites_no,sessions_no):
        input=np.array([[sites_no,sessions_no]]).astype(np.float64)
        prediction=model.predict(input)
        return float(prediction)

    def main():
        st.title("Predictor")
        html_temp = """
        <div style="background-color:blue ;padding:10px">
        <h2 style="color:white;text-align:center;">Covid19 ML App </h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        
        sites_no = st.text_input("Number of Sessions","Type Here")
        sessions_no = st.text_input("Number of Sites","Type Here")

        if st.button("Predict"):
            output = predict(sites_no,sessions_no)
            st.success('Total Number of Individuals Vaccinated is {}'.format(output))

        fig = px.choropleth_mapbox(vacc,locations="id",geojson = states,
                                   color="Total Individuals Vaccinated",hover_name="State",
                                   hover_data = ['Total Individuals Vaccinated', 'Total Sessions Conducted',
                                                   'Total Sites'],
                                   mapbox_style="carto-positron",center={"lat": 24, "lon": 78},
                                   zoom=3,opacity=0.5,
                                   )
        st.plotly_chart(fig)

    if __name__=='__main__':
        main()

