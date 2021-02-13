'''
!/usr/bin/env python
-*-coding:utf-8-*-
@author:ayanava_dutta,rohan_sasmal,shivam_gupta
'''
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
import plotly.express as px
import streamlit as st
import calendar
from plotly.graph_objs import *
import datetime
from datetime import timedelta
from collections import OrderedDict




def dash():
    st.title("RP DashBoard")
    
    df=pd.read_csv("sheet.csv")
    over=pd.read_csv("over.csv")

    #====================================Generate year and month list=================================================
    if st.checkbox(label="Show Data",key=1):
        st.table(over)
    #========================================================================================================

    df.dropna(subset=['Top 3 %'],how='any',inplace=True)
    df = df[df['Top 3 %'] !='-']
    df = df[df['Top 3 %'] !=' ']
    df['Top 3 %'].fillna(0,inplace=True)
    df=df.astype({'Top 3 %': 'int64'})

    conditions = [
        (df['Top 3 %'] >=80),
        (df['Top 3 %'] >=50) & (df['Top 3 %'] <80),
        (df['Top 3 %'] >0) & (df['Top 3 %'] <50),
        (df['Top 3 %'] ==0)
        ]
    values = ['Green', 'Yellow', 'Red', 'White']
    df['Status'] = np.select(conditions, values)


    lp=df.groupby(['Month_Year', 'Status']).size().unstack(fill_value=0).reset_index()
    lp['Month']=lp["Month_Year"].str.split(" ",expand=True)[0]
    lp["Month"] = pd.to_datetime(lp.Month, format='%B').dt.month
    lp = lp.sort_values(by="Month")


    st.write("Choose the Range")
    d1,d2=st.beta_columns(2)
    date_of_extraction_from = d1.date_input('Start date', datetime.date(2020, 8, 1))
    date_of_extraction_upto = d2.date_input('End date', datetime.date.today())

    if date_of_extraction_from < date_of_extraction_upto:
        dates = [str(date_of_extraction_from),str(date_of_extraction_upto.replace(day=5))]
        start, end = [datetime.datetime.strptime(_, "%Y-%m-%d") for _ in dates]
        month_list=list(OrderedDict(((start + timedelta(_)).strftime(r"%B %Y"), None) for _ in range((end - start).days)).keys())
        #st.text(month_list)
    else:
        st.error('Error: End date must fall after start date.')
    
    

    #----line plot-------
    choice = st.selectbox("How do you want to view the data",("ALL", "Compare", "Monthly"))
    if choice=='ALL':
        filtered_df = lp[lp['Month_Year'].isin(month_list)]
        fig = px.line(filtered_df, x='Month_Year', y=['Green','#f5f5f5','Yellow','Red'],color_discrete_sequence=['green','#D4D4D4','yellow','red'])
        st.plotly_chart(fig)

    #-----Count plot-------
    elif choice =='Compare':
        filtered_df_bar=df[df['Month_Year'].isin(month_list)]
        bar=px.histogram(filtered_df_bar, x="Month_Year", color='Status',barmode='group' ,color_discrete_sequence=['green','red','white','yellow'])
        st.plotly_chart(bar)

    #----------Pie--------
    else:
        all_stat=df.Status.value_counts().rename_axis('Account_stat').reset_index(name='count')
        if st.checkbox(label="Analyze for a single month"):
            c1,c2=st.beta_columns(2)
            date_from = c1.date_input('Start date', datetime.date.today())
            


        pie = px.pie(all_stat, values='count', names='Account_stat', title="Monitoring Stats",
             color_discrete_sequence=['green','yellow','#D4D4D4','red'])
        st.plotly_chart(pie)

dash()




