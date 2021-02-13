'''
!/usr/bin/env python
@author:ayanava_dutta,shivam_gupta,rohan_sasmal
-*-coding:utf-8-*-
'''
#--------Packages-----------
import streamlit as st
st.set_page_config(page_title="HighRadius™ | CASH APPLICATION CLOUD", page_icon='/root/caascript/res/bg/logo.png')
import pandas as pd
import numpy as np
import time
from res.bg_css import page_bg
#----------RP---------------
from rpscript.rpmodelling import login
from rpscript.rpmodelling import mod2
from rpscript.rpmodelling import data_prep
from rpscript.rpmodelling import DataExt
#from rpscript.rpmodelling import dashdem
from rpscript.rpmodelling import auto_pilot
from rpscript.rpmonitor import RP_monitoring
from rpscript.rpmonitor import RP_Monitoring_Automated_modified
from rpscript.rpanalysis import rp_analysis_functions
from rpscript.rpanalysis import RP_Analysis
#----------LITM-------------
from litmscript.litmmonitor import LITM_Monitoring_automation_modified
from litmscript.litmmonitor import Monitor_start
from litmscript.litmanalysis import LITM_Analysis
#st.title("CASH APPS")

def all_screen(choice):
    img_path='/root/caascript/res/bg/'
    #st.set_page_config(page_title="HighRadius™ |CASH APPLICATION CLOUD", page_icon='/root/caascript/res/bg/logo.png')
    if choice:
        page_bg(img_path+'Picture_Login.png')
        login.login()
        

    else:
        side_bar = st.sidebar.selectbox(label='What do you want to do?', options=['RP','LITM'])

        if side_bar =='RP':
            rp_bar = st.sidebar.selectbox(label='What do you want to do?', options=['Data Extraction','RP-Modelling','Rp-Monitoring','RP-Analysis'],key=1)

            if rp_bar=='Data Extraction':
                page_bg(img_path+'dataext_pic.png')
                st.header("Data Extraction")
                DataExt.main()
            
                
            elif rp_bar=='RP-Modelling':
                modelling_bar=st.sidebar.selectbox(label='What do you want to do?', options=['Data Preparation','Modelling','Auto-Pilot Mode'],key=2)
                
                if modelling_bar=='Data Preparation':
                    page_bg(img_path+'dataprep_pic.png')
                    st.header("Data Preparation")                    
                    data_prep.data_prep()
     
                elif modelling_bar=='Modelling':
                    page_bg(img_path+'modelling_pic.png')
                    st.header("Model Training") 
                    mod2.modelling_main()

                else:
                    page_bg(img_path+'autopilot_pic.png')
                    st.header("Auto-Pilot Mode")
                    auto_pilot.auto_pilot()


                
            elif rp_bar=='Rp-Monitoring':
                page_bg(img_path+'rpmonitor_pic.png')
                st.header("RP-Monitoring")
                RP_monitoring.main()


            
            elif rp_bar=='RP-Analysis':
                page_bg(img_path+'rpanalysis_pic.png')
                st.header("RP-Analysis")
                if st.checkbox("Show Warning",value=True):
                    st.warning("This feature is still in devolopement phase \n\n Some features may or may not run properly")
                RP_Analysis.main()


        else:
            litm_bar = st.sidebar.selectbox(label='What do you want to do?', options=['LITM Monitoring','LITM Analysis'],key=3)
            if litm_bar=='LITM Monitoring':
                page_bg(img_path+'litmmonitor_pic.png')
                st.header("LITM Monitoring")
                Monitor_start.main()
            else:
                page_bg(img_path+'litm_analysis.png')
                st.header("LITM Analysis")
                if st.checkbox("Show Warning",value=True):
                    st.warning("This feature is still in devolopement phase \n\n Some features may or may not run properly")
                LITM_Analysis.analysis()






if __name__ == '__main__':
    
    choice = st.sidebar.checkbox('Login',value=True)
    all_screen(choice)

