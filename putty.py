import pandas as pd
import pymysql as db
import logging
import sys
import os
import numpy as np
import sqlalchemy
from datetime import datetime
import json
import streamlit as st
import pandas as pd

def ssh_auto(for_accounts,cred_path):
    server_port = []
    localhost_port = []
    cred = pd.read_csv(cred_path)
    #print(cred)
    for index, row in for_accounts.iterrows():
        server_port.append(row['port'])
        localhost_port.append(row['local_port'])

    server_localhost = {"Server Port": server_port,
                        "LocalHost Port": localhost_port}
    server_localhost_df = pd.DataFrame(server_localhost)

    server_localhost_df.drop_duplicates(keep="first", inplace=True)

    server_localhost_df = server_localhost_df.astype({"Server Port": int, "LocalHost Port": int})

    from sshtunnel import SSHTunnelForwarder

    for index, row in server_localhost_df.iterrows():
        server = SSHTunnelForwarder('172.27.128.59', ssh_username=ldap_user, ssh_password=ldap_pass,
                                    remote_bind_address=('127.0.0.1', int(row["Server Port"])),
                                    local_bind_address=('0.0.0.0', int(row["LocalHost Port"])))

        print(f"Destination Server Port {row['Server Port']} and Source Port {row['LocalHost Port']} in execution")
        print(f" Establishing Connection with Destination Server Port {row['Server Port']} and Source Port {row['LocalHost Port']} in execution")

        server.start()

    print("*********Putty SSH Connection Done************")
    st.success("Connection Extablished Successfully")


