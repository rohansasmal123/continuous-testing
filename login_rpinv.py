import pandas as pd
import numpy as np

def login():
    
    cred_path=""
    
    server_list=[]
    data=pd.read_csv("Non BAML_ Remittance Prediction Account List - Sheet1.csv")
    data=data[data["Accounts status (by the prod team)"]=="Deployed"]
    server_list=data.Environment.unique()

    cred={'server':server_list,
          'Id': np.nan,
          'Ldap_pass': np.nan,
          'Db_pass': np.nan,
        }
    cred_df=pd.DataFrame.from_dict(cred)

    choice=input("Do you have access to " + str(", ".join(server_list))+" ? [y/n]:")

    if choice.lower()=='y':
        
        ldap_id=input('Enter LDAP-ID')
        ldap_pass=input("Enter LDAP password")
        db_pass=input("Enter DB password")

        for i in range(0,len(server_list)):
            cred_df.Id[i]=ldap_id
            cred_df.Ldap_pass[i]=ldap_pass
            cred_df.Db_pass[i]=db_pass

        cred_df.to_csv("cred.csv",index=False)

    
    else:
        choice=input("******** Enter Credentials For AWS-US ********[y/n]")
        if choice.lower()=='y':

            ldap_id=input('Enter LDAP-ID for AWS-US:')
            ldap_pass=input("Enter LDAP password for AWS-US:")
            db_pass=input("Enter DB password for AWS-US:")
            cred_df.Id[0]=ldap_id
            cred_df.Ldap_pass[0]=ldap_pass
            cred_df.Db_pass[0]=db_pass
            
        choice=input("******** Enter Credentials For AWS-EU ********[y/n]")
        if choice.lower()=='y':
            ldap_id=input('Enter LDAP-ID for AWS-EU:')
            ldap_pass=input("Enter LDAP password for AWS-EU:")
            db_pass=input("Enter DB password for AWS-EU:")

            cred_df.Id[1]=ldap_id
            cred_df.Ldap_pass[1]=ldap_pass
            cred_df.Db_pass[1]=db_pass

        choice=input("******** Enter Credentials For AWS-Gold ********[y/n]")
        if choice.lower()=='y':
            ldap_id=input('Enter LDAP-ID for AWS-Gold:')
            ldap_pass=input("Enter LDAP password for AWS-Gold:")
            db_pass=input("Enter DB password for AWS-Gold:")

            cred_df.Id[2]=ldap_id
            cred_df.Ldap_pass[2]=ldap_pass
            cred_df.Db_pass[2]=db_pass
        
    cred_df.to_csv("cred.csv",index=False)
    
    
