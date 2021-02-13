import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

def get_data(sheet_name):

    SCOPES=['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_file('client_secret.json', scopes=SCOPES)
    client=gspread.authorize(credentials)
    sheet=client.open('Dummy_LITM_Monitoring').worksheet(sheet_name)
    data=sheet.get_all_records()
    dataset = pd.DataFrame(data)
    
    return dataset