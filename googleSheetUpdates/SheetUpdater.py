import webbrowser
import dataScraping.LegoPriceScraper as LegoPriceScraper
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials 
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

webbrowser.register('chrome', None, webbrowser.BackgroundBrowser('/usr/bin/google-chrome')) 

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

load_dotenv()
SPREADSHEET_ID = os.getenv("SPREADSHEETID")

if not SPREADSHEET_ID: 
  raise ValueError("No spreadsheet ID found in .env file.")

def authenticate(): 
  creds = None
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
      creds = flow.run_local_server(port=0,open_browser=False)
      webbrowser.get('chrome').open('http://localhost:8080')
    with open("token.json", "w") as token:
      token.write(creds.to_json())
    
  return creds

def brickUpdateValues(values):
  creds = authenticate()
  try: 
    service = build("sheets", "v4", credentials=creds)
    data = []
    for i, row_values in enumerate(values,start=46):
        range_name = f"DATA_Lego!B{i}:E{i}"
        data.append({"range": range_name, "values": [row_values]})
    body = {"valueInputOption": "USER_ENTERED", "data": data}
    result = (
        service.spreadsheets()
        .values()
        .batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body)
        .execute()
    )
    print(f"{(result.get('totalUpdatedCells'))} cells updated.")
    return result
  except HttpError as error:
    print(f"An error occurred: {error}")
    return error

def legoUpdateValues():
  creds = authenticate()
  try:
    service = build("sheets", "v4", credentials=creds)
    values = LegoPriceScraper.UpdatedLEGOData()
    data = [] 
    for i, row_values in enumerate(values,start=4):
        range_name = f"DATA_Lego!C{i}:F{i}"
        data.append({"range": range_name, "values": [row_values]})
    body = {"valueInputOption": "USER_ENTERED", "data": data}
    result = (
        service.spreadsheets()
        .values()
        .batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body)
        .execute()
    )
    
    print(f"{(result.get('totalUpdatedCells'))} cells updated.")
    return result
  except HttpError as error:
    print(f"An error occurred: {error}")
    return error

def main():
  legoUpdateValues()
    
if __name__ == "__main__":
  main()