from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os.path
import pickle
from datetime import datetime, timedelta

class CalendarService:
    def __init__(self, config):
        self.config = config
        self.SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
        self.creds = None
        self._authenticate()
        self.service = build('calendar', 'v3', credentials=self.creds)

    def _authenticate(self):
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                self.creds = pickle.load(token)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(self.creds, token)

    async def get_upcoming_events(self, days=7):
        now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        time_max = (now + timedelta(days=days))
        
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now.isoformat() + 'Z',
                timeMax=time_max.isoformat() + 'Z',
                maxResults=100, 
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                return "No upcoming events found."
                
            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
                formatted_events.append(f"- {event['summary']} on {start_time.strftime('%B %d at %I:%M %p')}")
                
            return "\n".join(formatted_events)
            
        except Exception as e:
            return f"Error accessing calendar: {str(e)}"