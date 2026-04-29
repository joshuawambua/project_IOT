import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate('ecosphere-434df-firebase-adminsdk-fbsvc-02667124cf.json')
firebase_admin.initialize_app(cred, {'databaseURL': 'https://ecosphere-434df-default-rtdb.firebaseio.com/'})

# Check live data
live_data = db.reference('air_quality/live').get()
print("Live sensor data:", live_data)

# Check if ESP32 is sending data
if live_data:
    print(f"Temperature: {live_data.get('temperature')}")
    print(f"PM2.5: {live_data.get('pm25')}")
else:
    print("No sensor data found - ESP32 may not be running")
