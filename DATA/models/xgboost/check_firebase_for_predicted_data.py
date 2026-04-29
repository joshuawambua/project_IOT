import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate('ecosphere-434df-firebase-adminsdk-fbsvc-02667124cf.json')
firebase_admin.initialize_app(cred, {'databaseURL': 'https://ecosphere-434df-default-rtdb.firebaseio.com/'})

# Check predictions
latest = db.reference('xgboost/predictions/latest').get()
print("Latest prediction:", latest)

# Check stats
stats = db.reference('xgboost/stats').get()
print("Stats:", stats)
