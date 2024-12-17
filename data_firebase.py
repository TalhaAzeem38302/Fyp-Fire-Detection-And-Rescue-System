import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import base64
import time

location = 'Lahore'
last_execution_time = 0
image_paths = ['images/image_1.jpg','images/image_2.jpg', 'images/image_3.jpg'] 

# Initialize Firebase Admin SDK
cred = credentials.Certificate('fire.json')  # Path to your Firebase service account key
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fire-fyp-509d8-default-rtdb.firebaseio.com/'  # Replace with your database URL
})

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_data( peoples_required):
    global last_execution_time
    current_time = time.time()
    if current_time - last_execution_time < 10:
        return # Exit if 30 seconds have not passed # Update the last execution time 
    last_execution_time = current_time


    current_datetime = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    
    
    # Encode images to base64
    encoded_images = [encode_image(image_path) for image_path in image_paths]

    data = {
        'timestamp': current_datetime,
        'location': location,
        'peoples_required': peoples_required,
        'images': encoded_images
    }

    # Use custom key to set data
    ref = db.reference('Data')  # Reference to the custom path in Firebase
    ref.set(data)  # Set the data to Firebase

    print('Data sent successfully!')


