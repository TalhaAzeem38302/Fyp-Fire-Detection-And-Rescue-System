from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import base64
from datetime import datetime
import threading
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db as data
from flask_socketio import SocketIO



app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('fire.json')  # Path to your Firebase service account key
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fire-fyp-509d8-default-rtdb.firebaseio.com/'  # Replace with your database URL
})

notifications = []  # List to store incoming notifications

# Secret key for session management and flash messages
app.config['SECRET_KEY'] = 'your_secret_key'

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
socketio = SocketIO(app)  # Initialize SocketIO

# Define the Image model
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, db.ForeignKey('record.id'), nullable=False)
    image = db.Column(db.LargeBinary, nullable=False)



def listen_to_firebase():
    ref = data.reference('Data')  # Reference to the 'Data' node in Firebase
    ref.listen(firebase_event)

def firebase_event(event):
    # print('Event type: {}'.format(event.event_type))  # 'put' or 'patch'
    # print('Path: {}'.format(event.path))  # Relative to the reference
    # print('Data: {}'.format(event.data))  # New data at the path
    if event.event_type in ['put', 'patch'] and event.data:
        notifications.append(event.data)
        print('New notification added')
        # Run save_to_database within application context
        with app.app_context():
            save_to_database(event.data)

def save_to_database(data):
    # Check if a record with the same date and time already exists
    existing_record = Record.query.filter_by(date=data.get('timestamp')).first()
    
    
    record = Record(
        location=data.get('location'),
        description="No description",  # Add a default description if not provided
        peoples_required=data.get('peoples_required'),
        deaths=0,  # Default value if not provided
        saved_lives=0,  # Default value if not provided
        average_time="0",  # Default value if not provided
        lose=0,  # Default value if not provided
        date=data.get('timestamp')
    )
    if existing_record:
        print('Record with the same date and time already exists. Skipping save.')
        return
    else:
        db.session.add(record)
        db.session.commit()

        # Save images to the database
        images = data.get('images', [])
        for img_data in images:
            image = Image(
                record_id=record.id,
                image=base64.b64decode(img_data)  # Decode base64 to binary
            )
            db.session.add(image)
        db.session.commit()
        print('Data and images saved to the database!')
        socketio.emit('new_notification', {'location': record.location})
      


@app.route('/notifications')
def get_notifications():
    return jsonify(notifications)
# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Define the Record model
class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(200))
    description = db.Column(db.String(200), nullable=False)
    peoples_required = db.Column(db.Integer, nullable=False)
    deaths = db.Column(db.Integer, nullable=False)
    saved_lives = db.Column(db.Integer, nullable=False)
    average_time = db.Column(db.String(50), nullable=False)
    lose = db.Column(db.Integer, nullable=False)
    date = db.Column(db.String(100), nullable=False)



# Initialize the database
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
    if existing_user:
        if existing_user.username == username:
            flash('Username already exists. Please choose a different username.', 'error')
        if existing_user.email == email:
            flash('Email already exists. Please use a different email.', 'error')
    else:
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful!', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/get_dashboard_data')
def get_dashboard_data():
    current_year = datetime.now().year
    start_date = datetime(current_year, 1, 1)
    end_date = datetime(current_year, 12, 31)

    locations = ["Islamabad", "Lahore"]

    data = {}
    for location in locations:
        records = Record.query.filter(Record.date >= start_date.strftime('%Y-%m-%d'),
                                      Record.date <= end_date.strftime('%Y-%m-%d'),
                                      Record.location == location).all()

        if records:
            monthly_data = {i: 0 for i in range(1, 13)}  # Initialize months with zero incidents
            deaths, saved_lives, lose = 0, 0, 0
            for record in records:
                month = datetime.strptime(record.date, "%Y-%m-%d %I:%M:%S %p").month
                monthly_data[month] += 1
                deaths += record.deaths
                saved_lives += record.saved_lives
                lose += record.lose

            data[location] = {
                "months": [datetime(current_year, i, 1).strftime('%B') for i in range(1, 13)],
                "incidents": [monthly_data[i] for i in range(1, 13)],
                "deaths": deaths,
                "saved_lives": saved_lives,
                "average_time": round(sum(int(record.average_time) for record in records) / len(records)) if records else 0,
                "lose": lose
            }
        else:
            data[location] = {
                "months": [datetime(current_year, i, 1).strftime('%B') for i in range(1, 13)],
                "incidents": [0] * 12,
                "deaths": 0,
                "saved_lives": 0,
                "average_time": 0,
                "lose": 0
            }

 
    return jsonify(data)


@app.route('/record', methods=['GET', 'POST'])
def record():
    if request.method == 'POST':
        location = request.form['location']
        description = request.form['description']
        peoples_required = int(request.form['peoples_required'])
        deaths = int(request.form['deaths'])
        saved_lives = int(request.form['saved_lives'])
        average_time = request.form['average_time']
        lose = int(request.form['lose'])
        date = request.form['timedate']

        # Handle image uploads
        images = request.files.getlist('images')

        new_record = Record(
            location=location,
            description=description,
            peoples_required=peoples_required,
            deaths=deaths,
            saved_lives=saved_lives,
            average_time=average_time,
            lose=lose,
            date=date
        )
        db.session.add(new_record)
        db.session.commit()

        # Add images to the database
        # for image in images:
        #     image_data = image.read()
        #     new_image = Image(record_id=new_record.id, image=image_data)
        #     db.session.add(new_image)

        for image in images: 
            image_data = image.read() 
            new_image = Image(record_id=new_record.id, image=image_data) 
            db.session.add(new_image)

        db.session.commit()
        flash('Record with images added successfully!', 'success')
        return redirect(url_for('record'))

    return render_template('record.html')

@app.route('/assign_work', methods=['GET', 'POST'])
def assign_work():
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check for the last entry created today with "No description"
    last_entry = Record.query.filter(Record.description=="No description", Record.date.contains(today)).order_by(Record.id.desc()).first()

    if request.method == 'POST':
        if not last_entry:
            flash('No Incident is reported.', 'warning')
            return redirect(url_for('assign_work'))

        # Get form data
        workers = request.form.getlist('worker')
        work_description = request.form['workDescription']
        priority = request.form['priority']

        # Create the description
        description = f"Priority Level: {priority}\nWorkers Names:\n"
        for index, worker in enumerate(workers, start=1):
            description += f"Worker {index}: {worker}\n"
        description += f"Description: {work_description}"

        # Update the last entry
        last_entry.description = description
        db.session.commit()

        flash('Work assigned successfully!', 'success')
        return redirect(url_for('assign_work'))

    return render_template('assign_work.html', last_entry=last_entry)

@app.route('/api/work/<int:record_id>', methods=['GET', 'PUT'])
def handle_work_record(record_id):
    if request.method == 'GET':
        record = Record.query.get_or_404(record_id)
        images = Image.query.filter_by(record_id=record.id).all()
        image_data = [base64.b64encode(image.image).decode('utf-8') for image in images]
        return jsonify({
            'id': record.id,
            'location': record.location,
            'description': record.description,
            'average_time': record.average_time,
            'date': record.date,
            'peoples_required': record.peoples_required,
            'deaths': record.deaths,
            'saved_lives': record.saved_lives,
            'lose': record.lose,
            'images': image_data
        })

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username, password=password).first()
    if user:
        session['user_id'] = user.id
        session['username'] = user.username
        return redirect(url_for('user_home'))  # Redirect to the home page on successful login
    else:
        flash('Invalid credentials!', 'error')
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/home')
def user_home():
    username = session.get('username')
    return render_template('home.html', username=username,  notifications=notifications)

@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        flash('You must be logged in to change your password.', 'error')
        return redirect(url_for('home'))

    user_id = session['user_id']
    old_password = request.form['old_password']
    new_password = request.form['new_password']
    repeat_password = request.form['repeat_password']

    user = User.query.filter_by(id=user_id).first()
    if user and user.password == old_password:
        if new_password == repeat_password:
            user.password = new_password
            db.session.commit()
            flash('Password changed successfully!', 'success')
        else:
            flash('New passwords do not match.', 'error')
    else:
        flash('Old password is incorrect.', 'error')

    return redirect(url_for('user_home'))

@app.route('/api/records', methods=['GET'])
def get_records():
    start_date = request.args.get('startDate')
    end_date = request.args.get('endDate')
    location = request.args.get('location')

    query = Record.query

    # Filter by date range
    if start_date:
        start_date = datetime.fromisoformat(start_date)
        query = query.filter(Record.date >= start_date.strftime("%Y-%m-%d %I:%M:%S %p"))
    
    if end_date:
        end_date = datetime.fromisoformat(end_date)
        query = query.filter(Record.date <= end_date.strftime("%Y-%m-%d %I:%M:%S %p"))

    # Filter by location
    if location:
        query = query.filter(Record.location.ilike(f'%{location}%'))
    
    records = query.all()
    result = []
    
    for record in records:
        images = Image.query.filter_by(record_id=record.id).all()
        image_data = [base64.b64encode(image.image).decode('utf-8') for image in images]
        
        result.append({
            'id': record.id,
            'location': record.location,
            'description': record.description,
            'average_time': record.average_time,
            'date': record.date,
            'peoples_required': record.peoples_required,
            'deaths': record.deaths,
            'saved_lives': record.saved_lives,
            'lose': record.lose,
            'images': image_data
        })
    return jsonify(result)

@app.route('/record/<int:record_id>')
def record_details(record_id):
    record = Record.query.get_or_404(record_id)
    return render_template('record_details.html', record=record)


@app.route('/api/records/<int:record_id>', methods=['GET', 'PUT'])
def handle_record(record_id):
    if request.method == 'GET':
        record = Record.query.get_or_404(record_id)
        images = Image.query.filter_by(record_id=record.id).all()
        image_data = [base64.b64encode(image.image).decode('utf-8') for image in images]
        return jsonify({
            'id': record.id,
            'location': record.location,
            'description': record.description,
            'average_time': record.average_time,
            'date': record.date,
            'peoples_required': record.peoples_required,
            'deaths': record.deaths,
            'saved_lives': record.saved_lives,
            'lose': record.lose,
            'images': image_data
                    })

    elif request.method == 'PUT':
        record = Record.query.get(record_id)  # Fetch the record, may return None

        if record is None:
            return jsonify({'error': 'Record not found'}), 404

        data = request.get_json()

        # Update fields only if they are present in the request
        if 'description' in data:
            record.description = data['description']
        if 'peoples_required' in data:
            record.peoples_required = data['peoples_required']
        if 'deaths' in data:
            record.deaths = data['deaths']
        if 'saved_lives' in data:
            record.saved_lives = data['saved_lives']
        if 'average_time' in data:
            record.average_time = data['average_time']
        if 'lose' in data:
            record.lose = data['lose']

        db.session.commit()
        return jsonify({'message': 'Record updated successfully'})

if __name__ == '__main__':
    # Start the Firebase listener in a separate 
    threading.Thread(target=listen_to_firebase, daemon=True).start() # Run the Flask-SocketIO app socketio.run(app, debug=True)
    socketio.run(app, debug=True)
