# app.py

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from feedback import grade_with_rubric, render_feedback_text

import os
import json, re
from functools import lru_cache
import logging  # Ensure logging is imported
import PyPDF2
import docx
from typing import Dict, Any, List
import openai
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

RUBRIC_PATH = os.path.join("rubrics", "current.json")
RUBRIC_PATH = os.getenv("RUBRIC_PATH", os.path.join("rubrics", "current.json"))
COURSE_NAME_OVERRIDE = os.getenv("COURSE_NAME")


DEFAULT_RUBRIC: Dict[str, Any] = {
    "course_name": "Unnamed Course",
    "total_points": 15,
    "scale": [3, 2, 1],
    "labels": ["Excellent", "Satisfactory", "Needs Improvement"],
    "criteria": [
        {"name": "Correctness", "desc": "", "levels": {"3": "", "2": "", "1": ""}},
        {"name": "Clarity", "desc": "", "levels": {"3": "", "2": "", "1": ""}},
        {"name": "Completeness", "desc": "", "levels": {"3": "", "2": "", "1": ""}},
        {"name": "Consistency", "desc": "", "levels": {"3": "", "2": "", "1": ""}},
        {"name": "Organization & Coherence", "desc": "", "levels": {"3": "", "2": "", "1": ""}}
    ]
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure maximum upload size (optional)
app.config['MAX_CONTENT_LENGTH'] = 80 * 1024 * 1024  # 80 MB

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set logging level and format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password_hash = db.Column(db.String(128), nullable=False)
    
    rubrics = db.relationship('Rubric', backref='user', lazy=True)
    chat_histories = db.relationship('ChatHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Rubric model
class Rubric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    score = db.Column(db.Float, nullable=True)  
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Chat history model
class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prompt_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    prompt_content = db.Column(db.Text, nullable=False)
    response_time = db.Column(db.DateTime, nullable=False)
    response_content = db.Column(db.Text, nullable=False)
    user_rating = db.Column(db.Integer)  # User rating (optional)
    user_feedback = db.Column(db.Text)   # User feedback (optional)

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.', 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        # Create new user and save to database
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        session.clear()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect to main page
    if 'user_id' in session and User.query.get(session['user_id']):
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user is None or not user.check_password(password):
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))

        session['user_id'] = user.id

        flash('Welcome back!', 'success')

        return redirect(url_for('index'))

    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# Main page route
@app.route('/')
def index():
    user_id = session.get('user_id')
    if not user_id or not User.query.get(user_id):
        flash('Please log in first.', 'danger')
        return redirect(url_for('login'))

    current_user = User.query.get(user_id)
    if not current_user:
        # Handle user not found
        return redirect(url_for('login'))

    user_rubrics = Rubric.query.filter_by(user_id=user_id).all()
    rubrics = [rubric.text for rubric in user_rubrics]

    # No longer needed LEED Form
    return render_template('index.html', user=current_user, rubrics=rubrics, leed_table_data=[])

@app.route('/get_user_rubrics', methods=['GET'])
def get_user_rubrics():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'User not logged in.'})

    user_rubrics = Rubric.query.filter_by(user_id=user_id).all()
    rubrics = [rubric.text for rubric in user_rubrics]

    return jsonify({'success': True, 'rubrics': rubrics})

@app.route('/get_feedback', methods=['POST'])
def get_feedback_route():
    user_id = session.get('user_id')
    if not user_id:
        logging.warning('User not logged in.')
        return jsonify({'success': False, 'error': 'User not logged in.'}), 401

    prompt_time = datetime.utcnow()

    # 1) Get user input (file has priority; fall back to textarea message)
    file_path = None
    uploaded_file = request.files.get('file')
    if uploaded_file and uploaded_file.filename:
        filename = secure_filename(uploaded_file.filename)
        root, ext = os.path.splitext(filename)
        ext = ext.lower().lstrip('.')
        if ext in ALLOWED_EXTENSIONS:
            # (optional) uniquify to avoid overwrite
            # import uuid at top if you enable next two lines
            # unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
            # file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            file_text = ""
            try:
                if ext == 'pdf':
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                file_text += text + "\n"
                elif ext == 'docx':
                    doc = docx.Document(file_path)
                    file_text = "\n".join([para.text for para in doc.paragraphs])
            except Exception:
                logging.exception("Error extracting text from file")
            prompt_content = file_text if file_text else f"Uploaded file: {filename}"
        else:
            return jsonify({'success': False, 'error': 'Invalid file type. Only PDF and DOCX files are allowed.'}), 400
    else:
        user_input = request.form.get('message', '').strip()
        if not user_input:
            return jsonify({'success': False, 'error': 'No user input provided.'}), 400
        prompt_content = user_input

    # 2) get rubric (rubrics/current.json)
    rubric = load_rubric()

    # 3) Call feedback.py to grade with the rubric (LLM call + JSON parsing inside)
    try:
        result = grade_with_rubric(prompt_content, rubric, model=OPENAI_MODEL)
    except Exception as e:
        logging.exception("Grading failed")
        return jsonify({'success': False, 'error': f'{e}'}), 500

    # 4) Unpack result and render a human-readable block
    items = result["items"]                  # [{name, score, reasons}]
    total = result["total"]                  # int
    overall_label = result["overall_label"]  # str
    data = result["raw_model_json"]          # original JSON from model (debugging)

    # 5) Render feedback text shown in the chat bubble
    feedback_text = render_feedback_text(rubric, items, total, overall_label)

    # 6) Capture response timestamp
    response_time = datetime.utcnow()

    # 7) Clean up temp upload (if any)
    if file_path:
        try:
            os.remove(file_path)
        except Exception as e:
            logging.warning(f"Failed to remove uploaded file: {file_path}. Error: {e}")

    # 8) Persist chat history and rubric scores
    try:
        chat_history = ChatHistory(
            user_id=user_id,
            prompt_time=prompt_time,
            prompt_content=prompt_content,
            response_time=response_time,
            response_content=feedback_text
        )
        db.session.add(chat_history)

        # Remove previous rubric rows for this user
        Rubric.query.filter_by(user_id=user_id).delete()

        # Store new per-criterion scores
        for c in items:
            db.session.add(Rubric(
                text=c['name'],
                score=float(c['score']),
                user_id=user_id
            ))
        db.session.commit()
    except Exception as e:
        logging.exception("Error saving chat history or rubrics")
        return jsonify({'success': False, 'error': f"Error saving data: {e}"}), 500

    # 9) Return response to frontend
    return jsonify({
        'success': True,
        'feedback': feedback_text,
        'rubric': {
            'course_name': rubric['course_name'],
            'total_points': rubric['total_points'],
            'labels': rubric['labels'],
            'criteria': [c['name'] for c in rubric['criteria']]
        },
        'scores': items,
        'total': total,
        'overall_label': overall_label,
        'chat_history_id': chat_history.id,
        'raw_model_json': data  # remove in prod if you prefer
    })


# Submit user feedback route
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'User not logged in.'})

    data = request.get_json()
    chat_history_id = data.get('chat_history_id')
    rating = data.get('rating')
    feedback_text = data.get('feedback')

    chat_history = ChatHistory.query.filter_by(id=chat_history_id, user_id=user_id).first()
    if not chat_history:
        return jsonify({'success': False, 'error': 'Chat history not found.'})

    chat_history.user_rating = rating if rating else None
    chat_history.user_feedback = feedback_text if feedback_text else None

    db.session.commit()

    return jsonify({'success': True})

@app.route('/save_rubrics', methods=['POST'])
def save_rubrics():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'User not logged in.'})
    
    # Get the current user object
    current_user = User.query.get(user_id)
    if not current_user:
        return jsonify({'success': False, 'error': 'User not found.'})
    
    # Permission check
    if current_user.username != 'admin':
        return jsonify({'success': False, 'error': 'You do not have permission to perform this action.'})
    
    data = request.get_json()
    rubrics_input = data.get('rubrics')

    if rubrics_input is not None:
        # Update rubrics in the database
        Rubric.query.filter_by(user_id=user_id).delete()
        for rubric_text in rubrics_input.strip().split('\n\n'):
            if rubric_text.strip():
                new_rubric = Rubric(text=rubric_text.strip(), user_id=user_id)
                db.session.add(new_rubric)
        db.session.commit()
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'No rubrics provided.'})
    

def load_general_rubric():
    """
    Load the general writing Rubric.
    Ensure the 'cleaned_leed_rubric.json' file exists in the same directory or provide the correct path.
    """
    rubric_path = os.path.join(os.path.dirname(__file__), "cleaned_leed_rubric.json")
    if not os.path.exists(rubric_path):
        raise FileNotFoundError(f"Rubric file not found at path: {rubric_path}")
    
    with open(rubric_path, "r", encoding="utf-8") as f:
        try:
            rubric_data = json.load(f)
            if not isinstance(rubric_data, list):
                raise ValueError("Rubric data should be a list of dictionaries.")
            return rubric_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing rubric JSON: {e}")
        

# Temporarily storing Rubric data
rubric_storage = None

# Receive WRITING_RUBRIC data from the front end
@app.route('/save_WRITING_RUBRICs', methods=['POST'])
def save_writing_rubrics():
    global rubric_storage  # Using global variables to store
    try:
        # Get JSON data from the request
        rubric_data = request.get_json()
        if not rubric_data:
            return jsonify({"error": "No data provided"}), 400

        rubric_storage = rubric_data  # Save to global variable
        return jsonify({"message": "Rubric saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Get the currently stored Rubric data
@app.route('/get_WRITING_RUBRICs', methods=['GET'])
def get_writing_rubrics():
    LEED_RUBRIC = [
        {
            "name": "LEED Certification Achievement",
            "scoringCriteria": [
                {"points": 3, "description": "This is a test rubric."}
            ]
        }
    ]
    return jsonify(LEED_RUBRIC)

@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception("Unhandled exception occurred:")
    return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500

@app.route('/api/rubric', methods=['GET'])
def api_rubric():
    return jsonify(load_rubric())

@app.route('/get_last_feedback', methods=['GET'])
def get_last_feedback():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'User not logged in.'}), 401

    # Query the last feedback record in descending order of response_time
    last_chat = ChatHistory.query.filter_by(user_id=user_id).order_by(ChatHistory.response_time.desc()).first()
    if not last_chat:
        return jsonify({'success': False, 'error': 'No previous feedback found.'})

    return jsonify({
        'success': True,
        'feedback': last_chat.response_content,
        'chat_history_id': last_chat.id  
    })

def load_rubric() -> Dict[str, Any]:
    """
    Load rubric from rubrics/current.json if it exists; otherwise fall back to DEFAULT_RUBRIC.
    """
    try:
        if os.path.exists(RUBRIC_PATH):
            with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Fill missing fields with defaults
                data.setdefault("course_name", DEFAULT_RUBRIC["course_name"])
                data.setdefault("total_points", DEFAULT_RUBRIC["total_points"])
                data.setdefault("scale", DEFAULT_RUBRIC["scale"])
                data.setdefault("labels", DEFAULT_RUBRIC["labels"])
                if "criteria" not in data or not isinstance(data["criteria"], list):
                    data["criteria"] = DEFAULT_RUBRIC["criteria"]
                if COURSE_NAME_OVERRIDE:
                    data["course_name"] = COURSE_NAME_OVERRIDE
                return data
    except Exception as e:
        logging.warning(f"[rubric] Failed to load rubric; using default. Error: {e}")
    return DEFAULT_RUBRIC


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get the port from the environment variable, default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)  # Host must be 0.0.0.0 to work on Heroku
