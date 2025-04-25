import base64
import os
from urllib import response

from flask import Flask, jsonify, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, UserMixin, logout_user, login_required, current_user
from sqlalchemy.exc import IntegrityError
from mistralai import Mistral
from mistralai.client import MistralClient
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv("SECRET_KEY")

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Class
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
# Load models and tokenizer
try:
    # Load text sentiment model and tokenizer
    text_model = load_model('nlp.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Load image classification model
    image_model = load_model('model.h5')
    
    # Define class labels for both models
    sentiment_mapping = {
        0: 'angry',
        1: 'fear',
        2: 'happy',   # instead of 'Joy'
        3: 'sad'      # instead of 'Sadness'
    }

    image_mapping = {
    1: "neutral",
    2: "sad",
    3: "happy",
    0: "angry"
}
    
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise e

# def predict_text_sentiment(text_content):
#     try:
#         # Preprocess the text
#         sequences = tokenizer.texts_to_sequences([text_content])
#         padded = pad_sequences(sequences, maxlen=100)  # assuming maxlen=100
        
#         # Make prediction
#         prediction = text_model.predict(padded)
#         predicted_class = np.argmax(prediction, axis=1)[0]
#         print("Predicted class index (text):", predicted_class)
#         print("Raw prediction output:", prediction)
#         print("Predicted class index:", predicted_class)
#         return sentiment_mapping.get(predicted_class, 'neutral')
#     except Exception as e:
#         print(f"Error in text prediction: {str(e)}")
#         return 'neutral'

def predict_image_emotion(image_path):
    try:
        # Load and preprocess the image to match model input shape (48, 48, 1)
        img = image.load_img(image_path, color_mode="grayscale", target_size=(48, 48))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 48, 48, 1)
        img_array /= 255.0  # Normalize if the model was trained on normalized data

        # Make prediction
        prediction = image_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]


        return image_mapping.get(predicted_class, 'neutral')

    except Exception as e:
        print(f"Error in image prediction: {str(e)}")
        return 'neutral'

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# SIGNUP Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        cnfPassword = request.form.get('cnfPassword')

        if password != cnfPassword:
            flash('Passwords do not match!', 'danger')
            return redirect('/signup')

        new_user = User(username=username, email=email, password=password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Sign-Up Successful! Please log in.', 'success')
            return redirect('/login')
        except IntegrityError:
            db.session.rollback()
            flash('Username or Email already exists!', 'danger')
            return redirect('/signup')

    return render_template("signup.html")

# LOGIN Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and password == user.password:
            login_user(user)
            flash('Login Successful!', 'success')
            return redirect('/')
        else:
            flash('Invalid email or password!', 'danger')
            return redirect('/login')

    return render_template('login.html')

# LOGOUT Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_prediction')
@login_required
def image_prediction():
    return render_template('image_prediction.html')

@app.route('/text_prediction')
@login_required
def text_prediction():
    return render_template('text_prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/predict_emotion_ml', methods=['POST'])
@login_required
def predict_emotion_ml():
    user_text = request.form.get('user_text')
    if not user_text:
        flash('Please enter some text to analyze', 'danger')
        return redirect('/text_prediction')

    # Tokenize and pad the text input
    sequence = tokenizer.texts_to_sequences([user_text])
    padded_seq = pad_sequences(sequence, maxlen=50)

    # Feed the same padded input into both branches of the model
    prediction = text_model.predict([padded_seq, padded_seq])  # `text_model` is your trained model
    predicted_class = np.argmax(prediction, axis=1)[0]

    sentiment_label = sentiment_mapping[predicted_class]

    return render_template(f'{sentiment_label}.html')
@app.route('/image_prediction', methods=['GET', 'POST'])
@login_required
def image_predictions():
    if request.method == 'POST':
        image_file = request.files.get('image')

        if not image_file:
            flash('No image uploaded', 'danger')
            return redirect('/image_prediction')

        try:
            # Save the file to a temporary location
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
            image_file.save(file_path)

            # Call the prediction function
            prediction_label = predict_image_emotion(file_path)
            
            # Clean up the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

            if prediction_label:
                return render_template(f'{prediction_label}.html')
            else:
                flash('Could not analyze the image. Please try another one.', 'danger')
                return redirect('/image_prediction')
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            flash('Error processing image. Please try again.', 'danger')
            return redirect('/image_prediction')
        

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_message = data['message']
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a kind, empathetic, and emotionally supportive AI assistant who helps users with their mental health. "
                    "You provide gentle advice, emotional encouragement, and always promote positivity and self-care. "
                    "You are not a licensed therapist and do not give medical or clinical advice."
                    "Try to generate brief and concise responses "
                    "When trying to give response in points, always make sure to prive each point in new line."
                    "If the user asks for any question other than mental health, please say 'I am not able to answer that question.'"
                )
            },
            {
                "role": "user",
                "content": user_message,
            }
        ]

        # Send the user's message to the Mistral model
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )

        # Extract the assistant's reply from the response
        response_text = chat_response.choices[0].message.content

        return jsonify({'response': response_text})

    except Exception as e:
        print("Error in chatbot endpoint:", str(e))
        return jsonify({'error': 'Sorry, I encountered an error.'}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))













































































@app.route('/predict_emotion', methods=['POST'])
@login_required
def predict_emotion():
    user_text = request.form.get('user_text')
    if not user_text:
        flash('Please enter some text to analyze', 'danger')
        return redirect('/text_prediction')

    # Our expected chatbot outputs
    valid_categories = ['angry', 'fear', 'happy', 'sad']

    # Chatbot system prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are an emotion classifier. Respond with exactly one word from this list only: Angry, Fear, Happy, Sad. "
                "Only return that one word. No punctuation. No explanation."
            )
        },
        {
            "role": "user",
            "content": f"Classify this message: \"{user_text}\""
        }
    ]

    try:
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )

        category = chat_response.choices[0].message.content.strip()

        if category in valid_categories:
            return render_template(f"{category}.html")
        else:
            flash(f"Unexpected chatbot response: {category}", "danger")
            return redirect('/text_prediction')

    except Exception as e:
        print("Error during chatbot classification:", str(e))
        flash("Something went wrong. Please try again.", "danger")
        return redirect('/text_prediction')
