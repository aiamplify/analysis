from flask import Flask, render_template, request, jsonify
import os
from flask_socketio import SocketIO
import assemblyai as aai
import threading
import json
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constant import assemblyai_api_key

app = Flask(__name__, 
            static_folder='../static',
            template_folder='../templates')
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize AssemblyAI
aai.settings.api_key = assemblyai_api_key
transcriber = None
session_id = None
transcriber_lock = threading.Lock()

prompt = """You are a customer service transcript analyzer. Your task is to detect and format words/phrases that fit into the following four categories:

Personal Identifiable Information (PII): Change the font color to red using <span class="pii">...</span>. This includes personal details like names, addresses, phone numbers, and emails.

Product/Service Issues: Highlight the text using <span class="issue">...</span>. This includes any complaints or issues with products or services.

Positive Feedback/Praise: Format the text using <span class="positive">...</span>. This includes any compliments or positive comments.

Order or Account Information: Highlight the text using <span class="order">...</span>. This includes any order numbers, account numbers, or transaction details.

You will receive a customer service transcript along with a list of entities detected by AssemblyAI. Use the detected entities to format the text accordingly and also identify and format any additional relevant items not included in the detected entities. Do not return anything except the original transcript which has been formatted. Don't write any additional prefacing text like "here's the formatted transcript"."""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def on_open(session_opened: aai.RealtimeSessionOpened):
    global session_id
    session_id = session_opened.session_id
    print("Session ID:", session_id)

def on_data(transcript: aai.RealtimeTranscript):
    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeFinalTranscript):
        socketio.emit('transcriptComplete', {'text': transcript.text})
    else:
        socketio.emit('transcriptProgress', {'text': transcript.text})

def on_error(error: aai.RealtimeError):
    print("Error:", error)
    socketio.emit('error', {'error': str(error)})

def on_close():
    print("Closing Session")

def transcribe_real_time():
    global transcriber
    with transcriber_lock:
        if transcriber and transcriber.is_connected():
            transcriber.close()
        
        transcriber = aai.RealtimeTranscriber(
            on_data=on_data,
            on_error=on_error,
            on_open=on_open,
            on_close=on_close,
            sample_rate=44100
        )
        
        transcriber.connect()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    text = request.json.get('text', '')
    
    config = aai.LemurConfig(
        prompt=prompt
    )
    
    lemur = aai.Lemur()
    result = lemur.task(text, config)
    
    return jsonify({'result': result.response})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Create transcription config
        config = aai.TranscriptionConfig(
            lemur=aai.LemurConfig(
                prompt=prompt
            )
        )
        
        # Create the transcriber
        transcriber = aai.Transcriber()
        
        # Submit the audio file
        transcript = transcriber.transcribe(
            filepath,
            config=config
        )
        
        # Get the result
        result = transcript.lemur
        
        # Clean up the file
        os.remove(filepath)
        
        return jsonify({'result': result.response})
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/start-recording', methods=['POST'])
def start_recording():
    try:
        with transcriber_lock:
            if transcriber and transcriber.is_connected():
                print("Transcriber already connected")
                return jsonify({'status': 'already_connected'})
            else:
                print("Starting transcriber session")
                threading.Thread(target=transcribe_real_time).start()
                return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/send-audio', methods=['POST'])
def send_audio():
    try:
        if not request.data:
            return jsonify({'error': 'No audio data received'}), 400
        
        with transcriber_lock:
            if transcriber and transcriber.is_connected():
                transcriber.stream(request.data)
                return jsonify({'status': 'streaming'})
            else:
                return jsonify({'error': 'Transcriber not connected'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop-recording', methods=['POST'])
def stop_recording():
    try:
        with transcriber_lock:
            if transcriber and transcriber.is_connected():
                transcriber.close()
                return jsonify({'status': 'stopped'})
            else:
                print("Transcriber already disconnected")
                return jsonify({'status': 'already_disconnected'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
