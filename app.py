from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import assemblyai as aai
import threading
import asyncio
from constant import assemblyai_api_key
import sys
import pyaudio
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
socketio = SocketIO(app, cors_allowed_origins="*")

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

def on_open(session_opened: aai.RealtimeSessionOpened):
    global session_id
    session_id = session_opened.session_id
    print("Session ID:", session_id)

def on_data(transcript: aai.RealtimeTranscript):
    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeFinalTranscript):
        socketio.emit('transcript', {'text': transcript.text}, namespace='/')
        asyncio.run(analyze_transcript(transcript.text))
    else:
        # Emit the partial transcript to be displayed in real-time
        socketio.emit('partial_transcript', {'text': transcript.text}, namespace='/')

async def analyze_transcript(transcript):
    result = aai.Lemur().task(
        prompt, 
        input_text = transcript,
        final_model=aai.LemurModel.claude3_5_sonnet
    ) 

    print("Emitting formatted transcript for:", transcript)

    socketio.emit('formatted_transcript', {'text': result.response}, namespace='/')

def on_error(error: aai.RealtimeError):
    print(f"An error occurred: {error}")
    socketio.emit('error', {'message': f'Transcription error: {str(error)}'})

def on_close():
    global session_id, transcriber
    session_id = None
    transcriber = None
    print("Closing Session")

def transcribe_real_time():
    global transcriber  
    try:
        print("Initializing microphone...")
        audio = pyaudio.PyAudio()
        
        # List available audio devices
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        print(f"\nAvailable audio devices:")
        for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print(f"Input Device id {i} - {audio.get_device_info_by_host_api_device_index(0, i).get('name')}")
        
        transcriber = aai.RealtimeTranscriber(
            sample_rate=16_000,
            on_data=on_data,
            on_error=on_error,
            on_open=on_open,
            on_close=on_close
        )

        print("Connecting to AssemblyAI...")
        transcriber.connect()

        try:
            print("Starting microphone stream...")
            microphone_stream = aai.extras.MicrophoneStream(
                sample_rate=16_000
            )
            print("Streaming audio to AssemblyAI...")
            transcriber.stream(microphone_stream)
        except Exception as e:
            print(f"Error with microphone stream: {str(e)}")
            socketio.emit('error', {'message': f'Microphone error: {str(e)}'})
            if audio:
                audio.terminate()
            raise
    except Exception as e:
        print(f"Error initializing transcription: {str(e)}")
        socketio.emit('error', {'message': f'Initialization error: {str(e)}'})
        if 'audio' in locals() and audio:
            audio.terminate()
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['audio']
    if file.filename == '':
        return 'No file selected', 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Start transcription in a separate thread
        threading.Thread(target=process_audio_file, args=(filepath,)).start()
        return 'File uploaded successfully', 200

def process_audio_file(filepath):
    try:
        print(f"Processing audio file: {filepath}")
        
        # Create a transcription config
        config = aai.TranscriptionConfig(
            language_detection=True
        )

        # Create a transcriber
        transcriber = aai.Transcriber()
        
        # Submit the audio file
        transcript = transcriber.transcribe(
            filepath,
            config=config
        )

        # Process the transcript with Lemur
        result = aai.Lemur().task(
            prompt,
            input_text=transcript.text,
            final_model=aai.LemurModel.claude3_5_sonnet
        )

        # Emit the formatted transcript
        socketio.emit('file_transcript', {'text': result.response})

    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        socketio.emit('error', {'message': f'Error processing audio: {str(e)}'})
    finally:
        # Clean up the uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error removing temporary file: {str(e)}")

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400

        # Start analysis in a separate thread
        threading.Thread(target=process_text, args=(text,)).start()
        return jsonify({'message': 'Text analysis started'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_text(text):
    try:
        print("Processing text input...")
        
        # Process the text with Lemur
        result = aai.Lemur().task(
            prompt,
            input_text=text,
            final_model=aai.LemurModel.claude3_5_sonnet
        )

        # Emit the formatted text
        socketio.emit('file_transcript', {'text': result.response})

    except Exception as e:
        print(f"Error processing text: {str(e)}")
        socketio.emit('error', {'message': f'Error analyzing text: {str(e)}'})

@socketio.on('toggle_transcription')
def handle_toggle_transcription():
    global transcriber, session_id
    with transcriber_lock:
        if session_id:
            if transcriber:
                print("Closing transcriber session")
                try:
                    transcriber.close()
                except Exception as e:
                    print(f"Error closing transcriber: {str(e)}")
                finally:
                    transcriber = None
                    session_id = None
        else:
            print("Starting transcriber session")
            threading.Thread(target=transcribe_real_time).start()

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    socketio.run(app)

# For Vercel deployment
app = socketio.WSGIApp(socketio, app)
