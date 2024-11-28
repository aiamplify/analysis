from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from flask_socketio import SocketIO
import assemblyai as aai
import threading
import json
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constant import assemblyai_api_key

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize AssemblyAI
aai.settings.api_key = assemblyai_api_key

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if path == "":
        return send_from_directory('../templates', 'index.html')
    elif os.path.exists(os.path.join('../static', path)):
        return send_from_directory('../static', path)
    elif os.path.exists(os.path.join('../templates', path)):
        return send_from_directory('../templates', path)
    else:
        return send_from_directory('../templates', 'index.html')

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    text = request.json.get('text', '')
    
    config = aai.LemurConfig(
        prompt="""Analyze this customer service transcript and format specific elements:
        - PII (names, addresses, emails): <span class="pii">text</span>
        - Issues/Complaints: <span class="issue">text</span>
        - Positive Feedback: <span class="positive">text</span>
        - Order Info: <span class="order">text</span>"""
    )
    
    lemur = aai.Lemur()
    result = lemur.task(text, config)
    
    return jsonify({'result': result.response})

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy"})

# For Vercel
handler = app
