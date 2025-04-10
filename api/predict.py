from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import numpy as np
import tensorflow as tf

# Add model directory to path so we can import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.preprocessing import preprocess_input

# Load model at module level so it persists between invocations
MODEL_PATH = r"C:\plantM\model\model.h5"  # Updated path to model.h5
model = tf.keras.models.load_model(MODEL_PATH)

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        # Preprocess the input data
        preprocessed_data = preprocess_input(data)
        
        # Make predictions with the model
        predictions = model.predict(preprocessed_data)
        
        # Prepare and send the response
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')  # CORS header for API access
        self.end_headers()
        
        response = {
            "predictions": predictions.tolist(),  # Convert numpy arrays to lists
            "status": "success"
        }
        
        self.wfile.write(json.dumps(response).encode())
        return
        
    def do_OPTIONS(self):
        # Handle preflight CORS requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return