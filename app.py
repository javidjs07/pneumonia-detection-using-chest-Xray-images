import os
import torch
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms

# Import from your src modules
from src.utils import create_model, load_model, load_config
from src.data_preprocessing import get_data_transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load configuration
config = load_config()

# Load the trained model
def load_trained_model():
    model = create_model()
    model_path = config['paths']['trained_models'] + '/pneumonia_resnet50.pth'
    model = load_model(model, model_path)
    model.eval()
    return model

# Initialize model
model = load_trained_model()
class_names = ['NORMAL', 'PNEUMONIA']

# Define image transformations
data_transforms = get_data_transforms()
transform = data_transforms['test']  # Use test transformations

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and preprocess the image
            image = Image.open(filepath).convert('RGB')
            image = transform(image).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted[0]].item()
            
            # Get prediction label
            prediction = class_names[predicted[0]]
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'prediction': prediction,
                'confidence': confidence
            })
            
        except Exception as e:
            # Clean up file if error occurs
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)