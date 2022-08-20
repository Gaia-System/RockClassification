#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import CV_model
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        # return render_template('index.html', filename=filename)

        #######################################################
        image = Image.open(file)
        input = np.resize(image, (1, 224, 224, 3))
        prediction = CV_model.model.predict(input)
        output = 'Metamorphic Prob : {:.5%} / Sedimentary Prob : {:.5%} / Volcanic Prob : {:.5%}'.format(prediction[0][0], prediction[0][1], prediction[0][2])
        return render_template('index.html', filename = filename, Output = output)
        #######################################################

    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()