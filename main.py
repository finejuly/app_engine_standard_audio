# all the imports
from __future__ import with_statement
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, \
     abort, render_template, flash

from contextlib import closing
import os

# for safe file upload
from werkzeug import secure_filename

import utils

UPLOAD_FOLDER = 'storage'
ALLOWED_EXTENSIONS = set(['wav'])

# create our little application :)
app = Flask(__name__)
app.config.from_object(__name__)
app.secret_key='itssecretman'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def main():
    wav_filename = os.path.join(app.config['UPLOAD_FOLDER'],session['wav_filename'])
    im_filename=wav_filename+'.png'
    fea_filename = wav_filename+'_feature.png'

    if session['wav_filename'] != None:
        input_batch = utils.utils.wavfile_to_melfile(wav_filename, im_filename)
        postprocessed_batch = utils.utils.feature_extraction(input_batch,fea_filename)

    return render_template('show_audio.html', wav_filename = wav_filename, im_filename = im_filename, fea_filename = fea_filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/select_wav', methods=['POST','GET'])
def select_wav():
    file = request.files['wav_file']
    #if file and allowed_file(file.filename):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        session['wav_filename']=filename
        file.save(os.path.join('static',os.path.join(app.config['UPLOAD_FOLDER'], filename)))
        return redirect(url_for('main'))


@app.before_first_request
def before_first_request():
    #session['wav_filename']="abjones_1_02.wav"
    session['wav_filename']="341739__deleted-user-6109353__roman-marching-music-loop-with-horns-1.wav"
    # https://freesound.org/people/deleted_user_6109353/sounds/341739/


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=False)
