from flask import Flask, render_template
from flask import request, redirect
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def run():
    if request.method == 'POST':
        #takes input in array and stores it
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]
        # final = final.reshape(1,-1)
        prediction = model.predict(final)
        
    if prediction[0] == 1:
        out = "You May Have Heart Disease"
    else: 
        out = "You Are Fit"

    return render_template("index.html", answer = out)

if __name__ =='__main__':
    app.run()
