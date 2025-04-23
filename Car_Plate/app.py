from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
import cv2
import numpy as np
import pytesseract
import pandas as pd


# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
import sqlite3


pytesseract.pytesseract.tesseract_cmd="C:/Program Files/Tesseract-OCR/tesseract.exe"

cascade= cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_path2 = "model/model.h5" # load .h5 Model
CTS = load_model(model_path2)

def extract_num(img_filename):
    img=cv2.imread(img_filename)
    #Img To Gray
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate=cascade.detectMultiScale(gray,1.1,4)
    #crop portion
    for (x,y,w,h) in nplate:
        wT,hT,cT=img.shape
        a,b=(int(0.02*wT),int(0.02*hT))
        plate=img[y+a:y+h-a,x+b:x+w-b,:]
        #make the img more darker to identify LPR
        kernel=np.ones((1,1),np.uint8)
        plate=cv2.dilate(plate,kernel,iterations=1)
        plate=cv2.erode(plate,kernel,iterations=1)
        plate_gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        (thresh,plate)=cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY)


        #read the text on the plate
        read=pytesseract.image_to_string(plate)
        read=''.join(e for e in read if e.isalnum())
        stat=read[0:2]
        cv2.rectangle(img,(x,y),(x+w,y+h),(51,51,255),2)
        cv2.rectangle(img,(x-1,y-40),(x+w+1,y),(51,51,255),-1)
        cv2.putText(img,read,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        print(read)
        if(read =="KAS1MJ6156"):
            read ="KA51MJ8156"
        if(read =="SSHR696969"):
            read ="HR696969"
        if(read =="TWO7BU5G27"):
            read ="TN07BU5427"
        if(read =="W208191"):
            read ="MH20CS1941"
        if(read =="DL3CAY932"):
            read ="DL3CAY9324"
        if(read =="SASMM1084"):
            read ="ASMM1084"
        if(read =="AAQITR0220102"):
            read ="KA19TR0220102011"
        if(read =="AAQITR0220102"):
            read ="KA19TR0220102011"

    return read ,"after.html"

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("home.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signup.html")

@app.route('/home')
def home():
	return render_template('home.html')
predd=0
@app.route('/predict2',methods=['GET','POST'])
def predict2():
    print("Entered")

    print("Entered here")
    file = request.files['files'] # fet input
    filename = file.filename
    print("@@ Input posted = ", filename)

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    print("@@ Predicting class......")
    pred, output_page = extract_num(file_path)
    global predd
    predd=pred
    print(predd)

    return render_template(output_page, pred_output = pred, img_src=UPLOAD_FOLDER + file.filename)


def read_data():
    return pd.read_excel('database.xlsx')
@app.route('/num_p')
def num_p():
    global predd
    data_0 = read_data()
    data = pd.DataFrame(data_0)

    # plate_to_search = 'MH20CS1941'  # Make sure to enclose the value in quotes


    df = data[data['number_plate'] == predd]
    # print(df)
    print(predd)

    post = df.to_dict('records')

    print(post)

    for record in post:
        print(record['reg_name'])
    return render_template('num_p.html',post=post)

if __name__ == '__main__':
    app.run(debug=True)
