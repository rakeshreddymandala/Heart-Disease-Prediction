from flask import Flask,render_template,request
import pickle 
import numpy  as np

model= pickle.load(open('our_model.pkl','rb'))

#Create a Flask app instance 
app = Flask(__name__)

#Define a route and a view function 
@app.route('/')
def hello():
    return render_template("home.html")

@app.route('/predict', methods=['post'])
def home():
    data1= request.form.get('a')
    data2= request.form.get('b')
    data3= request.form.get('c')
    data4= request.form.get('d')
    data5= request.form.get('e')
    data6= request.form.get('f')
    data7= request.form.get('g')
    data8= request.form.get('h')
    data9= request.form.get('i')
    data10= request.form.get('j')
    data11= request.form.get('k')
    data12= request.form.get('l')
    data13= request.form.get('m')
    arr=np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13]])
    pred=model.predict(arr)
    return render_template('after.html',data=pred[0])


#Run the app if the file is executed 
if __name__ == '__main__':
    app.run(debug=True)