import numpy as np
from PIL import Image
from flask import Flask,render_template,request
app=Flask(__name__)
@app.route("/")
def home():
    return render_template("upload.html")
@app.route("/result",methods=['POST',"GET"])
def result():
    output = request.form.to_dict()
    image = output["image"]
    image =r"F:"+"/"+image
    print(image)
    image = Image.open(image)
    image = image.resize((30,30))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    predict_x=model.predict(image)
    pred=np.argmax(predict_x,axis=1)
    sign = classes[pred[0]+1]
    image=sign
    return render_template("upload.html",i=image,s=sign)
if __name__ == '__main__':
    app.run(debug=True,port=5001)