from django.http.response import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import json
import joblib
import tensorflow as tf
import numpy as np

from tensorflow import Graph
img_height, img_width = 256,256
with open('model.json','r') as f:
    labelInfo=f.read()
labelInfo = json.loads(labelInfo)


model_graph=Graph()
with model_graph.as_default():
    tf_session=tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model("model.h5")

def home(request):
    return render(request,"home.html")
# Create your views here.

    
def images(request):
    
    print(request)
    fileObj = request.FILES.get('covidtest')
    fs=FileSystemStorage()
    filePathName= fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)

    testimage= '.'+filePathName
    img= image.load_img(testimage,target_size=(img_height,img_width))
    x=image.img_to_array(img)
    x=x/255
    x=x.reshape(1,img_height,img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)
    print(predi)
    predictedLabel = predi
    if predictedLabel>0.5:

        context={'filePathName':filePathName,'predictedLabel':'Covid Positive'}
    else:
        context={'filePathName':filePathName,'predictedLabel':'Covid Negative'}


    return render(request,'home.html',context)
def result(request):
    cls = joblib.load('CovidDetectSym.sav')
    lis1=[]
    lis=['BP','Fev','DC','ST','HT','Ab','CC','Alg','Pub','Fam']
    if request.method=='POST':
        com=request.POST.getlist('co')
        print(com)
        if 'BP' in com:
            lis1.append(1)
        else:
            lis1.append(0)

        if 'Fev' in com:
            lis1.append(1)
        else:
            lis1.append(0)
        if 'DC' in com:
            lis1.append(1)
        else:
            lis1.append(0)
        if 'ST' in com:
            lis1.append(1)
        else:
            lis1.append(0)
        if 'HT' in com:
            lis1.append(1)
        else:
            lis1.append(0)
        if 'Ab' in com:
            lis1.append(1)
        else:
            lis1.append(0)
        if 'CC' in com:
            lis1.append(1)
        else:
            lis1.append(0)
        if 'Alg' in com:
            lis1.append(1)
        else:
            lis1.append(0)
        if 'Pub' in com:
            lis1.append(1)
        else:
            lis1.append(0)
        if 'Fam' in com:
            lis1.append(1)
        else:
            lis1.append(0)
        

    print(lis1)
    ans=cls.predict([lis1])
    if ans==1:
        ans='Covid Positive'
    else:
        ans='Covid Negative'

    return render(request,"home.html",{'ans':ans})