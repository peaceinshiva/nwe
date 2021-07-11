
import pickle
from flask import Flask,render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import os

xgb=pickle.load(open('static/XGBoost.pkl', 'rb'))
cab=pickle.load(open('static/cab.pkl','rb'))
pdt=pickle.load(open('static/product.pkl','rb'))
name=pickle.load(open('static/name.pkl','rb'))
desti=pickle.load(open('static/destination.pkl','rb'))
pt=pickle.load(open("static/power.pkl",'rb'))

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
@app.route('/')

def upload():
    return render_template("pred.html")

@app.route('/uploader', methods=['GET','POST'])
def uploader():
    if request.method=='POST':
        k=request.form['options']
        if k=='1':
            return render_template("open.html")
        else:
            return render_template('index.html')
            
        
@app.route('/uploadfile', methods=['GET','POST'])
def uploadfile():
    if request.method=='POST':
        f=request.files['file']
        

        fsplit=f.filename.split(".")[-1]
        if fsplit=='xlsx':
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename("test_data.xlsx")))
            test=pd.read_excel("static/test_data.xlsx")
        else:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('test_data.csv')))
            test=pd.read_csv("static/test_data.csv")
            
        test=test[['cab_type', 'name', 'product_id', 'distance', 'surge_multiplier',
           'destination']]
        cat_col=[]
        num_col=[]
        for i in test.columns:
                if test[i].dtype==object:
                    cat_col.append(i)
                else:
                    num_col.append(i)
        conti_df=test[num_col]
        cat=test[cat_col]
        conti_df=conti_df.astype(float)  #Converting to float 
        for i in conti_df.columns:
                if conti_df[i].dtype== float :
                    conti_df[i] = conti_df[i].round(2)
        cat['cab_type']=cab.transform(test['cab_type'])

        cat['product_id']=pdt.transform(test['product_id'])

        cat['name']=name.transform(test['name'])

        cat['destination']=desti.transform(test['destination'])    
        conti_df=pt.transform(conti_df)
        conti_df=pd.DataFrame(conti_df, columns=num_col)
        df=pd.concat([conti_df,cat],axis=1)
        predxg=xgb.predict(df)
        Pred=pd.DataFrame(predxg)
        test['Price Prediction']=Pred
        
        return render_template("simple.html", tables=[test.to_html(classes='data')], titles=test.columns.values)
    
@app.route('/fildetails', methods=['GET','POST'])
def fildetails():
    if request.method=='POST':
        final=[]
        c=request.form['cab_type']
        cab_type=cab.transform([c])
        final.append(int(cab_type))
        
        n=request.form['name']
        name_cab=name.transform([n])
        final.append(int(name_cab))
        
        p=request.form['product']
        product=pdt.transform([p])
        final.append(int(product))

        di=request.form['distance']
        multi=request.form['multiplier']
        data=[[di,multi]]
        df=pd.DataFrame(data,columns=['distance','surge_multiplier'])
        df=pt.transform(df)
        df=pd.DataFrame(df, columns=['distance', 'surge_multiplier'])
                               
        d=request.form['destination']
        destination=desti.transform([d])
        final.append(int(destination))

        data=pd.DataFrame([final], columns=['cab_type', 'name', 'product_id',
       'destination'])
        data=pd.concat([data,df],axis=1)

        predxg=xgb.predict(data)
        
        return render_template("result.html", msg=int(predxg))
                
            
if __name__ == '__main__':
    app.run()

