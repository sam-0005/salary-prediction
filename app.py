import numpy as np
from flask  import Flask ,request,render_template
import pickle

app=Flask(__name__)
with open('saved_steps.pkl','rb') as file:
    data=pickle.load(file)
    
regressor_loaded=data['model']
le_country=data['le_country']
le_education=data['le_education']
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features=[x for x in request.form.values()]
    a=np.array([int_features])
    a[:,0]=le_country.transform(a[:,0])
    a[:,1]=le_education.transform(a[:,1])
    a=a.astype(float)
    output=regressor_loaded.predict(a)
    
    
    
    
    return render_template('index.html', prediction_Salary='Salary=${}'.format(output[0]))



if __name__=="__main__":
    app.run(debug=True)


