from flask import Flask, render_template, request
#import urllib.request
from six.moves import urllib
import json
from datetime import date
import holidays
from datetime import datetime

app = Flask(__name__)

###################### Home Page #######################
@app.route('/home')
def home():
    return render_template('/home.html')
############### Home Page End #######################    

###################### Prediction Starts ########################
@app.route('/predict')
def runPred():
    return render_template('/predict.html')


@app.route('/predict', methods=['POST'])
def get_data_Pred():
    #print("inside POST")
    data = request.form    
    data_type = data['select']
    date = data['datetime']
    time = int(data['time'])
    temp = data['temp']
    wind = data['wind']
    dewpoint = data['dewpoint']
    condition = data['condition']

    #print("read form completed")
    date_value = date
    dt = datetime.strptime(date_value, '%Y-%m-%d')

    #month
    month_value = dt.month
    #Weekday
    #Monday is 0 and Sunday is 6
    if dt.weekday()==5 or dt.weekday()== 6:
        week_day = 0
    else:
        week_day = 1        
    
    #Base Hour Flag
    if time>4 and time< 22:
        Base_hour_Flag = "false"
    else:
        Base_hour_Flag = "true"        
    

    resultLR = processPred(algo="lr",Base_hour_Flag=Base_hour_Flag,condition=condition,week_day=week_day, temp=temp, wind=wind, data_type=data_type,dewpoint=dewpoint,month_value=month_value)
    resultForest = processPred(algo="forest",Base_hour_Flag=Base_hour_Flag,condition=condition,week_day=week_day, temp=temp, wind=wind, data_type=data_type,dewpoint=dewpoint,month_value=month_value)
    resultKNN = processPred(algo="knn",Base_hour_Flag=Base_hour_Flag,condition=condition,week_day=week_day, temp=temp, wind=wind, data_type=data_type,dewpoint=dewpoint,month_value=month_value)
    resultNN = processPred(algo="nn",Base_hour_Flag=Base_hour_Flag,condition=condition,week_day=week_day, temp=temp, wind=wind, data_type=data_type,dewpoint=dewpoint,month_value=month_value)    
    
    resultLR = json.loads(resultLR)
    resultForest = json.loads(resultForest)
    resultKNN = json.loads(resultKNN)
    resultNN = json.loads(resultNN)
    #print(resultLR['Results']['output1'])
    return render_template('/predict.html',wind=wind,date=date,hour=time,temp=temp,dewPoint=dewpoint,labelReg=resultLR['Results']['output1'][0]['Scored Labels'],labelForest=resultForest['Results']['output1'][0]['Scored Label Mean'],labelNN=resultNN['Results']['output1'][0]['Scored Labels'],labelKNN=resultKNN['Results']['output1'][0]['predict.model..subset.dataset..select....c.Norm_Consumption...'])


### function for calling Linear regression Azure ML API
def predict_lr(body):

    url = 'https://ussouthcentral.services.azureml.net/workspaces/5fa55a1569a747d68e18eef0239636a4/services/27608b1318e6439d81a81a0f085d8d31/execute?api-version=2.0&format=swagger'
    api_key = 'e3xAYK5EW4pMShJ7yFfGPRvrRoDfJCcV1rOLwmLkEqR50ta2OQtWYUxBr+KAAUzdsmL4nZnoisS2CDBKUPV8+g=='
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    
    req = urllib.request.Request(url, body, headers)
    #print("request ready")
    #print(body)
    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf-8")
        #print("Response ready")
        #print(result)
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        return None

### function for calling Boosted Decision Tree Prediction Azure ML API
def predict_knn(temp,dew):

    data = {
            "Inputs": {
                    "input1":
                    [
                        {
                            'Norm_Consumption': "",   
                            'Dew_PointF': dew,   
                            'TemperatureF': temp,   
                        }
                    ],
            },
        "GlobalParameters":  {
        }
    }


    body = str.encode(json.dumps(data))

    url = 'https://ussouthcentral.services.azureml.net/workspaces/5fa55a1569a747d68e18eef0239636a4/services/6229924063524e4ab5533a1f823dece5/execute?api-version=2.0&format=swagger'
    api_key = '9EoiQ5Sbg2yAxBhgVRSsd2EAx0gSYtZ3reFt80k1M1EsFENFvS5EL1GesyThNedkgMrwe4Ah2eeXEw6oVUJGHw=='
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)
    #print("request ready")
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf-8")
        #print("Response ready")
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        return None

### function for calling Decision Forest Prediction Azure ML API
def predict_forest(body):

    url = 'https://ussouthcentral.services.azureml.net/workspaces/5fa55a1569a747d68e18eef0239636a4/services/05c5473d9a58464caf02c5288d25a96a/execute?api-version=2.0&format=swagger'
    api_key = 'zJKORRmQ7sotjmL+AKjojnOWhvUuNVzrdQ6RfaHyZH1y560Qbs/pq2gydWrZfhLCror4D5Ixz9zRw2+09Vtyiw=='
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)
    #print("request ready")
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf-8")
        
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        return None

### function for calling Neural Network Prediction Azure ML API
def predict_nn(body):

    url = 'https://ussouthcentral.services.azureml.net/workspaces/5fa55a1569a747d68e18eef0239636a4/services/f545e29c747c452a9071dbcc5a7855c8/execute?api-version=2.0&format=swagger'
    api_key = '/IV+8lOT3H3W+3/2WbPF+w5Y6AWmv1C15+SuLd90AhudLNl2kOwtcpbC4LoJlXrTq7wo/s8gspN7CvH2QQdPhg=='
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)
    #print("request ready")
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf-8")
        #print("inside NN Pred")
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        return None

def processPred(**kwargs):  
    #print("inside processPred")  
    data = {
            "Inputs": {
                    "input1":
                    [
                        {
                                'type': kwargs['data_type'],   
                                'month': kwargs['month_value'],   
                                'Base_hour_Flag': kwargs['Base_hour_Flag'],   
                                'Weekday': kwargs['week_day'],   
                                'Conditions': kwargs['condition'],   
                                'Norm_Consumption': "1",   
                                'TemperatureF': kwargs['temp'],   
                                'Dew_PointF': kwargs['dewpoint'],   
                                'Wind_SpeedMPH': kwargs['wind'],

                        }
                    ],
            },
        "GlobalParameters":  {
        }
    }

    body = str.encode(json.dumps(data))
    #print(body)
    if kwargs['algo']=="lr":
        return predict_lr(body)
    elif kwargs['algo']=="knn":
        return predict_knn(kwargs['temp'],kwargs['dewpoint'])
    elif kwargs['algo']=="forest":
        return predict_forest(body)
    elif kwargs['algo']=="nn":
        return predict_nn(body)
###################### Prediction Ends ########################

###################### Classification Starts ########################
@app.route('/classify')
def run():
    return render_template('/classify.html')


@app.route('/classify', methods=['POST'])
def get_data():
    #print("inside POST")
    data = request.form    
    data_type = data['select']
    date = data['datetime']
    time = int(data['time'])
    temp = data['temp']
    humidity = data['humidity']
    dewpoint = data['dewpoint']

    #print("read form completed")
    date_value = date
    dt = datetime.strptime(date_value, '%Y-%m-%d')

    #month
    month_value = dt.month
    #Weekday
    #Monday is 0 and Sunday is 6
    if dt.weekday()==5 or dt.weekday()== 6:
        week_day = 0
    else:
        week_day = 1        
    
    #Base Hour Flag
    if time>4 and time< 22:
        Base_hour_Flag = "false"
    else:
        Base_hour_Flag = "true"        
    
    #Holiday
    us_holidays = holidays.UK(years=dt.year)
    if dt in us_holidays:
        Holiday = 1
    else:
        Holiday = 0

    resultGlm = process(algo="glm",Base_hour_Flag=Base_hour_Flag,Holiday=Holiday,week_day=week_day, temp=temp, humidity=humidity, data_type=data_type,dewpoint=dewpoint,month_value=month_value)
    resultForest = process(algo="forest",Base_hour_Flag=Base_hour_Flag,Holiday=Holiday,week_day=week_day, temp=temp, humidity=humidity, data_type=data_type,dewpoint=dewpoint,month_value=month_value)
    resultTree = process(algo="tree",Base_hour_Flag=Base_hour_Flag,Holiday=Holiday,week_day=week_day, temp=temp, humidity=humidity, data_type=data_type,dewpoint=dewpoint,month_value=month_value)
    resultNN = process(algo="nn",Base_hour_Flag=Base_hour_Flag,Holiday=Holiday,week_day=week_day, temp=temp, humidity=humidity, data_type=data_type,dewpoint=dewpoint,month_value=month_value)    
    
    resultGlm = json.loads(resultGlm)
    resultForest = json.loads(resultForest)
    resultTree = json.loads(resultTree)
    resultNN = json.loads(resultNN)
    
    return render_template('/classify.html',humidity=humidity,date=date,hour=time,temp=temp,dewPoint=dewpoint,labelGlm=resultGlm['Results']['output1'][0]['Scored Labels'],labelForest=resultForest['Results']['output1'][0]['Scored Labels'],labelNN=resultNN['Results']['output1'][0]['Scored Labels'],labelTree=resultTree['Results']['output1'][0]['Scored Labels'])


### function for calling Logistic regression Azure ML API
def classify_glm(body):

    url = 'https://ussouthcentral.services.azureml.net/workspaces/bde71633784a4179938464733837a692/services/3f7b835f18be4bd2b57e1f7551f3d81c/execute?api-version=2.0&format=swagger'
    api_key = '8lqL5XvT9xQSlB7IHIcl9zyHof51pIRYrmAfvkpXKEPOKbuQpE7/TeaOtxbO/KeuG3QO/U6sdoZmmpPDDGqf6A==' # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)
    #print("request ready")
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf-8")
        #print("Response ready")
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        return None

### function for calling Boosted Decision Tree Classification Azure ML API
def classify_tree(body):

    url = 'https://ussouthcentral.services.azureml.net/workspaces/bde71633784a4179938464733837a692/services/737525a092dc4b9da9330ad9a644f7a9/execute?api-version=2.0&format=swagger'
    api_key = 'YXJOUEia07zaomZoN2nblNER2RukVJWB4j3tYLwQzmlMna1iBvCWMExvuEVvyPz6djKP5oZTRN1+i1SEkc7mDA=='
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)
    #print("request ready")
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf-8")
        #print("Response ready")
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        return None

### function for calling Decision Forest Classification Azure ML API
def classify_forest(body):

    url = 'https://ussouthcentral.services.azureml.net/workspaces/bde71633784a4179938464733837a692/services/d1caa0d4070746939b630b590db1b0c3/execute?api-version=2.0&format=swagger'
    api_key = 'GiDdLdrv/hG50230FvS7H+uqZYgPr/sgEf7DSHIuODapp6u5HHr7KEbwuaAAgty9xGE0jctiKkKwjNisurIaAw=='
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)
    #print("request ready")
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf-8")
        #print("Response ready")
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        return None

### function for calling Neural Network Classification Azure ML API
def classify_nn(body):

    url = 'https://ussouthcentral.services.azureml.net/workspaces/bde71633784a4179938464733837a692/services/ac93991e01524456b44fee97ed48dc9d/execute?api-version=2.0&format=swagger'
    api_key = 'SwHZFXK7I/HU4b2FlFlSRVWJJX420ShRdX/0vTkSOBywwDLdryqYVAx9zLIM9CUp4ra1mO/AwFIa8w5wld39IA==' 
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)
    #print("request ready")
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf-8")
        #print("Response ready")
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        return None

def process(**kwargs):  
    #print("inside process")  
    data = {
            "Inputs": {
                    "input1":
                    [
                        {
                                'type': kwargs['data_type'],   
                                'month': kwargs['month_value'],   
                                'Base_hour_Flag': kwargs['Base_hour_Flag'],   
                                'Weekday': kwargs['week_day'],   
                                'Holiday': kwargs['Holiday'],   
                                'Base_Hour_Class': "",   
                                'TemperatureF': kwargs['temp'],   
                                'Dew_PointF': kwargs['dewpoint'],   
                                'Humidity': kwargs['humidity'],   
                        }
                    ],
            },
        "GlobalParameters":  {
        }
    }

    body = str.encode(json.dumps(data))
    #print("json ready")
    if kwargs['algo']=="glm":
        return classify_glm(body)
    elif kwargs['algo']=="tree":
        return classify_tree(body)
    elif kwargs['algo']=="forest":
        return classify_forest(body)
    elif kwargs['algo']=="nn":
        return classify_nn(body)
###################### Classification Ends ########################

###################### Clustering Starts ########################
@app.route('/cluster')
def runClust():
    return render_template('/cluster.html')


@app.route('/cluster', methods=['POST'])
def get_data_clust():
    #print("inside POST")
    data = request.form    
    area = data['area']
    latitude = data['latitude']
    longitude = data['longitude']
    electric = data['electric']
    heat = data['heat']


    resultKMean = processClust(algo="kMean",area=area,latitude=latitude,longitude=longitude,electric=electric,heat=heat)
    
    resultKMean = json.loads(resultKMean)
    
    return render_template('/cluster.html',area=area,latitude=latitude,longitude=longitude,electric=electric,heat=heat,labelKMean=resultKMean['Results']['output1'][0]['Assignments'])


### function for calling K-Means Clustering Azure ML API
def cluster_kMean(body):

    url = 'https://ussouthcentral.services.azureml.net/workspaces/4cf184d6c98d43debeddaef6ffe92725/services/6988ed21dd3348458d85615ae6f041dc/execute?api-version=2.0&format=swagger'
    api_key = '0T02SXPBoZ38LivdPpMWTxnNeHRZx9AMKibe9j36dQWIRgnwDRIwJSdM3cRo1NM/hlK1ZFUEJ9ZEF+vwWMxtIg==' # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}


    req = urllib.request.Request(url, body, headers)
    #print("request ready")
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf-8")
        #print("Response ready")
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        return None


def processClust(**kwargs):  
    #print("inside process")  
    data = {
            "Inputs": {
                    "input1":
                    [
                        {
                            'Column 0': "1",   
                            'vac': "",   
                            'area_floor._m.sqr': kwargs['area'],   
                            'latitudes': kwargs['latitude'],   
                            'longitudes': kwargs['longitude'],   
                            'Dist_Heating': kwargs['heat'],   
                            'elect': kwargs['electric'],   
                        }
                    ],
            },
        "GlobalParameters":  {
        }
    }

    body = str.encode(json.dumps(data))
    #print("json ready")
    if kwargs['algo']=="kMean":
        return cluster_kMean(body)
###################### Clustering Ends ########################

def main():
    print("inside main")
    app.run()


if __name__ == '__main__':
    main()