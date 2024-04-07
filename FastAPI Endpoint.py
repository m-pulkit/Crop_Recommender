# 1. Import libraries
import uvicorn
from fastapi import FastAPI
from CropEnvironment import CropEnvironment
import pickle

# 2. Create app and classifier objects
app = FastAPI()
pickle_in = open('RF_classifier.pkl', 'rb')
Crop_Mappings = pickle.load(pickle_in)
RF_classifier = pickle.load(pickle_in)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, stranger'}


# 4. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the Crop prediction with confidence
@app.post('/predict')
def predict_crop(data: CropEnvironment):
    data = data.model_dump()
    print(data)
    # print('Hello')

    N = data['N']
    P = data['P']
    K = data['K']
    temperature = data['temperature']
    humidity = data['humidity']
    ph = data['ph']
    rainfall = data['rainfall']

    prediction = RF_classifier.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    crop = [i for i in Crop_Mappings if Crop_Mappings[i] == prediction]
    print(f'For the conditions entered, the crop recommended by our model is {crop[0]}')

    return {
        'For the conditions entered, the crop recommended by our model is ': crop[0]
    }


# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


'''
input A, output 1
{
  
}

input B, output 0
{
  
}
'''

# Hosted (on Render) WebApp link: https://bank-churn-fastapi.onrender.com
