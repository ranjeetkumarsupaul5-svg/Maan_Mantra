import joblib
import pandas as pd
from pydantic import BaseModel,Field
from fastapi import FastAPI
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
model = joblib.load('Mental_Health_Score.pkl')

class StudentData(BaseModel):
    Age: int = Field(..., ge=0, le=100, description="Age of the student (0-100)")
    Gender: Literal["Male", "Female", "Other"]
    Country: str
    Academic_Level: Literal["High School", "Undergraduate", "Graduate"]
    Most_Used_Platform:Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter',
       'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp',
       'WeChat']
    Purpose_Of_Use: Literal['Networking', 'Education', 'Entertainment', 'News']
    Avg_Daily_Usage_Hours: float = Field(..., ge=0, le=24, description="Average daily usage hours (0-24)")
    Daily_Unlocks: int=Field(..., ge=0,description="Number of times the phone is unlocked daily (0-1000)")
    Study_Hours: float = Field(..., ge=0, le=24, description="Number of study hours per day (0-24)")
    Physical_Activity_Hours: float = Field(..., ge=0, le=24, description="Number of physical activity hours per day (0-24)")
    Sleep_Hours_Per_Night: float = Field(..., ge=0, le=24, description="Number of sleep hours per night (0-24)")
    Stress_Level: Literal["Low", "Medium", "High","Very High"]
    #describe what we send back to the user
class PredictionResponse(BaseModel):
        predicted_mental_health_score: float 

@app.get("/")

def read_root():
    return {"message": "Welcome to the Mental Health Score Prediction API!"}
top_countries =[
'Other',
 'India',
 'USA',
 'Canada',
 'Australia',
 'UK',
 'Germany',
 'Mexico',
 'Turkey',
 'France']

@app.post("/predict",response_model=PredictionResponse)
def predict(data: StudentData):
   country_group = 'Other' if data.Country not in top_countries else data.Country
   input_row = pd.DataFrame([{ 
       'Age': data.Age,
       'Gender': data.Gender,
       'group_countries': country_group,
       'Academic_Level': data.Academic_Level,
       'Most_Used_Platform': data.Most_Used_Platform,
       'Purpose_Of_Use': data.Purpose_Of_Use,
       'Avg_Daily_Usage_Hours': data.Avg_Daily_Usage_Hours,
       'Daily_Unlocks': data.Daily_Unlocks,
       'Study_Hours': data.Study_Hours,
       'Physical_Activity_Hours': data.Physical_Activity_Hours,
       'Sleep_Hours_Per_Night': data.Sleep_Hours_Per_Night,
       'Stress_Level': data.Stress_Level
   }])
   prediction = model.predict(input_row)[0]
   return PredictionResponse(predicted_mental_health_score=round(prediction, 2))
   