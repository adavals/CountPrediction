from fastapi import FastAPI
from pydantic import BaseModel
from api.model.model import predict

app = FastAPI()

class DateIn(BaseModel):
    dateToPredict: str

class ReceiptCountPrediction(BaseModel):
    receiptCountPrediction: int

@app.get("/")
def home():
    return {"status_check": "OK"}

#@app.post("/predict/{date_to_predict}", response_model = ReceiptCountPrediction)
@app.post("/predict", response_model = ReceiptCountPrediction)
def predict_post(payload: DateIn):
    receiptCountPrediction = predict(payload.dateToPredict)
    return {"receiptCountPrediction": receiptCountPrediction}


