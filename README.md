# house_price_predict

## Create Virtual Environment
```bash
python -m venv venv
``` 
** Activate it: **
- On Windows: venv\Scripts\activate
- On Mac/Linux: source venv/bin/activate

## Install Dependencies
```bash
pip install -r requirements.txt
```
 No need to set specific library versions.

## Create Sample Data Set
```bash
python create_sample_data.py
```

## Training with MLflow
```bash
mlflow ui -- port 5000
```
Open another terminal and run:
```bash
python src/train.py
```

Run API locally:
```bash
python src/app.py
```
Access the API at: http://localhost:8000/

### Monitoring
Model will be monitored for 1 minute after every 10 seconds.

### Retraining
To retrain the model, run:
```bash
python src/retrain.py
```
This will retrain the model based upon the scheduled interval or periodically as configured for 5 mins or so.

