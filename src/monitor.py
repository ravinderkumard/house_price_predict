import requests
import json
import time
from datetime import datetime
import pandas as pd

class APIMonitor:
    def __init__(self, base_url):
        self.base_url = base_url
        self.logs = []
    
    def check_health(self):
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return {
                "timestamp": datetime.now().isoformat(),
                "endpoint": "health",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "healthy": response.status_code == 200
            }
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "endpoint": "health",
                "status_code": None,
                "response_time": None,
                "healthy": False,
                "error": str(e)
            }
    
    def test_prediction(self):
        test_data = {
            "square_feet": 2000,
            "num_bedrooms": 3,
            "num_bathrooms": 2,
            "year_built": 2010,
            "location_quality": 7
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/predict",
                json=test_data,
                timeout=10
            )
            response_time = time.time() - start_time
            
            return {
                "timestamp": datetime.now().isoformat(),
                "endpoint": "predict",
                "status_code": response.status_code,
                "response_time": response_time,
                "prediction_successful": response.status_code == 200
            }
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "endpoint": "predict",
                "status_code": None,
                "response_time": None,
                "prediction_successful": False,
                "error": str(e)
            }
    
    def run_monitoring(self, duration_minutes=1, interval_seconds=10):
        """Run monitoring for specified duration"""
        print(f"Starting monitoring for {duration_minutes} minutes...")
        
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            health_check = self.check_health()
            prediction_test = self.test_prediction()
            
            self.logs.append(health_check)
            self.logs.append(prediction_test)
            
            print(f"[{health_check['timestamp']}] Health: {health_check['healthy']}")
            print(f"[{prediction_test['timestamp']}] Prediction: {prediction_test['prediction_successful']}")
            
            time.sleep(interval_seconds)
    
    def generate_report(self):
        df = pd.DataFrame(self.logs)
        if len(df) > 0:
            print("\n=== Monitoring Report ===")
            print(f"Total checks: {len(df)}")
            print(f"Health check success rate: {df[df['endpoint'] == 'health']['healthy'].mean():.2%}")
            print(f"Prediction success rate: {df[df['endpoint'] == 'predict']['prediction_successful'].mean():.2%}")
            print(f"Average response time: {df['response_time'].mean():.3f}s")
            
            # Save report
            df.to_csv('monitoring_report.csv', index=False)
            print("Report saved to monitoring_report.csv")

if __name__ == "__main__":
    # Monitor local API
    monitor = APIMonitor("http://localhost:8000")
    
    # Run for 1 minute, checking every 10 seconds
    monitor.run_monitoring(duration_minutes=1, interval_seconds=10)
    
    # Generate report
    monitor.generate_report()