import schedule
import time
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retraining.log'),
        logging.StreamHandler()
    ]
)

def retrain_job():
    """Job to retrain the model"""
    logging.info("Starting scheduled retraining...")
    
    try:
        # Run training script
        result = subprocess.run(
            ['python', 'src/train.py'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logging.info("Retraining completed successfully")
            logging.info(f"Output: {result.stdout}")
        else:
            logging.error(f"Retraining failed: {result.stderr}")
            
    except Exception as e:
        logging.error(f"Error during retraining: {str(e)}")

def main():
    """Main function to schedule retraining"""
    logging.info("Starting retraining scheduler")
    
    # Schedule retraining (example: weekly on Sunday at 2 AM)
    schedule.every().sunday.at("02:00").do(retrain_job)
    
    # For testing, run every 5 minutes
    # schedule.every(5).minutes.do(retrain_job)
    
    logging.info("Scheduler started. Press Ctrl+C to stop.")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user")