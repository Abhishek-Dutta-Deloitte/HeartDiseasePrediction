import logging
import os
import sys
from datetime import datetime


# Get the directory path for current script
current_directory =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# print(current_directory)
# Set the log file path
logs_directory = os.path.join(current_directory, "logs")

# Create Directory if does not exisots
os.makedirs(logs_directory, exist_ok=True)

# Define log file name and path
log_file = f"HEART_DISEASE_DETECTION_LOG {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log"
log_path = os.path.join(logs_directory, log_file)

#Custome logger
logger = logging.getLogger("HEART_DISEASE_DETECTION_LOG")
logger.setLevel(logging.DEBUG)

#Create Handler
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(log_path)

c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

#Setting the Format
c_format = logging.Formatter("%(asctime)s - %(name)s - %(module)s - %(levelname)s : %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
f_format = logging.Formatter("%(asctime)s - %(name)s - %(module)s - %(levelname)s : %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

#Add Handler
logger.addHandler(c_handler)
logger.addHandler(f_handler)
