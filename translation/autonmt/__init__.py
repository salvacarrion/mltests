import logging
import os.path
from pathlib import Path

LOGS_PATH = "logs"

# Create logs path
path = Path(LOGS_PATH)
path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=os.path.join(path, 'logger.log'), filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logging.info('########## LOGGER INITIATED ##########')
