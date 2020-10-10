import os
from dotenv import load_dotenv
load_dotenv()

SAVE_FILES = os.getenv('SACRED_SAVE_FILES', 'false').lower() == 'true'