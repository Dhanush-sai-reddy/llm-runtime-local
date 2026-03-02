import sys
sys.path.insert(0, '../pathway/python')

import pathway as pw
import os
from dotenv import load_dotenv

load_dotenv()

app = pw.load_yaml("app.yaml")

if __name__ == "__main__":
    pw.run(app)
