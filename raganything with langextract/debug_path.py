import langextract
import os

with open("debug_output.txt", "w") as f:
    try:
        f.write(f"FILE: {langextract.__file__}\n")
    except Exception as e:
        f.write(f"FILE ERROR: {e}\n")
    
    try:
        f.write(f"PATH: {langextract.__path__}\n")
    except Exception as e:
        f.write(f"PATH ERROR: {e}\n")
        
    try:
        f.write(f"DIR: {dir(langextract)}\n")
    except Exception as e:
        f.write(f"DIR ERROR: {e}\n")
