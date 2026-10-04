import os
import re

def _extract_number(text):

    cleaned = text.replace("$", "").replace(",","")
    math = re.search(r"[-+]?\d*\.\d+|\d+",cleaned)
    print("Original text:",text)
    print("Cleaned",cleaned)
    print("Matched",math)

    raise RuntimeError("DEBUG STOP: stopped after cleaned and match")