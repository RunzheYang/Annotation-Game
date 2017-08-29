import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

f = open("log.json", 'r')
records = json.load(f)
print records