from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data_dir_raw = Path('../data')

HDData = pd.read_csv('~/Documents/UCIHeartDisease/HeartDiseaseProject/data/cardio_train.csv', sep=';')
