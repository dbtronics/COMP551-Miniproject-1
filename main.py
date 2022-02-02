from scipy.io import arff
import pandas as pd
import numpy as np


def main():
    data = arff.loadarff('messidor_features.arff')
    df = pd.DataFrame(data[0])
    df.replace({'?' : np.nan}).dropna()
    print(df)
    # basic statistics df
    print(df.describe(include='all'))
    # Document messidor features.arff above
    # Document hepatitis below
    df2 = pd.read_csv('hepatitis1.csv', names=["class", "age", "sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia", "liver_big", "liver_firm", "spleen_palpable", "spiders", "ascites", "varices", "bilirubin", "alk_phosphate", "sgot", "albumin", "protime", "histology"])
    print(df2.replace({'?': np.nan}).dropna())
    # basic statistics df2
    print(df2.describe(include='all'))

# Diabetic Retinopathy Debrecen Data Set Data Set
if __name__ == '__main__':
    main()
