#!/usr/local/bin/python3
import sklearn
import sklearn.model_selection as skMS
import sklearn.preprocessing as skPP
import numpy as np
import pandas as pd

def createDataframe():
    """
    Reads csv file containing names and ethnicities
    Returns:
        pandas dataframe with cols [['name', 'ethnicity']]
    """
    return pd.read_csv("names.txt")

def encodeNames(df):
    """
    Converts Nx2 dataframe where N is samples into NxD+1 dataframe
    where D columns are the individual character encoding of each name with max
    name length zero padding and 1 column is the label encoding

    Input:
        dataframe: pandas dataframe with cols [['name', 'ethnicity']]

    Returns:
        encodedDataFrame: pandas dataframe with dims NxD with all characters encoded
        labelDataFrame:   pandas dataframe with dims Nx1 with labels
        nameEncoder:  scikit name encoding model
        labelEncoder: scikit label encoding model
        labelBinarizer: Transforms ints to one hot encodings
        maxCharCount: number of chars per name with padding
    """

    # Get all characters that need to be encoded
    # Get max name character count
    charSet = set()
    maxCharCount = 0
    for name in df['name']:
        for char in name:
            charSet.add(char)
        if len(name) > maxCharCount:
            maxCharCount = len(name)

    chars = list(charSet)

    # Name Encoding
    nameEncoder  = skPP.LabelEncoder() # char based
    nameEncoder.fit(chars)

    # Encode names and expand to list of size maxCharCount
    # Can probably do this using pandas apply more efficiently
    encodedNames = -1*np.ones([df.shape[0], maxCharCount])

    for i,name in enumerate(df['name']):
        encoding = nameEncoder.transform(list(name))
        encodedNames[i,0:len(encoding)] = encoding

    encodedDF = pd.DataFrame(encodedNames)
    encodedDataFrame = pd.concat([df,encodedDF], axis=1)

    # Label encoding
    labelEncoder = skPP.LabelEncoder() # label based
    labelEncoder.fit(encodedDataFrame['ethnicity'])
    encodedDataFrame['ethnicity'] = labelEncoder.transform(encodedDataFrame['ethnicity'])

    # One hot encoding for labels
    labelBinarizer = skPP.LabelBinarizer()
    labelBinarizer.fit(range(max(encodedDataFrame['ethnicity'])+1))

    # Final dataframes
    labelDataFrame = encodedDataFrame[['ethnicity']]
    encodedDataFrame.drop(['ethnicity', 'name'], axis=1, inplace=True)

    return [encodedDataFrame, labelDataFrame, nameEncoder, labelEncoder, labelBinarizer, maxCharCount]

def create_train_test_set(data, labels, test_size):
    """
    Splits dataframe into train/test set

    Inputs:
        data: encoded dataframe containing encoded name chars
        labels: encoded label dataframe
        test_size: percentage of input data set to use for test set

    Returns:
        data_train: Subset of data set for training
        data_test : Subset of data set for test
        label_train: Subset of label set for training
        label_test:  Subset of label set for testing
    """
    data_train, data_test, label_train, label_test = skMS.train_test_split(data, labels, test_size=test_size)
    return [data_train, data_test, label_train, label_test]
