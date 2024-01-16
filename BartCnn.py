"""
this is the main file for NumEval task 3 subtask 1. The Goal of this script is to generate embeddings using a bart
model then traing a CNN on thoes embedings and coresponding loabels. 
"""

from huggingBart import BartModel
import json
import numpy as np
from keras.models import Sequential, save_model, load_model
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.model_selection import train_test_split

modelName = "facebook/bart-base" #this is the name of the hugging face model
dataPath = "Train_Numerical_Reasoning.json" #this is the path to the training data
data = None #this is the data. should be a dictionary
with open(dataPath, "r") as dataFile: #get data out of .json
    data = json.load(dataFile)
bartModel = BartModel(modelName) #inintalize bart model

def bartProcessing(data:dict, numSamples:int): 
    """
    This function takes in the data and and creats a dictionary with the formated contents.
    Input: data = dictionary train_numerical_reasoning.json, numSamples: the number of samples outputted
    output: dict
    X:embeddings of data, 
    y:tokenized lables, 
    max_words:the max number of words in any given body of text, 
    max_sequence_length:the max length of a sequence of the tokenized data
    max_label_length:max length of a label sequence
    num_classes:the number of unique classes/answesr/lables}
    """
    retData = {"X":[], "y":[], "max_words":0, "max_sequence_length":0, "num_classes":0, "max_label_length":0}
    tests = [] #this is a list of all texts in the data
    lables = [] #list of lables coresponding to each text entry
    sampleCount = 0
    for sample in data:
        text = str(sample["news"])
        label = str(sample["ans"])
        if sampleCount > numSamples: #break the loop when num samples reached
            break
        sampleCount += 1


    return retData




