from HierarchicalBERT.BERTSentenceClassification import BSC
from BlurbDataset.BlurbDataset import BlurbDataset
import os
import pandas as pd
from utils.utils import tokenization

if __name__=="__main__":
    # Initialize parameters
    nbrHierachiesTrain = 3
    listModels = []
    PATH = os.path.join("HierarchicalBERT", "checkpoints")
    
    # Load dataset to get listLabelDict TODO: Save listLabelDict in model as well 
    data = BlurbDataset(earlyStop=20, batch_size=16,
                          tokenizedDataPath="BlurbDataset")
    
    # Text Input to classifiy
    textInput = ["A raw, powerful account of an infantryman’s life during wartime– complete with all the horrors and the heroism . . .Robert Peterson arrived in Vietnam in the fall of 1966, a young American ready to serve his country and seize his destiny. What happened in that jungle war would change his life forever. Peterson vividly relives the tense patrols in the Viet Cong-infested Central Highlands, the fierce firefights along the Cambodian border, the ambushes and enemy charges. Daily he and his fellow grunts put their lives on the line, forced to follow orders blindly from higher-ups solely interested in reaping their personal glory.Yet out of the deadly hell of Vietnam came a brotherhood–forged in blood and courage, sacrifice and survival–of men who continuously risked their lives for one another, whatever the odds. Rites of Passage is a shining testament to their valor.From the Paperback edition."]
    
    # Load models
    for hierarchyLevel in range(nbrHierachiesTrain):
        listModels.append(BSC(
            finetunedModel = True,
            finetunedModelName = f"BERTonBLURBHierarchyLevel{hierarchyLevel}.pt",
            hierarchyLevel = hierarchyLevel,
            listLabelDict = data.listLabelDict,
        ))
        
    # Preprocess text input
    inputDataframe = pd.DataFrame(textInput, columns=["body"])
        
    # Tokenize Input. All input are tokenized with the same tokenizer. Can be changed in the future.
    tokenization(inputDataframe)    
    
    # Predict
    predictions = []; translatedPredictions = []
    for model in listModels:
        predictions.append(model.inference(df = inputDataframe)) # Toeknization happens here. Inputs are currently tokenized multiple times
        translatedPredictions.append(model.translatePredictions(predictions[-1]))
        
    # Print predictions
    print(f"Predictions: {predictions}")
    print(f"Translated predictions: {translatedPredictions}")
        