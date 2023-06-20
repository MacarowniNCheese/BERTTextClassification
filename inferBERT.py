from HierarchicalBERT.BERTSentenceClassification import BSC
from BlurbDataset.BlurbDataset import BlurbDataset
import os
import pandas as pd

if __name__=="__main__":
    # Initialize parameters
    nbrHierachiesTrain = 3
    listModels = []
    PATH = os.path.join("HierarchicalBERT", "checkpoints")
    
    # Load dataset to get listLabelDict TODO: Save listLabelDict in model as well 
    data = BlurbDataset(earlyStop=20, batch_size=16,
                          tokenizedDataPath="BlurbDataset")
    
    # Text Input to classifiy
    textInput = ["This is a test sentence", "This is another test sentence"]
    
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
        
    # Predict
    predictions = []; translatedPredictions = []
    for model in listModels:
        predictions.append(model.inference(df = inputDataframe))
        translatedPredictions.append(model.translatePredictions(predictions[-1]))
        
    # Print predictions
    print(f"Predictions: {predictions}")
    print(f"Translated predictions: {translatedPredictions}")
        