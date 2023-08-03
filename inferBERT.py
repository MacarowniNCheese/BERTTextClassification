from HierarchicalBERT.BERTSentenceClassification import BSC
from blurb_dataset.blurb_dataset import BlurbDataset
import os
import pandas as pd
from utils.utils import tokenization
import boto3

AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
s3 = boto3.resource('s3',
    endpoint_url='http://10.240.5.123:9099',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=None,
    config=boto3.session.Config(signature_version='s3v4'),
    verify=False
)
bucket_name = "idoml" # define your bucket name

minio_path = "bert-training-v1.0.0/preprocessed_data/"
def generateFileNames(earlyStop):
    """ 
    Generates the file names of the files to be downloaded
    """
    PATH = "blurb_dataset" 
    appendix = ""
    if earlyStop != 1e50:
        appendix = f"EarlyStop{earlyStop}"

    pathNames = []
    pathNames.append(os.path.join(PATH, f"preprocessedTrainBLURBDataset{appendix}.pt"))
    pathNames.append(os.path.join(PATH, f"preprocessedTestBLURBDataset{appendix}.pt"))
    pathNames.append(os.path.join(PATH, f"preprocessedValBLURBDataset{appendix}.pt"))
    pathNames.append(os.path.join(PATH, "listLabelDict.json"))
    return pathNames


if __name__=="__main__":
    # Initialize parameters
    nbrHierachiesTrain = 1
    listModels = []
    PATH = os.path.join("HierarchicalBERT", "checkpoints")

    earlyStop = 1e50
    # Download and save files locally
    fileNames = generateFileNames(earlyStop) # define your model file name. THIS NEEDS TO BE CODED PROPERLY! COULD ALSO BE ADDED TO THE CLASS DIRECTLY
    for fileName in fileNames:
        print(f"Downloading {fileName} at {minio_path + fileName}")
        # the second argument tells the function where to store the data locally
        # I chose the same folder structure locally as remotely -> Can use the same variable locally and remotely
        s3.Bucket(bucket_name).download_file(minio_path + fileName, fileName)  
    
    # Load dataset to get listLabelDict TODO: Save listLabelDict in model as well 
    data = BlurbDataset(earlyStop=earlyStop, batch_size=16,
                          tokenizedDataPath="blurb_dataset")
    
    # Text Input to classifiy
    textInput = ["A raw, powerful account of an infantryman’s life during wartime– complete with all the horrors and the heroism . . .Robert Peterson arrived in Vietnam in the fall of 1966, a young American ready to serve his country and seize his destiny. What happened in that jungle war would change his life forever. Peterson vividly relives the tense patrols in the Viet Cong-infested Central Highlands, the fierce firefights along the Cambodian border, the ambushes and enemy charges. Daily he and his fellow grunts put their lives on the line, forced to follow orders blindly from higher-ups solely interested in reaping their personal glory.Yet out of the deadly hell of Vietnam came a brotherhood–forged in blood and courage, sacrifice and survival–of men who continuously risked their lives for one another, whatever the odds. Rites of Passage is a shining testament to their valor.From the Paperback edition."]
    
    # Load models
    for hierarchyLevel in range(nbrHierachiesTrain):
        dir_path = os.path.join("HierarchicalBERT","checkpoints", f"BERTonBLURBHierarchyLevel{hierarchyLevel}.pt")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        model_path = os.path.join(dir_path, "pytorch_model.bin")
        s3.Bucket(bucket_name).download_file(minio_path + model_path, model_path)
        model_path = os.path.join(dir_path, "config.json")
        s3.Bucket(bucket_name).download_file(minio_path + model_path, model_path)
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
        