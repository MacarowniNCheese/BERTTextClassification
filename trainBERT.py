from HierarchicalBERT.BERTSentenceClassification import BSC
from blurb_dataset.blurb_dataset import BlurbDataset
import os    
import boto3

if __name__=="__main__":
    

    
    # parameters
    earlyStop = 20 # taking a large number will default the early stop to length of the dataloader provided
    batch_size = 16
    nbrHierachiesTrain = 1
    listModels = []
    epochs = 3

    # Parameter Initialization
    AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
    AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
    earlyStop = 1e50 # need to enter this to specify which version of preprocessed datasets should be accessed
    s3 = boto3.resource('s3',
        endpoint_url='http://10.240.5.123:9099',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=None,
        config=boto3.session.Config(signature_version='s3v4'),
        verify=False
    )

    # Helper function
    def generateFileNames():
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

    # Set the correct pathing
    bucket_name = "idoml" # define your bucket name
    fileNames = generateFileNames() # define your model file name. THIS NEEDS TO BE CODED PROPERLY! COULD ALSO BE ADDED TO THE CLASS DIRECTLY
    minio_path = "bert-training-v1.0.0/preprocessed_data/"

    # Download and save files locally
    for fileName in fileNames:
        print(f"Downloading {fileName} at {minio_path + fileName}")
        # the second argument tells the function where to store the data locally
        # I chose the same folder structure locally as remotely -> Can use the same variable locally and remotely
        s3.Bucket(bucket_name).download_file(minio_path + fileName, fileName)  
    
    
    
    
    


    # Load dataset
    data = BlurbDataset(earlyStop=earlyStop, batch_size=batch_size,
                        tokenizedDataPath="blurb_dataset") # Load tokenized data
    data.prepareData() # This should just output that the data does not need to be prepocessed anymore
    
    
    # Load and train models for each hierarchy level
    for hierarchyLevel in range(nbrHierachiesTrain):
        print("======================================")
        print(f"Training model for hierarchy level {hierarchyLevel}")
        listModels.append(BSC(
            dataloaders = data.dataloaders,
            epochs = epochs,
            num_labels = data.num_labels,
            hierarchyLevel = hierarchyLevel,
            listLabelDict = data.listLabelDict
        ))
        listModels[-1].train()
        listModels[-1].saveModel(f"BERTonBLURBHierarchyLevel{hierarchyLevel}.pt")
        predictions = listModels[-1].test()
        
        

        model_path = os.path.join("HierarchicalBERT","checkpoints", f"BERTonBLURBHierarchyLevel{hierarchyLevel}.pt", "pytorch_model.bin")
        s3.Bucket(bucket_name).upload_file(model_path, minio_path + model_path)
        
        model_path = os.path.join("HierarchicalBERT","checkpoints", f"BERTonBLURBHierarchyLevel{hierarchyLevel}.pt", "config.json")
        s3.Bucket(bucket_name).upload_file(model_path, minio_path + model_path)
        
        
        listModels[-1].saveTrainingResults(hierarchyLevel=hierarchyLevel)
        
        res_path = os.path.join("HierarchicalBERT","trainingResults")
        counter = 0
        pathAndName = os.path.join(res_path, f"trainingResultsHierarchy{hierarchyLevel}Run{counter}.json")

        s3.Bucket(bucket_name).upload_file(pathAndName, minio_path + pathAndName)