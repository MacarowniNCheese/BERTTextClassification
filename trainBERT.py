from HierarchicalBERT.BERTSentenceClassification import BSC
from BlurbDataset.BlurbDataset import BlurbDataset
import os

if __name__=="__main__":
    # parameters
    earlyStop = 1e50 # taking a large number will default the early stop to length of the dataloader provided
    batch_size = 16
    nbrHierachiesTrain = 3
    listModels = []
    epochs = 3
    PATH = os.path.join("HierarchicalBERT", "checkpoints") 

    # Load dataset
    data = BlurbDataset(earlyStop=earlyStop, batch_size=batch_size,
                        tokenizedDataPath="BlurbDataset") # Load tokenized data
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
        listModels[-1].saveTrainingResults(hierarchyLevel=hierarchyLevel)
        
        

     