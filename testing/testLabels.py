from BlurbDataset.BlurbDataset import BlurbDataset
import torch

if __name__=="__main__":
    # Initialize Parameters
    hierarchyLevel = 0
    earlyStop = 1e50
    batch_size = 16
    
    # Load dataset
    data = BlurbDataset(earlyStop=earlyStop, batch_size=batch_size,
                        tokenizedDataPath="BlurbDataset") # Load tokenized data
    data.prepareData() # This should just output that the data does not need to be prepocessed anymore
    
    for loader in data.dataloaders.values():
        for step, batch in enumerate(loader):
            # Load everyhting in the batch and excludes datapoints with label -1
            b_labels = batch[2][:,hierarchyLevel]#.to(self.device)
            b_input_ids = batch[0][b_labels!=-1,:].to("cpu")
            b_input_mask = batch[1][b_labels!=-1,:].to("cpu")
            b_labels = torch.tensor(b_labels[b_labels!=-1],dtype=torch.long).to("cpu")
            
            assert (b_labels > -1).all() , f"Labels must be bigger than -1! Found a false label in {b_labels} at step {step}"
            assert (b_labels < data.num_labels[hierarchyLevel]).all() , f"Labels must be smaller than number of classes which is {data.num_labels[hierarchyLevel]}! Found a false label in {b_labels} at step {step}"

    print("Test passed.")