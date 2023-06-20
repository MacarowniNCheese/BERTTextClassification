from BlurbDataset.BlurbDataset import BlurbDataset


if __name__=="__main__":
    # parameters
    earlyStop = 20
    batch_size = 16
    epochs = 3
    hierarchyLevel = 0


    data = BlurbDataset(batch_size=batch_size)
    data.prepareData()
    data.saveData()