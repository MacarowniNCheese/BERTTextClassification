from BlurbDataset.BlurbDataset import BlurbDataset


if __name__=="__main__":
    # parameters
    earlyStop = 1e50
    batch_size = 16
    epochs = 3
    hierarchyLevel = 0


    data = BlurbDataset(earlyStop=earlyStop, batch_size=batch_size)
    data.prepareData()
    data.saveData()