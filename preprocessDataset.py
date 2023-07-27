from blurb_dataset.blurb_dataset import BlurbDataset


if __name__=="__main__":
    # parameters
    earlyStop = 20
    batch_size = 16
    epochs = 3
    hierarchyLevel = 0


    data = BlurbDataset(earlyStop=earlyStop, batch_size=batch_size)
    data.prepareData()
    data.saveData()