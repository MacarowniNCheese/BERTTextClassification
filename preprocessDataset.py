from blurb_dataset.blurb_dataset import BlurbDataset    
import boto3
import os


if __name__=="__main__":
    # parameters
    earlyStop = 1e50
    batch_size = 16
    epochs = 3
    hierarchyLevel = 0


    data = BlurbDataset(earlyStop=earlyStop, batch_size=batch_size)
    data.prepareData()
    data.saveData()
    
    # Save to Minio
    AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
    AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

    def generateFileNames():
        """ 
        Generates the file names of the files to be uploaded
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

    s3 = boto3.resource('s3',
        endpoint_url='http://10.240.5.123:9099',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=None,
        config=boto3.session.Config(signature_version='s3v4'),
        verify=False
    )

    bucket_name = "idoml" # define your bucket name
    fileNames = generateFileNames() # define your model file name. THIS NEEDS TO BE CODED PROPERLY! COULD ALSO BE ADDED TO THE CLASS DIRECTLY
    minio_path = "bert-training-v1.0.0/preprocessed_data/"

    for fileName in fileNames:
        s3.Bucket(bucket_name).upload_file(fileName, minio_path + fileName)