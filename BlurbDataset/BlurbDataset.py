import torch
import os
import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from math import ceil

from utils.utils import tokenization, convertDataframetoDataloader


class BlurbDataset:
    # TODO implement option to save the tensordatas and load them instead of creating them again
    def __init__(
            self, 
            earlyStop:int=1e50, 
            batch_size:int=8, 
            tokenizerModel:str="bert-base-uncased",
            tokenizedDataPath:str=""
        ):
        self.earlyStop = earlyStop
        self.batch_size = batch_size
        self.tokenizerModel = tokenizerModel
        
        if tokenizedDataPath == "":
            self.trainDF, self.testDF, self.valDF = loadBlurbData()
            
            # Get the maximum length of lists in the 'topics' column
            self.max_length = self.extractMaxNbrLabels(self.trainDF) 
            self.listLabelDict = [{} for _ in range(self.max_length)]
            
            self.isPreprocessed = False
            self.num_labels = self.extractNbrLabels(self.testDF)
            self.dataloaders = dict()
            
        else:
            self.dataloaders, self.listLabelDict = self.loadData(tokenizedDataPath)
            self.isPreprocessed = True
            self.num_labels = [len(dict) for dict in self.listLabelDict]   
            # print(f"num_labels = {self.num_labels}")     
        
    def saveData(self) -> None:
        """
        Method saves the train, val and test tensordatasets as well as the listLabelDict 
        as json file
        """
        if self.isPreprocessed == False:
            print("Data not preprocessed yet. Please preprocess data first.")
        else: 
            PATH = "BlurbDataset"
            appendix = ""
            if self.earlyStop != 1e50:
                appendix = f"EarlyStop{self.earlyStop}"
                
            torch.save(self.dataloaders["train"].dataset, 
                    os.path.join(PATH, f"preprocessedTrainBLURBDataset{appendix}.pt"))    
            torch.save(self.dataloaders["val"].dataset, 
                    os.path.join(PATH, f"preprocessedValBLURBDataset{appendix}.pt"))
            torch.save(self.dataloaders["test"].dataset, 
                    os.path.join(PATH, f"preprocessedTestBLURBDataset{appendix}.pt")) 
            with open(os.path.join(PATH, "listLabelDict.json"), "w") as f:
                json.dump(self.listLabelDict, f)
     
    def loadData(self, PATH:str="BlurbDataset") -> Tuple[Dict, Dict]:
        """
        Method loads the train, val and test files as csv files as well as the listLabelDict
        for a given value of self.earlyStop
        """
        appendix = ""
        if self.earlyStop != 1e50:
            appendix = f"EarlyStop{self.earlyStop}"
            
        loaders = {}
        datasetPATH = [
            os.path.join(PATH, f"preprocessedTrainBLURBDataset{appendix}.pt"),
            os.path.join(PATH, f"preprocessedValBLURBDataset{appendix}.pt"),
            os.path.join(PATH, f"preprocessedTestBLURBDataset{appendix}.pt")
        ]
    
        trainDataset = torch.load(datasetPATH[0])
        loaders["train"] = DataLoader(
            trainDataset, 
            sampler = RandomSampler(trainDataset),
            batch_size = self.batch_size
        )
        
        valDataset = torch.load(datasetPATH[1])
        loaders["val"] = DataLoader(
            valDataset, 
            sampler = SequentialSampler(valDataset),
            batch_size = self.batch_size
        )
        
        testDataset = torch.load(datasetPATH[2])
        loaders["test"] = DataLoader(
            testDataset, 
            sampler = SequentialSampler(testDataset),
            batch_size = self.batch_size
        )
            
        with open(os.path.join(PATH, "listLabelDict.json"), "r") as f:
            listLabelDict = json.load(f)
        
        return (loaders, listLabelDict)
             
    def trialTokenizer(self):
        # Load the tokenizer
        tokenizer = self.loadTokenizer()
        
        trailTextNr = 2 
        # Print the original sentence.
        print(' Original: ', self.trainDf["body"][trailTextNr])

        # Print the sentence split into tokens.
        print('Tokenized: ', tokenizer.tokenize(self.trainDf["body"][trailTextNr]))

        # Print the sentence mapped to token ids.
        print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.trainDf["body"][trailTextNr])))    
        
    def loadTokenizer(self) -> BertTokenizer:
        """
        Method loads the tokenizer
        :Returns:
            tokenizer (BertTokenizer): tokenizer for the BERT model
        """
        # We will be using the uncased version here i.e. the tokenization does not regarded upper or lower case 
        return BertTokenizer.from_pretrained(
                                                self.tokenizerModel, 
                                                do_lower_case=True, 
                                                truncation=True
                                            )
    
    def checkMaxLen(self) -> int:
        """
        Function rerturns the token vector length of the longest text input in the dataset
        :Args:
            textInputs (_type_): list of the textinputs in the dataset
        :Returns:
            int: length of longest text input in dataset
        """
        max_len = 0
        tokenizer = loadTokenizer()
        for text in self.trainDF["body"]:
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = tokenizer.encode(text, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
        print('Max text length: ', max_len)
        return max_len # for BlurbSet we have maxlen = 3534   
    
    def preprocessLabels(self) -> None:
        """ 
        Function encodes and fills the label matrix with -1 for the labels that are not present 
        Example: ["Sicence Fiction", "Fantasy"] -> [35, 12, -1, -1, -1]
        """  
        for df in [self.trainDF, self.valDF, self.testDF]:
            encoded_topics = []

            for _, row in df.iterrows():
                encoded_row = []

                for i, topic in enumerate(row["topics"]):
                    if topic not in self.listLabelDict[i]:
                        self.listLabelDict[i][topic] = len(self.listLabelDict[i])

                    encoded_row.append(self.listLabelDict[i][topic])

                if len(encoded_row) < self.max_length:
                    encoded_row.extend([-1] * (self.max_length - len(encoded_row)))  # Padding with -1 for shorter lists

                # Does not seem to be needed. Although, this change did not fix the RUntimeError: CUDA Error
                # Add 1 to all labels to avoid 0 labels. This is needed for the loss function
                encoded_topics.append(np.array(encoded_row))
            
            df["labels"] = encoded_topics
                      
    def tokenization(self) -> torch.tensor:
        """
        Function tokenize the body (textinput) of all the dataframes and maps the 
        tokens to thier word IDs
        Args:
            textInputs (np.array): untokenized text inputs
        """
        for i, df in enumerate([self.trainDF, self.valDF, self.testDF]):
            
            # Early stop for testing purposes
            nbrExamples = min([self.earlyStop,len(df["body"])])
            if i != 0: # For validation and test set we arbitrarely only take half the examples
                nbrExamples = int(nbrExamples/2)
            
            tokenization(df, self.tokenizerModel, nbrExamples)
                     
    def extractMaxNbrLabels(self, df:pd.DataFrame) -> int:
        """
        Get the maximum length of lists in the 'topics' column
        Args:
            df (pd.dataframe): dataframe input

        Returns:
            int: length of the longest list in the 'topics' column
        """
        return max(df['topics'].apply(len))

    def convertToDataloader(self) -> None: 
        """
        Function converts the train, validation and test dataframe to dataloaders
        """    
        
        loaders = []
        for i, df in enumerate([self.trainDF, self.valDF, self.testDF]):
            
            sampler = RandomSampler
            nbrExamples = min([self.earlyStop,len(df["body"])])
            # For validation and test set I arbitrarely only take half the examples 
            if i != 0: 
                nbrExamples = int(nbrExamples/2)
                sampler = SequentialSampler
                
            loaders.append(convertDataframetoDataloader(
                df, 
                nbrExamples = nbrExamples,
                batch_size = self.batch_size,
                sampler = sampler
            ))
        
        self.dataloaders = {
            "train" : loaders[0], 
            "val" : loaders[1], 
            "test" : loaders[2]
            }

    def extractNbrLabels(self, df:pd.DataFrame) -> int:
        """
        Method extracts the number of labels at each hierarchy level in the dataset
        :Args:
            df (pandas.DataFrame): dataframe containing the labels
        :Returns:
            list containg number of labels at each hierarchy level in the dataset
        """
        
        distinct_labels = []
        
        # Find the maximum length of sublists
        max_length = max(len(sublist) for sublist in df["topics"])
        
        # Iterate over each level (index) in the lists
        for i in range(max_length):
            labels = set()
            
            # Iterate over each sublist and add labels to the set
            for sublist in df["topics"]:
                if i < len(sublist):
                    labels.add(sublist[i])
            
            distinct_labels.append(len(labels))
        
        return distinct_labels
    
    def prepareData(self) -> None: 
        """
        Function that prepares the blurb data for the BERT model
        Args:
            earlyStop (int): number of examples to use for testing purposes
            batch_size (int): batch size for the dataloaders
        :Returns:
            tuple of dataloader of the Blurb data
        """
        if self.isPreprocessed == False:
            # Preprocess labesl. Example: ["Sicence Fiction", "Fantasy"] -> [35, 12, -1, -1, -1]
            self.preprocessLabels()

            # Add tokenized topics and attention mask to dataframe
            self.tokenization()

            # Create the DataLoaders for test, validation and training set
            self.convertToDataloader()
            
            self.isPreprocessed = True
        
        else: 
            print("Data already preprocessed.")
    
class myDataLoader(DataLoader):
    # Wrapper class for the DataLoader class. Needed to fix the __len__ function. Fixed this issue by eleminiating bug in convertToDataloader function
    # CURRENTLY NOT USED
    def __init__(self, tensorDataset:TensorDataset, **args):
        self.tensorDataset = tensorDataset
        super().__init__(self.tensorDataset, **args)

    def __len__(self):
        return ceil(len(self.tensorDataset)/self.batch_size)

    def __getitem__(self, index):
        if index == len(self)-1:
            return self.tensorDataset[index*self.batch_size:]
        else:
            return self.tensorDataset[index*self.batch_size:(index+1)*self.batch_size]

def loadBlurbDataset(file_name:str="BlurbGenreCollection_EN_train.txt", path:str="./BlurbDataset/") -> pd.DataFrame: 
    """
    Extracts the infromation of one .txt files and returns a pandas Dataframe
    :Args:
        file_name (str, optional)

    :Returns:
        pd.DataFrame
    """

    # Path to the text file
    text_file = os.path.join(path,file_name)

    # Read the text file
    with open(text_file, 'r', encoding="utf-8", errors="ignore") as file:
        data = file.read()

    # Define regular expressions to extract book information
    book_pattern = re.compile(r'<book.*?>.*?</book>', re.DOTALL)
    title_pattern = re.compile(r'<title>(.*?)</title>')
    body_pattern = re.compile(r'<body>(.*?)</body>')
    author_pattern = re.compile(r'<author>(.*?)</author>')
    published_pattern = re.compile(r'<published>(.*?)</published>')
    topic_pattern = re.compile(r'<d\d+>(.*?)</d\d+>') # could drop the "+" here as there is only ever one digit in the dataset

    # Extract book information using regular expressions
    books = []
    for book_match in book_pattern.findall(data):
        book_info = {}

        title_match = title_pattern.search(book_match)
        if title_match:
            book_info['title'] = title_match.group(1)

        body_match = body_pattern.search(book_match)
        if body_match:
            book_info['body'] = body_match.group(1)

        author_match = author_pattern.search(book_match)
        if author_match:
            book_info['author'] = author_match.group(1)

        published_match = published_pattern.search(book_match)
        if published_match:
            book_info['published'] = published_match.group(1)

        topic_matches = topic_pattern.findall(book_match)
        if topic_matches:
            book_info['topics'] = topic_matches

        books.append(book_info)

    # Return pandas DataFrame from the extracted book information
    return pd.DataFrame(books)

def loadBlurbData() -> tuple:
    """
    Loads the entire blurb dataset 
    Returns:
        tuple containing train, test and validation as pandas.DataFrame
    """
    trainSet = loadBlurbDataset("BlurbGenreCollection_EN_train.txt")
    testSet = loadBlurbDataset("BlurbGenreCollection_EN_test.txt")
    valSet = loadBlurbDataset("BlurbGenreCollection_EN_dev.txt")
    return (trainSet, testSet, valSet)
    
def saveBlurbCSV(file_name:str,dataframe:pd.DataFrame,path:str="BlurbDataset") -> None:
    """
    Save a dataframe as a CSV file
    :RETRUN: 
        None
    :Args:
        file_name (str)
        dataframe (pd.DataFrame)
        path (str, optional)
    """
    try:
        dataframe.to_csv(os.path.join(path,file_name))
    except Exception as e:
        print(f"Problems saving the dataframe as CSV. Exception thrown: {e}")
    
def loadBlurbCSV(file_name:str,path:str="BlurbDataset") -> pd.DataFrame:
    """
    Load a CSV file and put in to pd.DataFrame object 
    :RETRUN: 
        None
    :Args:
        file_name (str)
        path (str, optional)
    """
    try:
        dataframe = pd.read_csv(os.path.join(path,file_name))
        return dataframe
    except Exception as e:
        print(f"Problems loading the CSV as dataframe. Exception thrown: {e}")
        return None