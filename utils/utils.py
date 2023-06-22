import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

def flat_accuracy(preds, labels):
    """
    Function calculates the accuracy of the predictions
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # print(pred_flat, labels_flat)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
 
def convertDataframetoDataloader(
        df:pd.DataFrame, 
        sampler:torch.utils.data.sampler.SubsetRandomSampler=SequentialSampler,
        batch_size:int=16,
        nbrExamples:int=0,
        includeLabels:bool=False,
    ) -> torch.utils.data.DataLoader:
            
    """
    Method converts pd.dataframe into a torch.utils.data.dataloader
    :Args:
        df (pd.DataFrame): dataframe containing the data
        sampler (torch.utils.data.sampler.SubsetRandomSampler): sampler to use
        batch_size (int): batch size
        nbrExamples (int): number of examples to load      
    :Returns:
        dataloader (torch.utils.data.DataLoader): dataloader containing the data
    """
    if nbrExamples == 0: 
        nbrExamples = len(df["body"])
    
    # Convert Dataframe into a TensorDataset
    if includeLabels == False:
        tensordataset = TensorDataset(
            torch.tensor(df["tokenizedTopics"][:nbrExamples]), 
            torch.tensor(df["attentionMask"][:nbrExamples]), 
            torch.tensor(df["labels"][:nbrExamples])
        )
    else:
        tensordataset = TensorDataset(
            torch.tensor(df["tokenizedTopics"][:nbrExamples]), 
            torch.tensor(df["attentionMask"][:nbrExamples])
        )

    return DataLoader(
        tensordataset, 
        sampler = sampler(tensordataset),
        batch_size = batch_size
    )
    
def loadTokenizer(tokenizerModel) -> BertTokenizer:
    """
    Method loads the tokenizer
    :Returns:
        tokenizer (BertTokenizer): tokenizer for the BERT model
    """
    # We will be using the uncased version here i.e. the tokenization does not regarded upper or lower case 
    return BertTokenizer.from_pretrained(
                                            tokenizerModel, 
                                            do_lower_case=True, 
                                            truncation=True
                                        )

def tokenization(
        df:pd.DataFrame, 
        tokenizerModel:str= "bert-base-uncased", 
        nbrExamples:int=0
    ) -> None:
    
    # 512 is max length of BERT model
    MAX_LENGTH = 512
    
    # Initialize lists
    input_ids = []
    attention_masks = []
    
    if nbrExamples == 0: 
        nbrExamples = len(df["body"])
        
    progress_bar = tqdm(total=nbrExamples, desc="Tokenization")
    
    tokenizer = loadTokenizer(tokenizerModel)

    # Tokenization
    for i in range(0,nbrExamples):
        # `encode_plus` will:
        #   (1) Tokenize the text.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the text to `max_length`.
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            df["body"][i],                  # text to encode.
                            add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                            max_length = MAX_LENGTH,        # Pad & truncate all textInputs.
                            pad_to_max_length = True,
                            truncation = True, 
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',          # Return pytorch tensors.
                    )
        
        # Add the encoded text to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        progress_bar.update(1)
    
    progress_bar.close()
    
    # Transform the lists into tensors to afterprocesss them
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    # Need to fill the tensor with zeros to match the dataframe size
    if nbrExamples < len(df["body"]): 
        filler = torch.zeros((len(df["body"]),MAX_LENGTH),dtype=torch.int32)
        filler[:nbrExamples,:] = input_ids
        input_ids = filler.tolist()
        filler[:nbrExamples,:] = attention_masks
        attention_masks = filler.tolist()
    else:
        input_ids = input_ids.tolist()
        attention_masks = attention_masks.tolist()
    
    df["tokenizedTopics"] = input_ids
    df["attentionMask"] = attention_masks