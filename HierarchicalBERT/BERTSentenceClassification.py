import torch
import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Bert library
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Own packages
from utils.utils import flat_accuracy, tokenization, convertDataframetoDataloader

def loadBSCmodel(name:str, 
        dataloaders:torch.utils.data.DataLoader=None, 
        listLabelDict:list=None,
        epochs:int=2, 
        hierarchyLevel:int=0, 
        num_labels:int=[11],
        lr:float=2e-5, 
        eps:float=1e-8
    ):
    """
    Function loads the BSC model
    """
    model = BSC(
                finetunedModel = True,
                finetunedModelName = name,
                dataloaders = dataloaders,
                epochs = epochs,
                num_labels = num_labels,
                hierarchyLevel = hierarchyLevel,
                listLabelDict = listLabelDict,
                lr = lr,
                eps = eps
            )
    return model

class BSC:
    def __init__(
                    self, 
                    finetunedModel:bool=False,
                    finetunedModelName:str='',
                    dataloaders:torch.utils.data.DataLoader=None, 
                    listLabelDict:list=None, 
                    hierarchyLevel:int=0,
                    num_labels:list=[11],
                    epochs:int=4, 
                    lr:float=2e-5, 
                    eps:float=1e-8
                ):
        
        self.listLabelDict = listLabelDict
        self.epochs = epochs
        self.device = self.selectDevice()
        self.dataloaders = dataloaders
        # Store the training stats between epochs to plot them later
        self.trainingStats = []
        self.hierarchyLevel = hierarchyLevel
        self.testPredictioins = []
            
        # The execpt will mostly be triggered when loading a model as one does not necessarily 
        # need to provide a dataloader when loading a model from a checkpoint
        try:
            self.num_labels = num_labels[hierarchyLevel]
        except IndexError:
            print("ERROR: num_labels is not determined for the given hierarchy level, setting it to 0. This is normal if you are reloading a model.")
            self.num_labels = 0    
           
        # If pretrained porvided load it, no fine-tuning necessary
        if finetunedModel == True:
            self.model = BertForSequenceClassification.from_pretrained(
                os.path.join("HierarchicalBERT",
                    "checkpoints", 
                    finetunedModelName
                )
            ) 
            self.num_labels = self.model.num_labels
        else: 
            # Pretrained BERT model with a single linear classification layer on top. 
            self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                num_labels = self.num_labels , # The number of output labels  
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False, # Whether the model returns all hidden-states.
            )  

        # Tell pytorch to run this model on the GPU if availabe
        if self.device == torch.device("cuda"):
            self.model.cuda()
        
        # Set optimizer 
        self.optimizer = AdamW(self.model.parameters(),
                    lr = lr, # args.learning_rate - default is 5e-5
                    eps = eps # args.adam_epsilon  - default is 1e-8.
                    )

        # Total number of training steps is [number of batches] x [number of epochs]. 
        if dataloaders != None:
            self.totalTrainSteps = len(self.dataloaders["train"]) * epochs
        else:
            self.totalTrainSteps = 0

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = self.totalTrainSteps)
    
    def test(self):
        # Put model in evaluation mode
        self.model.eval()
        
        progress_bar = tqdm(total=len(self.dataloaders["train"])-1, desc='Testing on batch', position=0)
        
        # Predict 
        for step, batch in enumerate(self.dataloaders["test"]):
            # `batch` contains three pytorch tensors: [0]: input ids, [1]: attention masks, [2]: labels
            b_labels = batch[2][:,self.hierarchyLevel]#.to(self.device)
            
            # Need to exclude the datapoints that are not labeled i.e. have a label of 0
            b_input_ids = batch[0][b_labels!=-1,:].to(self.device)
            b_input_mask = batch[1][b_labels!=-1,:].to(self.device)
            # Need to perform a type converion to long
            b_labels = torch.tensor(b_labels[b_labels!=-1],dtype=torch.long).to(self.device)

            # Forward pass, calculate logit predictions
            with torch.no_grad():
                result = self.model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                return_dict=True)

            self.testPredictioins.append(result.logits.detach().cpu().numpy())
            progress_bar.update(1)
            
            if (step % (len(self.dataloaders["test"])/10)) == 0:
                print(f"Step: {step}/{len(self.dataloaders['test'])}")
                print(f"True labels: {b_labels.detach().cpu().numpy()} and the decoded true labels: {self.translatePredictions(b_labels.detach().cpu().numpy())}")
                print(f"Predictions: {np.argmax(result.logits.detach().cpu().numpy(), axis=1).flatten()} and the decoded predicted labels {self.translatePredictions(np.argmax(result.logits.detach().cpu().numpy(), axis=1).flatten())}")
            
        # Finalize predictions
        progress_bar.close()  
        print('Testing done.')
        self.testPredictioins = np.concatenate(self.testPredictioins)
        
        # Load true labels
        trueLabels = self.dataloaders["test"].dataset[:][2][:,self.hierarchyLevel]
        
        # The following function automatically transforms the logits in to predictions
        return flat_accuracy(self.testPredictioins, trueLabels[trueLabels!=0])
    
    def train(self, seed_val:int=42):
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        for epoch_i in range(0, self.epochs):
            print("")
            print(f'======== Epoch {epoch_i + 1} / {self.epochs} ========')
            print('Training...')

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. `dropout` and `batchnorm` layers behave differently during training
            self.model.train()

            progress_bar = tqdm(total=len(self.dataloaders["train"])-1, desc='Training on batch', position=0)

            for batch in self.dataloaders["train"]:
                # `batch` contains three pytorch tensors: [0]: input ids, [1]: attention masks, [2]: labels
                b_labels = batch[2][:,self.hierarchyLevel]#.to(self.device)
                
                # Need to exclude the datapoints that are not labeled i.e. 
                # have a label of 0
                b_input_ids = batch[0][b_labels!=-1,:].to(self.device)
                b_input_mask = batch[1][b_labels!=-1,:].to(self.device)
                # Need to perform a type converion to long
                b_labels = torch.tensor(b_labels[b_labels!=-1],dtype=torch.long).to(self.device)
                
                # Need to one-hot encode the labels
                #b_labels = torch.nn.functional.one_hot(b_labels, num_classes=self.num_labels)#.unsqueeze(1)

                # Clear any previously calculated gradients before performing a backward pass
                self.model.zero_grad()        

                # Perform a forward pass and calculate the logits
                result = self.model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=torch.tensor(b_labels),
                            return_dict=True)
                loss = result.loss
                logits = result.logits

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0, prevents the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                self.optimizer.step()

                # Update the learning rate & progress bar
                self.scheduler.step()
                progress_bar.update(1)

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.dataloaders["train"])          

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            progress_bar.close()  
                
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            # Put the model in evaluation mode -- dropout layers behave differently	
            self.model.eval()

            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0

            # Evaluate data for one epoch
            for batch in self.dataloaders["val"]:
                
                # `batch` contains three pytorch tensors: [0]: input ids, [1]: attention masks, [2]: labels
                b_labels = batch[2][:,self.hierarchyLevel]#.to(self.device)
                
                # Need to exclude the datapoints that are not labeled i.e. 
                # have a label of 0
                b_input_ids = batch[0][b_labels!=-1,:].to(self.device)
                b_input_mask = batch[1][b_labels!=-1,:].to(self.device)
                # Need to perform a type converion to long
                b_labels = torch.tensor(b_labels[b_labels!=-1],dtype=torch.long).to(self.device)
                
                # Need to one-hot encode the labels
                # b_labels = torch.nn.functional.one_hot(b_labels, num_classes=self.num_labels)

                # No need to accumulate the gradients
                with torch.no_grad():        

                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which 
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    result = self.model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels,
                                return_dict=True)

                loss = result.loss
                logits = result.logits
                    
                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Accumulate the flat accuracy 
                total_eval_accuracy += flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(self.dataloaders["val"])
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.dataloaders["val"])
            
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))

            # Record all statistics from this epoch.
            self.trainingStats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                }
            )

        print("")
        print("Training complete!")
        
    def saveTrainingResults(self, hierarchyLevel:int=0, PATH:str=os.path.join("HierarchicalBERT","trainingResults")):
        counter = 0
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        pathAndName = os.path.join(PATH, f"trainingResultsHierarchy{hierarchyLevel}Run{counter}.json")
        while os.path.exists(pathAndName):
            counter += 1
            pathAndName = os.path.join(PATH, f"trainingResultsHierarchy{hierarchyLevel}Run{counter}.json")
        with open(pathAndName, "w") as f:
            json.dump(self.trainingStats, f)

    def getTrainingStats(self):
        df_stats = pd.DataFrame(data=self.trainingStats)
        df_stats = df_stats.set_index('epoch')
        return df_stats
     
    def plotTrainingLoss(self):
        df_stats = self.getTrainingStats()
        
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3, 4])

        plt.show()

    def saveModel(self, name:str):
        """
        Function to save the model
        :Args:
            :name: name of the version to save the model as
        TODO Need to update this in the future to save other parameters as well
        """
        PATH = os.path.join("HierarchicalBERT","checkpoints", name)
        if os.path.exists(PATH):
            print("ERROR: Checkpoint already exists. Change the name of the version.")
        else: 
            self.model.save_pretrained(PATH)  

    def inference(
            self, 
            df:pd.DataFrame, 
            tokenizerModel:str= "bert-base-uncased", 
            batch_size:int=16
        ) -> np.array:
        """
        Function to perform inference on a dataframe
        :Args:
            :df: dataframe with the text to perform inference on
            :tokenizerModel: name of the tokenizer model to use
        :RETRUN: array with the predictions
        """
        # tokenization(df, tokenizerModel = tokenizerModel)
        dataloader = convertDataframetoDataloader(df, includeLabels=True, batch_size=batch_size)
        
        predictions = []
        for batch in dataloader:
            # `batch` contains two pytorch tensors: [0]: input ids, [1]: attention masks
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)

            # Forward pass, calculate logit predictions
            with torch.no_grad():
                result = self.model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                return_dict=True)
            logits = result.logits.detach().cpu().numpy()
            predictions.append(np.argmax(logits, axis=1).flatten())
        return np.concatenate(predictions)
    
    def translatePredictions(self, predictions:np.array) -> np.array:
        dict = self.listLabelDict[self.hierarchyLevel]
        return np.array([list(dict.keys())[list(dict.values()).index(prediction)] 
                         for prediction in predictions])

    def selectDevice(self) -> torch.device:
        """
        Returns torch device to run on. If GPU is availabe will choose GPU otherwise CPU.
        :RETRUN: device to run on 
        """
        if torch.cuda.is_available():    
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
            return torch.device("cuda")

        else:
            print('No GPU available, using the CPU instead.')
            return torch.device("cpu")