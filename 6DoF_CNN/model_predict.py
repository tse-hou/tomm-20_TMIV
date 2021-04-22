# used trained model to predict
import tensorflow as tf
import keras
import numpy as np
import csv
import sys
import os
class Evaluation:
    def __init__(self,model_path='',CNN_dataset_path='',including_dataset='', testing_video='', isTesting=False, videoType='PTP'):
        self.testing_data = np.load(CNN_dataset_path)
        self.model = keras.models.load_model(model_path)
        self.including_dataset = including_dataset
        filename = os.path.basename(CNN_dataset_path)
        self.CNN_dataset_filename = os.path.splitext(filename)[0]
        self.testing_video = testing_video
        self.isTesting = isTesting
        self.videoType = videoType

    # using model and data to predict numviewperpass
    def model_predict(self):
        self.model_output = self.model.predict(self.testing_data)

    # combine model output and other data(datasetName, frame, target view) and output a .csv file
    def output_csv(self):
        # reconvert the output data to number of view per pass
        for i in range(len(self.model_output)):
            for j in range(3):
                self.model_output[i][j]=round(self.model_output[i][j]*7,0)
            if(self.model_output[i][1]!=0):
                self.model_output[i][1] = self.model_output[i][1] + self.model_output[i][0] 
                if(self.model_output[i][2]!=0):
                    self.model_output[i][2] = self.model_output[i][1] + self.model_output[i][2] 
        # combine data into a csv file
        output_csv_data = [["Dataset", "Frame", "Synthesized.View", "p1", "p2", "p3"]]
        model_output_idx = 0
        if(self.videoType == "PTP"):
            csvfileFolder = "code_silver/raw_datasets/states/PTP"
        elif(self.videoType == "ERP" and self.isTesting == True):
            csvfileFolder = f"code_silver/raw_datasets/states/ERP/test"
        elif(self.videoType == "ERP" and self.isTesting == False):
            csvfileFolder = f"code_silver/raw_datasets/states/ERP/train"
        for datasetName in self.including_dataset:
            with open(f"{csvfileFolder}/{datasetName}.csv", newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                idx = 0
                single_row_output = [0,0,0,0,0,0]
                for row in reader:
                    if(idx == 0):
                        single_row_output[0] = datasetName
                        if(self.videoType == "ERP"):
                            single_row_output[1] = row['Frame']
                            single_row_output[2] = row['Synthesized.View']
                        else:
                            single_row_output[1] = row['Frame'][1:]
                            single_row_output[2] = row['Synthesized.View'][2:]
                       
                        # if(single_row_output[2][0]=='0'):
                        #     single_row_output[2]=single_row_output[2][1:]
                        single_row_output[3] = int(self.model_output[model_output_idx][0])
                        single_row_output[4] = int(self.model_output[model_output_idx][1])
                        single_row_output[5] = int(self.model_output[model_output_idx][2])
                        model_output_idx+=1
                    idx += 1
                    if(idx == 63):
                        output_csv_data.append(single_row_output)
                        idx = 0
                        single_row_output = [0,0,0,0,0,0]
        # output data as a csv file
        if(self.videoType == "PTP"):
            if(self.isTesting==True):
                output_filename = f"CNN_csv/20201202_3/{self.CNN_dataset_filename}_{testing_video}.csv"
            else:
                output_filename = f"CNN_csv/20201202_3/{self.CNN_dataset_filename}.csv"
        else:
            if(self.isTesting==True):
                output_filename = "CNN_csv/20201202_3/Testing_data_RND.csv"
            else:
                output_filename = "CNN_csv/20201202_3/Training_data_NSV.csv"
        with open(output_filename,"w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(output_csv_data)

if __name__ == "__main__":
    '''Set GPU'''
    gpus = [3] # Here I set CUDA to only see one GPU
    os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(i) for i in gpus])

    dataset_list = ["IntelFrog","OrangeKitchen","PoznanFencing","PoznanStreet","TechnicolorPainter","PoznanCarpark","PoznanHall"]
    videoType = sys.argv[1]
    isTrainingOrTesting = sys.argv[2]
    if (videoType == "PTP"):
        for i in range(len(dataset_list)):
            if(isTrainingOrTesting == "training"):
                # training
                testing_video = dataset_list[i]
                including_dataset = dataset_list.copy()
                including_dataset.remove(testing_video)
                model_path = f'CNN_model/model/CNN_{testing_video}_20201202_3.h5'
                CNN_dataset_path = f"CNN_dataset/Training/Training_data_{testing_video}.npy"
                evaluation_training = Evaluation(model_path = model_path,
                                                CNN_dataset_path = CNN_dataset_path, 
                                                including_dataset = including_dataset)
                evaluation_training.model_predict()
                evaluation_training.output_csv()
                print(f"Training_data_{testing_video}.npy done")
            elif(isTrainingOrTesting == "testing"):
                # testing
                testing_video = dataset_list[i]
                model_path = f'CNN_model/model/CNN_{testing_video}_20201202_3.h5'
                including_dataset = dataset_list.copy()
                CNN_dataset_path = "CNN_dataset/Testing/Testing_data.npy"
                evaluation_testing = Evaluation(model_path = model_path, 
                                                CNN_dataset_path = CNN_dataset_path, 
                                                including_dataset = including_dataset, 
                                                testing_video = testing_video, 
                                                isTesting = True)
                evaluation_testing.model_predict()
                evaluation_testing.output_csv()
                print(f"Testing_data.npy done")
    elif (videoType == "ERP"):
        including_dataset = ["ClassroomVideo", "TechnicolorHijack", "TechnicolorMuseum"]
        model_path = f'CNN_model/model/CNN_ERP_20201109_3.h5'
        if (isTrainingOrTesting == "training"):
            CNN_dataset_path = "CNN_dataset/Training/Training_data_NSV.npy"
            evaluation_training = Evaluation(model_path = model_path,
                                            CNN_dataset_path = CNN_dataset_path, 
                                            including_dataset = including_dataset,
                                            videoType = "ERP")
            evaluation_training.model_predict()
            evaluation_training.output_csv()
            print(f"Training_data.npy done")
        elif (isTrainingOrTesting == "testing"):
            CNN_dataset_path = "CNN_dataset/Testing/Testing_data_RND.npy"
            evaluation_testing = Evaluation(model_path = model_path,
                                            CNN_dataset_path = CNN_dataset_path, 
                                            including_dataset = including_dataset,
                                            videoType = "ERP",
                                            isTesting = True)
            evaluation_testing.model_predict()
            evaluation_testing.output_csv()
            print(f"Testing_data.npy done")


