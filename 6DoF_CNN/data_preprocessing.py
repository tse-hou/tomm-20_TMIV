# This script is used to produce traning and testing dataset/label 
# training (testing) dataset compose of:
# source view (texture)     : 7
# source view (depth)       : 7
# orientation diff. (Yaw)   : 7
# orientation diff. (Pitch) : 7
# orientation diff. (Row)   : 7
# position diff. (x)        : 7
# position diff. (y)        : 7
# position diff. (z)        : 7
# -------------------------------
# total:                     56
import pickle
import numpy as np
import pandas
import os
import csv
import sys


class dataset:
    def __init__(self, pkl_path='', csvFilePath=''):
        if(pkl_path==''):
            print("error: pkl_path is empty")
            return -1
        f=open(pkl_path,'rb')
        data=pickle.load(f)
        self.csvFilePath = csvFilePath
        self.pkl_depth = data['depth'] # numpy array
        self.pkl_imgs = data['imgs'] # numpy array
        self.pkl_camera_para = data['c_para'] # pandas dataframe
        self.pkl_idx_frames = data['fn_frames'] # list
        self.pkl_idx_sourceView = data['fn_sv'] # numpy array
        self.traning_label = []
        self.get_opt_label()
        self.gray_scale_imgs()
        self.depth2Distance_depthmaps()
        
    # ---------------------------------------------------------------
    # get label data from .csv file
    def get_opt_label(self):
        idx = 0
        opt_numofPass = [0,0,0,0,0]
        max_CEL = -999999.0
        with open(self.csvFilePath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if(float(row['CEL'])>max_CEL):
                    opt_numofPass[0] = row['Frame']
                    opt_numofPass[1] = row['Synthesized.View']
                    opt_numofPass[2] = row['p1']
                    opt_numofPass[3] = row['p2']
                    opt_numofPass[4] = row['p3']
                    max_CEL = max(float(row['CEL']),max_CEL)
                idx += 1
                if(idx == 63):
                    self.traning_label.append(opt_numofPass)
                    idx = 0
                    opt_numofPass = [0,0,0,0,0]
                    max_CEL = -999999

    # ---------------------------------------------------------------
    
    # reshape depth map
    def depth2Distance_depthmaps(self):
        self.pkl_depth = np.delete(self.pkl_depth,[1,2],axis=-1).reshape((7, 5, 256, 256))  
        # new_map = np.zeros((7, 5, 256, 256))
        # for view in range(7):
        #     for frame in range(5):
        #         new_map[view][frame] = np.delete(self.pkl_depth[view][frame],[1,2],axis=2).reshape(256,256)
        # self.pkl_depth = new_map
    # ---------------------------------------------------------------
    # convert imgs to gray scale
    def gray_scale(self,img, width, height):
        img = np.sum(img,axis=2)/3
        return img.reshape(256,256)
        
    # gray scale all of imgs in dataset
    def gray_scale_imgs(self):
        new_imgs = np.zeros((7, 5, 256, 256))
        for view in range(7):
            for frame in range(5):
                new_img = self.gray_scale(self.pkl_imgs[view][frame],256,256)
                new_imgs[view][frame] = new_img
        self.pkl_imgs = new_imgs
    # ---------------------------------------------------------------
    # map generator
    def para_map_generator(self,source_para, target_para, width, height):
        para_diff = target_para - source_para
        return np.full((width,height), para_diff)
    # ---------------------------------------------------------------
    # output training data
    def output_training_data(self):
        datas = []
        for row in self.traning_label:
            single_data = np.zeros((56, 256, 256))
            idx = 0
            frameIdx = -1
            for i in range(len(self.pkl_idx_frames)):
                if(str(row[0])==self.pkl_idx_frames[i]):
                    frameIdx = i
            targetViewIdx = int(row[1].replace("v",""))
            # imgs
            for view in range(7):
                single_data[idx] = self.pkl_imgs[view][frameIdx]
                idx+=1
            # depth
            for view in range(7):
                single_data[idx] = self.pkl_depth[view][frameIdx]
                idx+=1
            # position
            for sourceViewIdx in self.pkl_idx_sourceView:
                for para_idx in range(1,4):
                    source_para = self.pkl_camera_para[f'Position{para_idx}'][int(sourceViewIdx)]
                    target_para = self.pkl_camera_para[f'Position{para_idx}'][targetViewIdx]
                    single_data[idx] = self.para_map_generator(source_para, target_para, 256, 256)
                    idx+=1
            # orientation
            for sourceViewIdx in self.pkl_idx_sourceView:
                for para_idx in range(1,4):
                    source_para = self.pkl_camera_para[f'Rotation{para_idx}'][int(sourceViewIdx)]
                    target_para = self.pkl_camera_para[f'Rotation{para_idx}'][targetViewIdx]
                    single_data[idx] = self.para_map_generator(source_para, target_para, 256, 256)
                    idx+=1
            datas.append(single_data)
        T_data = np.zeros((len(datas), 56, 256, 256))
        for i in range(0,len(datas)):
            T_data[i] = datas[i]
        return T_data

def produce_data(dataset_name_list,T_data_path,T_label_path, pklFolderPath, csvFolderPath):
    allofdataset = []
    alloflabel = []
    for i in range(len(dataset_name_list)):
        print(f"producing dataset'{dataset_name_list[i]}'")
        dataset_temp = dataset(pkl_path = f"{pklFolderPath}/{dataset_name_list[i]}.pkl",
                                csvFilePath = f"{csvFolderPath}/{dataset_name_list[i]}.csv")
        dataset_temp_T_data = dataset_temp.output_training_data()
        allofdataset.append(dataset_temp_T_data)
        for row in dataset_temp.traning_label:
            temp = np.zeros((3))
            temp[0] = (int(row[2])/7)
            temp[1] = ((int(row[3])-int(row[2]))/7)
            temp[2] = ((int(row[4])-int(row[3]))/7)
            alloflabel.append(temp)
    # combine all of data
    for i in range(1,len(allofdataset)):
        allofdataset[0] = np.concatenate((allofdataset[0],allofdataset[i]),axis=0)
    T_data = allofdataset[0]

    # combine all of lable
    T_label = np.zeros((len(alloflabel), 3))
    for i in range(len(alloflabel)):
        T_label[i] = alloflabel[i]

    T_data = np.transpose(T_data,(0,2,3,1))

    print(type(T_data))
    print(T_data.shape)
    print(type(T_label))
    print(T_label.shape)
    
    np.save(T_data_path, T_data)
    np.save(T_label_path,T_label)
    print("done")

if __name__ == "__main__":
    dataset_type = sys.argv[1]
    if(dataset_type == "PTP"):
        pklFolderPath = "code_silver/prepare_datasets_PTP/pickle"
        csvFolderPath = "code_silver/raw_datasets/states/PTP"
        dataset_list = ["IntelFrog","OrangeKitchen","PoznanFencing","PoznanStreet","TechnicolorPainter","PoznanCarpark","PoznanHall"]
        for i in range(len(dataset_list)):
            # training
            testing_data = dataset_list[i]
            dataset_name_list = dataset_list.copy()
            dataset_name_list.remove(testing_data)
            T_data_path = f'CNN_dataset/Training/Training_data_{testing_data}.npy'
            T_label_path = f'CNN_dataset/Training/Training_label_{testing_data}.npy'
            produce_data(dataset_name_list,T_data_path,T_label_path, pklFolderPath, csvFolderPath)
        # testing
        produce_data(dataset_list,"CNN_dataset/Testing/Testing_data.npy","CNN_dataset/Testing/Testing_label.npy", pklFolderPath, csvFolderPath)
    
    elif(dataset_type == "ERP"):
        pklFolderPath = "code_silver/prepare_datasets_ERP/pickle"
        dataset_list = ["ClassroomVideo","TechnicolorHijack","TechnicolorMuseum"]
        # training
        # csvFolderPath = "code_silver/raw_datasets/states/ERP/train"
        # T_data_path = 'CNN_dataset/Training/Training_data_NSV.npy'
        # T_label_path = 'CNN_dataset/Training/Training_label_NSV.npy'
        # produce_data(dataset_list,T_data_path,T_label_path, pklFolderPath, csvFolderPath)
        
        # testing
        csvFolderPath = "code_silver/raw_datasets/states/ERP/test"
        T_data_path = 'CNN_dataset/Testing/Testing_data_RND.npy'
        T_label_path = 'CNN_dataset/Testing/Testing_label_RND.npy'
        produce_data(dataset_list,T_data_path,T_label_path, pklFolderPath, csvFolderPath)
