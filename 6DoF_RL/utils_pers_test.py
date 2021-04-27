import pickle
import pandas
import numpy as np
import random
import time


class read_db:
    def __init__(self, is_train=False, test_folder=""):
        # train and testing setting:
        # equi
        # self.dbs = ["TechnicolorMuseum", "ClassroomVideo", "TechnicolorHijack"]
        # pers
        # self.dbs = ['PoznanFencing', 'IntelFrog', 'PoznanHall','PoznanCarpark', 'TechnicolorPainter', 'OrangeKitchen'] # , 'PoznanStreet'
        # test pers
        self.dbs = [
            "PoznanHall",
            "OrangeKitchen",
            "IntelFrog",
            "TechnicolorPainter",
            "PoznanCarpark",
            "PoznanStreet",
            "PoznanFencing",
        ]
        
        self.sv = self.read_source_views()
        self.cp = self.read_camera_parameters()
        if is_train:
            self.db = self.read_databases("./datasets/states/train/")
        else:
            self.db = self.read_databases("./datasets/states/test/" + test_folder + "/")

    def read_source_views(self):  # read source views
        sv = {}
        for db_fn in self.dbs:
            with open("./datasets/pickle/" + db_fn + ".pkl", "rb") as f:
                sv[db_fn] = pickle.load(f)
        return sv

    def read_camera_parameters(self):
        # read camera parameters
        cp = {}
        for db_fn in self.dbs:
            with open("./datasets/camera_parameters/" + db_fn + ".csv", "rb") as f:
                cp[db_fn] = pandas.read_csv(f)
        return cp

    def read_databases(self, path):  # read db
        db = {}
        for db_fn in self.dbs:
            with open(path + db_fn + ".csv", "rb") as f:
                db[db_fn] = pandas.read_csv(f)
        return db


class observe:
    def __init__(self, sv, cp, db):
        # train and testing setting:
        # equi
        # self.dbs = ["TechnicolorMuseum", "ClassroomVideo", "TechnicolorHijack"]
        # pers
        # self.dbs = ['PoznanFencing', 'IntelFrog', 'PoznanHall','PoznanCarpark', 'TechnicolorPainter', 'OrangeKitchen'] # , 'PoznanStreet'
        # test pers
        self.dbs = [
            "PoznanHall",
            "OrangeKitchen",
            "IntelFrog",
            "TechnicolorPainter",
            "PoznanCarpark",
            "PoznanStreet",
            "PoznanFencing",
        ]
        

        self.sv = sv
        self.cp = cp
        self.db = db
        # self.random_settings()
        # train and testing setting:
        # equi
        # self.full_data = pandas.concat(
        #     [self.db[self.dbs[0]], self.db[self.dbs[1]], self.db[self.dbs[2]]], axis=0
        # )
        # training
        # self.full_data = pandas.concat([self.db[self.dbs[0]], self.db[self.dbs[1]], self.db[self.dbs[2]],
        #                                 self.db[self.dbs[3]], self.db[self.dbs[4]], self.db[self.dbs[5]]], axis=0)
        # testing
        self.full_data = pandas.concat(
            [
                self.db[self.dbs[0]],
                self.db[self.dbs[1]],
                self.db[self.dbs[2]],
                self.db[self.dbs[3]],
                self.db[self.dbs[4]],
                self.db[self.dbs[5]],
                self.db[self.dbs[6]],
            ],
            axis=0,
        )
        #
        self.seq_idx = 0

    def random_settings(self):
        self.r_db = random.choice(self.dbs)
        # print(self.r_db)
        temp_db = self.db[self.r_db]
        self.r_frame = random.choice(temp_db.Frame.unique())
        # print(self.r_frame)
        self.r_tvs = random.choice(
            temp_db.loc[(temp_db.Frame == self.r_frame)]["Synthesized.View"].unique()
        )
        # print(self.r_tvs)
        self.r_data = temp_db.loc[
            (temp_db.Frame == self.r_frame)
            & (temp_db["Synthesized.View"] == self.r_tvs)
        ]
        # state transition
        self.state_trans = self.r_data[
            ["p1", "p2", "p3", "CEL", "WS.PSNR", "theo_time"]
        ].copy()
        self.gen_observation()

    def sequence_select(self):
        if self.seq_idx < self.full_data.shape[0]:
            self.r_db = self.full_data.iloc[self.seq_idx].Dataset
            temp_db = self.db[self.r_db]
            self.r_frame = self.full_data.iloc[self.seq_idx].Frame
            self.r_tvs = self.full_data.iloc[self.seq_idx]["Synthesized.View"]
            self.r_data = temp_db.loc[
                (temp_db.Frame == self.r_frame)
                & (temp_db["Synthesized.View"] == self.r_tvs)
            ]
            self.state_trans = self.r_data[
                ["p1", "p2", "p3", "CEL", "WS.PSNR", "theo_time"]
            ].copy()
            self.gen_observation()
            self.seq_idx += 1
        else:
            print("data exhaust")
            self.seq_idx = 0
            self.r_db = self.full_data.iloc[self.seq_idx].Dataset
            temp_db = self.db[self.r_db]
            self.r_frame = self.full_data.iloc[self.seq_idx].Frame
            self.r_tvs = self.full_data.iloc[self.seq_idx]["Synthesized.View"]
            self.r_data = temp_db.loc[
                (temp_db.Frame == self.r_frame)
                & (temp_db["Synthesized.View"] == self.r_tvs)
            ]
            self.state_trans = self.r_data[
                ["p1", "p2", "p3", "CEL", "WS.PSNR", "theo_time"]
            ].copy()
            self.gen_observation()
            self.seq_idx += 1

    def print_settings(self):
        print("[{0}] frame:{1}, V_tar:{2}".format(self.r_db, self.r_frame, self.r_tvs))

    def gen_observation(self):
        # sorted images and depths
        # get index
        # print(self.r_db)
        # print(self.sv[self.r_db]['fn_frames'])
        # print(str(self.r_frame))
        temp_sv = self.sv[self.r_db]
        frame_idx = np.where(np.array(temp_sv["fn_frames"]) == str(self.r_frame))[0][0]
        # print(frame_idx)
        # print(self.sv[self.r_db]['fn_sv'])
        # print(np.array(self.r_data.filter(regex='^V',axis=1).iloc[0]))
        sort_order = [
            np.where(temp_sv["fn_sv"] == str(i))[0][0]
            for i in np.array(self.r_data.filter(regex="^V", axis=1).iloc[0])
        ]
        # print(sort_order)
        self.I_src = temp_sv["imgs"][sort_order, frame_idx, ...]
        self.D_src = temp_sv["depth"][sort_order, frame_idx, ...]
        # get index# get index
        # get source camera parameters, positions and orientation
        sort_order = [
            np.where(np.asarray(temp_sv["c_para"].Name) == ("v" + str(i)))[0][0]
            for i in np.array(self.r_data.filter(regex="^V", axis=1).iloc[0])
        ]
        # print(sort_order)
        self.P_src = temp_sv["c_para"].loc[sort_order, "Position1":"Position3"].values
        # print(self.P_src)
        self.O_src = temp_sv["c_para"].loc[sort_order, "Rotation1":"Rotation3"].values
        # print(self.O_src)
        # get target parameter
        # print(np.asarray(temp_sv['c_para'].Name))
        # print(('v'+str(self.r_tvs)))
        tar_idx = np.where(
            np.asarray(temp_sv["c_para"].Name) == ("v" + str(self.r_tvs))
        )[0][0]
        self.P_tar = temp_sv["c_para"].loc[tar_idx, "Position1":"Position3"].values
        # print(tar_cp)
        self.O_tar = temp_sv["c_para"].loc[tar_idx, "Rotation1":"Rotation3"].values
        # print(tar_co)

    def print_training_data_infos(self):
        print(
            "[{0}] frame:{1}, V_tar:{2}, I_src:{3}, D_src:{4}\nP_src:\n{5}\nO_src:\n{6}\nP_tar:\n{7}\nO_tar:\n{8}".format(
                self.r_db,
                self.r_frame,
                self.r_tvs,
                self.I_src.shape,
                self.D_src.shape,
                self.P_src,
                self.O_src,
                self.P_tar,
                self.O_tar,
            )
        )
