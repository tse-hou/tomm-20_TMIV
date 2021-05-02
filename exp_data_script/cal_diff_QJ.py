#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import cal_prediction_performance
from utils import cal_DEF_performance
from utils import cal_OPT_performance


def PTP_diff_QJ(requested_quality_in_CEL, istrain):
    TMIV_csv_folder_path = "../db_preprocessing/datasets_pers/states/train"
    PTP_dataset_names = [
        "IntelFrog",
        "OrangeKitchen",
        "PoznanCarpark",
        "PoznanFencing",
        "PoznanHall",
        "PoznanStreet",
        "TechnicolorPainter",
    ]

    DEF_performances = []
    OPT_performances = []
    DRL_performances = []
    CNN_performances = []
    DEF_min = 0.0
    OPT_min = 0.0
    DRL_min = 0.0
    CNN_min = 0.0
    for idx, dataset in enumerate(PTP_dataset_names):
        # DEF
        print(f"loading DEF")
        DEF_performance = cal_DEF_performance(
            "DEF_OPT_template_PTP.csv",
            "../db_preprocessing/datasets_pers/states/train",
            requested_quality_in_CEL,
        )
        if istrain == 1:
            DEF_performance = DEF_performance[DEF_performance["Dataset"] != dataset]
        else:
            DEF_performance = DEF_performance[DEF_performance["Dataset"] == dataset]
        DEF_performances += DEF_performance
        DEF_min = min(DEF_min, DEF_performance["CEL"].min())
        #
        # OPT
        print(f"loading OPT")
        OPT_performance = cal_OPT_performance(
            "DEF_OPT_template_PTP.csv",
            "../db_preprocessing/datasets_pers/states/train",
            requested_quality_in_CEL,
        )
        if istrain == 1:
            OPT_performance = OPT_performance[OPT_performance["Dataset"] != dataset]
        else:
            OPT_performance = OPT_performance[OPT_performance["Dataset"] == dataset]
        OPT_performances += OPT_performance
        OPT_min = min(OPT_min, OPT_performance["CEL"].min())
        #
        # DRL
        print(f"loading {dataset}, DRL")
        prediction_csv = (
            f"../NN-based_algorithm_seleted_predict_result/PTP/{dataset}-6_obj_nsv_best.csv"
        )
        DRL_performance = cal_prediction_performance(
            prediction_csv, TMIV_csv_folder_path, requested_quality_in_CEL
        )
        if istrain == 1:
            DRL_performance = DRL_performance[DRL_performance["Dataset"] != dataset]
        else:
            DRL_performance = DRL_performance[DRL_performance["Dataset"] == dataset]
        DRL_performances += DRL_performance
        DRL_min = min(DRL_min, DRL_performance["CEL"].min())

        #
        # CNN
        print(f"loading {dataset}, CNN")
        prediction_csv = (
            f"../NN-based_algorithm_seleted_predict_result/PTP/{dataset}-6-cnn_obj_nsv.csv"
        )
        CNN_performance = cal_prediction_performance(
            prediction_csv, TMIV_csv_folder_path, requested_quality_in_CEL
        )
        if istrain == 1:
            CNN_performance = CNN_performance[CNN_performance["Dataset"] != dataset]
        else:
            CNN_performance = CNN_performance[CNN_performance["Dataset"] == dataset]
        CNN_performances += CNN_performance
        CNN_min = min(CNN_min, CNN_performance["CEL"].min())

        #
    
    gmin = min(DEF_min, OPT_min, CNN_min, DRL_min)
    DEF_performances['optimal score'] = (DEF_performances['CEL'] - gmin) / (OPT_performances['CEL'] - gmin)
    DEF_OS = DEF_performances['optimal score'].mean()
    CNN_performances['optimal score'] = (CNN_performances['CEL'] - gmin) / (OPT_performances['CEL'] - gmin)
    CNN_OS = CNN_performances['optimal score'].mean()
    DRL_performances['optimal score'] = (DRL_performances['CEL'] - gmin) / (OPT_performances['CEL'] - gmin)
    DRL_OS = DRL_performances['optimal score'].mean()
    pd_OS = pd.DataFrame(data={"name":["DEF", "CNN", "DRL"],"OS":[DEF_OS, CNN_OS, DRL_OS]})
    pd_OS.plot.bar(x="name",y="OS")
    # sns.barplot(
    #     x=["DEF", "CNN", "DRL"], y=[DEF_OS, CNN_OS, DRL_OS]
    # )
    print([DEF_OS, CNN_OS, DRL_OS])


#%%
PTP_diff_QJ(18, 0)
#%%
# PTP_diff_QJ(20, 0)
# #%%
# PTP_diff_QJ(22, 0)
# #%%
# PTP_diff_QJ(24, 0)
# #%%
# PTP_diff_QJ(26, 0)

# %%
