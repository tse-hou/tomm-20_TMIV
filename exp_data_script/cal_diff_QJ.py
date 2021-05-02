#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import cal_prediction_performance
from utils import cal_DEF_performance
from utils import cal_OPT_performance


def PTP_diff_QJ(requested_quality_in_CEL, istrain):
    TMIV_csv_folder_path = "..\\db_preprocessing\\datasets_pers\\states\\train"
    PTP_dataset_names = [
        "IntelFrog",
        "OrangeKitchen",
        "PoznanCarpark",
        "PoznanFencing",
        "PoznanHall",
        "PoznanStreet",
        "TechnicolorPainter",
    ]

    DEF_CEL = 0.0
    OPT_CEL = 0.0
    DRL_CEL = 0.0
    CNN_CEL = 0.0
    for idx, dataset in enumerate(PTP_dataset_names):
        # DEF
        print(f"loading DEF")
        DEF_performance = cal_DEF_performance(
            "DEF_OPT_template_PTP.csv",
            "..\\db_preprocessing\\datasets_pers\\states\\train",
            requested_quality_in_CEL,
        )
        if istrain == 1:
            DEF_performance = DEF_performance[DEF_performance["Dataset"] != dataset]
        else:
            DEF_performance = DEF_performance[DEF_performance["Dataset"] == dataset]
        DEF_CEL += DEF_performance["CEL"].mean()
        #
        # OPT
        print(f"loading OPT")
        OPT_performance = cal_OPT_performance(
            "DEF_OPT_template_PTP.csv",
            "..\\db_preprocessing\\datasets_pers\\states\\train",
            requested_quality_in_CEL,
        )
        if istrain == 1:
            OPT_performance = OPT_performance[OPT_performance["Dataset"] != dataset]
        else:
            OPT_performance = OPT_performance[OPT_performance["Dataset"] == dataset]
        OPT_CEL += OPT_performance["CEL"].mean()
        #
        # DRL
        print(f"loading {dataset}, DRL")
        prediction_csv = (
            f"..\\NN-based_algorithm_seleted_predict_result\\PTP\\{dataset}-6_obj_nsv_best.csv"
        )
        DRL_performance = cal_prediction_performance(
            prediction_csv, TMIV_csv_folder_path, requested_quality_in_CEL
        )
        if istrain == 1:
            DRL_performance = DRL_performance[DRL_performance["Dataset"] != dataset]
        else:
            DRL_performance = DRL_performance[DRL_performance["Dataset"] == dataset]
        DRL_CEL += DRL_performance["CEL"].mean()
        #
        # CNN
        print(f"loading {dataset}, CNN")
        prediction_csv = (
            f"..\\NN-based_algorithm_seleted_predict_result\\PTP\\{dataset}-6-cnn_obj_nsv.csv"
        )
        CNN_performance = cal_prediction_performance(
            prediction_csv, TMIV_csv_folder_path, requested_quality_in_CEL
        )
        if istrain == 1:
            CNN_performance = CNN_performance[CNN_performance["Dataset"] != dataset]
        else:
            CNN_performance = CNN_performance[CNN_performance["Dataset"] == dataset]
        CNN_CEL += CNN_performance["CEL"].mean()
        #
    DEF_CEL /= 7
    OPT_CEL /= 7
    DRL_CEL /= 7
    CNN_CEL /= 7
    sns.barplot(
        x=["DEF", "CNN", "DRL"], y=[DEF_CEL / OPT_CEL, CNN_CEL / OPT_CEL, DRL_CEL / OPT_CEL]
    )
    print([DEF_CEL / OPT_CEL, CNN_CEL / OPT_CEL, DRL_CEL / OPT_CEL])


#%%
PTP_diff_QJ(18, 0)
#%%
PTP_diff_QJ(20, 0)
#%%
PTP_diff_QJ(22, 0)
#%%
PTP_diff_QJ(24, 0)
#%%
PTP_diff_QJ(26, 0)

# %%
