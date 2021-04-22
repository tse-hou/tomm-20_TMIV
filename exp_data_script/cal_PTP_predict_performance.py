#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import cal_prediction_performance
from utils import cal_DEF_performance
from utils import cal_OPT_performance

sns.set_theme(style="whitegrid")
PTP_dataset_names = [
    "IntelFrog",
    "OrangeKitchen",
    "PoznanCarpark",
    "PoznanFencing",
    "PoznanHall",
    "PoznanStreet",
    "TechnicolorPainter",
]
# DEF
print(f"loading DEF")
DEF_performance = cal_DEF_performance(
    "DEF_OPT_template_PTP.csv", "..\\db_preprocessing\\datasets_pers\\states\\train", 20
)
# OPT
print(f"loading OPT")
OPT_performance = cal_OPT_performance(
    "DEF_OPT_template_PTP.csv", "..\\db_preprocessing\\datasets_pers\\states\\train", 20
)
# PTP_dataset_names = ["IntelFrog"]
requested_quality_in_CEL = 20
TMIV_csv_folder_path = "..\\db_preprocessing\\datasets_pers\\states\\train"
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
for idx, dataset in enumerate(PTP_dataset_names):
    # DRL
    print(f"loading {dataset}, DRL")
    prediction_csv = (
        f"..\\NN-based_algorithm_seleted_predict_result\\PTP\\{dataset}-6_obj_nsv_best.csv"
    )
    DRL_performance = cal_prediction_performance(
        prediction_csv, TMIV_csv_folder_path, requested_quality_in_CEL
    )
    # CNN
    print(f"loading {dataset}, CNN")
    prediction_csv = (
        f"..\\NN-based_algorithm_seleted_predict_result\\PTP\\{dataset}-6-cnn_obj_nsv.csv"
    )
    CNN_performance = cal_prediction_performance(
        prediction_csv, TMIV_csv_folder_path, requested_quality_in_CEL
    )
    print("CNN_diff_OPT: ", CNN_performance["CEL"].mean() - OPT_performance["CEL"].mean())
    print("DRL_diff_OPT: ", DRL_performance["CEL"].mean() - OPT_performance["CEL"].mean())
    # plot figure
    plot_data = pd.DataFrame(
        np.array(
            [
                ["DEF", DEF_performance["CEL"].mean()],
                ["CNN", CNN_performance["CEL"].mean()],
                ["DRL", DRL_performance["CEL"].mean()],
                ["OPT", OPT_performance["CEL"].mean()],
            ]
        ),
        columns=["algo", "CEL"],
    )
    plot_data["CEL"] = pd.to_numeric(plot_data["CEL"])
    local_axes = axes.flatten()
    sns.barplot(ax=local_axes[idx], x="algo", y="CEL", data=plot_data)
    local_axes[idx].set_title(dataset)
# %%
