#%%
from utils import cal_prediction_performance
from utils import cal_DEF_performance
from utils import cal_OPT_performance
import seaborn as sns

testing_dataset = "PoznanStreet"
DRL_prediction_csv = "B:/NMSL/my_paper_data/tomm20_TMIV/NN-based_algorithm_seleted_predict_result/response/PTP/PTP_QJ18_obj_nsv_best.csv"
TMIV_csv_folder_path = "..\\db_preprocessing\\datasets_pers\\states\\train"
requested_quality_in_CEL = 18
istrain = 1


print(f"loading DEF")
DEF_performance = cal_DEF_performance(
    "DEF_OPT_template_PTP.csv",
    TMIV_csv_folder_path,
    requested_quality_in_CEL,
)
if istrain == 1:
    DEF_performance = DEF_performance[DEF_performance["Dataset"] != testing_dataset]
else:
    DEF_performance = DEF_performance[DEF_performance["Dataset"] == testing_dataset]
DEF_CEL = DEF_performance["CEL"].mean()
DEF_CEL_STD = DEF_performance["CEL"].std()


print(f"loading OPT")
OPT_performance = cal_OPT_performance(
    "DEF_OPT_template_PTP.csv",
    TMIV_csv_folder_path,
    requested_quality_in_CEL,
)
if istrain == 1:
    OPT_performance = OPT_performance[OPT_performance["Dataset"] != testing_dataset]
else:
    OPT_performance = OPT_performance[OPT_performance["Dataset"] == testing_dataset]
OPT_CEL = OPT_performance["CEL"].mean()
OPT_CEL_STD = OPT_performance["CEL"].std()

print(f"loading DRL")
prediction_csv = (
    f"..\\NN-based_algorithm_seleted_predict_result\\PTP\\{testing_dataset}-6_obj_nsv_best.csv"
)
DRL_performance = cal_prediction_performance(
    DRL_prediction_csv, TMIV_csv_folder_path, requested_quality_in_CEL
)
if istrain == 1:
    DRL_performance = DRL_performance[DRL_performance["Dataset"] != testing_dataset]
else:
    DRL_performance = DRL_performance[DRL_performance["Dataset"] == testing_dataset]
DRL_CEL = DRL_performance["CEL"].mean()
DRL_CEL_STD = DRL_performance["CEL"].std()

print(f"loading CNN")
CNN_prediction_csv = (
    f"..\\NN-based_algorithm_seleted_predict_result\\PTP\\{testing_dataset}-6-cnn_obj_nsv.csv"
)
CNN_performance = cal_prediction_performance(
    CNN_prediction_csv, TMIV_csv_folder_path, requested_quality_in_CEL
)
if istrain == 1:
    CNN_performance = CNN_performance[CNN_performance["Dataset"] != testing_dataset]
else:
    CNN_performance = CNN_performance[CNN_performance["Dataset"] == testing_dataset]
CNN_CEL = CNN_performance["CEL"].mean()
CNN_CEL_STD = CNN_performance["CEL"].std()
sns.set_theme(style="whitegrid")
sns.barplot(
    x=["DEF", "CNN", "DRL"],
    y=[DEF_CEL / OPT_CEL, CNN_CEL / OPT_CEL, DRL_CEL / OPT_CEL],
    ci=[DEF_CEL_STD / OPT_CEL, CNN_CEL_STD / OPT_CEL, DRL_CEL_STD / OPT_CEL],
)

# %%
print([DEF_CEL / OPT_CEL, CNN_CEL / OPT_CEL, DRL_CEL / OPT_CEL])
print([DEF_CEL_STD / OPT_CEL, CNN_CEL_STD / OPT_CEL, DRL_CEL_STD / OPT_CEL])
# %%
