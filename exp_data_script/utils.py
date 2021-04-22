# %%
import pandas as pd


def cal_CEL(video_quality, running_time, requested_quality_in_CEL):
    return (video_quality - requested_quality_in_CEL) / running_time


def cal_prediction_performance(prediction_csv, TMIV_csv_folder_path, requested_quality_in_CEL):
    performance = {"theo_time": [], "CEL": [], "WS.PSNR": []}
    # read prediction output
    prediction_csv_data = pd.read_csv(prediction_csv)
    # search performance from TMIV data
    for index, row in prediction_csv_data.iterrows():
        TMIV_csv_data = pd.read_csv(f"{TMIV_csv_folder_path}/{row['Dataset']}.csv")
        TMIV_query_row = TMIV_csv_data.loc[
            (TMIV_csv_data["Synthesized.View"] == row["Synthesized.View"])
            & (TMIV_csv_data["Frame"] == row["Frame"])
            & (TMIV_csv_data["p1"] == row["p1"])
            & (TMIV_csv_data["p2"] == row["p2"])
            & (TMIV_csv_data["p3"] == row["p3"])
        ]
        performance["theo_time"].append(TMIV_query_row["theo_time"])
        performance["CEL"].append(
            cal_CEL(
                TMIV_query_row["WS.PSNR"].iloc[0],
                TMIV_query_row["theo_time"].iloc[0],
                requested_quality_in_CEL,
            )
        )
        performance["WS.PSNR"].append(TMIV_query_row["WS.PSNR"])
    # add in original csv, and output Dataframe
    prediction_csv_data["theo_time"] = performance["theo_time"]
    prediction_csv_data["CEL"] = performance["CEL"]
    prediction_csv_data["WS.PSNR"] = performance["WS.PSNR"]
    return prediction_csv_data


def cal_DEF_performance(DEF_csv, TMIV_csv_folder_path, requested_quality_in_CEL):
    performance = {"theo_time": [], "CEL": [], "WS.PSNR": []}
    # read template, and set numberofViewPerPass as the default setting
    DEF_csv_data = pd.read_csv(DEF_csv)
    DEF_csv_data["p1"] = 2
    DEF_csv_data["p2"] = 4
    DEF_csv_data["p3"] = 7
    # search performance from TMIV data
    for index, row in DEF_csv_data.iterrows():
        TMIV_csv_data = pd.read_csv(f"{TMIV_csv_folder_path}/{row['Dataset']}.csv")
        TMIV_query_row = TMIV_csv_data.loc[
            (TMIV_csv_data["Synthesized.View"] == row["Synthesized.View"])
            & (TMIV_csv_data["Frame"] == row["Frame"])
            & (TMIV_csv_data["p1"] == 2)
            & (TMIV_csv_data["p2"] == 4)
            & (TMIV_csv_data["p3"] == 7)
        ]
        performance["theo_time"].append(TMIV_query_row["theo_time"])
        performance["CEL"].append(
            cal_CEL(
                TMIV_query_row["WS.PSNR"].iloc[0],
                TMIV_query_row["theo_time"].iloc[0],
                requested_quality_in_CEL,
            )
        )
        performance["WS.PSNR"].append(TMIV_query_row["WS.PSNR"])
    # add in original csv, and output Dataframe
    DEF_csv_data["theo_time"] = performance["theo_time"]
    DEF_csv_data["CEL"] = performance["CEL"]
    DEF_csv_data["WS.PSNR"] = performance["WS.PSNR"]
    return DEF_csv_data


def cal_OPT_performance(OPT_csv, TMIV_csv_folder_path, requested_quality_in_CEL):
    performance = {"theo_time": [], "CEL": [], "WS.PSNR": [], "p1": [], "p2": [], "p3": []}
    # read template
    OPT_csv_data = pd.read_csv(OPT_csv)
    # search optimal performance from TMIV data
    for index, row in OPT_csv_data.iterrows():
        TMIV_csv_data = pd.read_csv(f"{TMIV_csv_folder_path}/{row['Dataset']}.csv")
        TMIV_csv_data["CEL"] = cal_CEL(
            TMIV_csv_data["WS.PSNR"], TMIV_csv_data["theo_time"], requested_quality_in_CEL
        )
        TMIV_query_rows = TMIV_csv_data.loc[
            (TMIV_csv_data["Synthesized.View"] == row["Synthesized.View"])
            & (TMIV_csv_data["Frame"] == row["Frame"])
        ]
        TMIV_query_row = TMIV_query_rows.iloc[pd.to_numeric(TMIV_query_rows["CEL"]).argmax()]
        performance["theo_time"].append(TMIV_query_row["theo_time"])
        performance["CEL"].append(
            cal_CEL(
                TMIV_query_row["WS.PSNR"],
                TMIV_query_row["theo_time"],
                requested_quality_in_CEL,
            )
        )
        performance["WS.PSNR"].append(TMIV_query_row["WS.PSNR"])
        performance["p1"].append(TMIV_query_row["p1"])
        performance["p2"].append(TMIV_query_row["p2"])
        performance["p3"].append(TMIV_query_row["p3"])
    # add in original csv, and output Dataframe
    OPT_csv_data["theo_time"] = performance["theo_time"]
    OPT_csv_data["CEL"] = performance["CEL"]
    OPT_csv_data["WS.PSNR"] = performance["WS.PSNR"]
    return OPT_csv_data


# %%
