#%%
import pandas as pd
from sys import argv
from pathlib import Path

#%%
def convert_CEL(csv, requested_quality):
    csv_data = pd.read_csv(csv)
    csv_data["CEL"] = (csv_data["WS.PSNR"] - requested_quality) / csv_data["theo_time"]
    print(csv_data)
    return csv_data


def main():
    requested_quality = argv[1]
    output_folder = "../datasets/states/train"
    # PTP
    p = Path("../datasets/states")
    csv_list = list(p.glob("train_original/*.csv"))
    # ERP
    p = Path("../datasets_equi/states")
    csv_list += list(p.glob("train/*.csv"))
    for csv in csv_list:
        converted_csv_data = convert_CEL(csv, requested_quality)
        converted_csv_data.to_csv(f"{output_folder}/{csv.name}", index=False)


if __name__ == "__main__":
    main()

# %%
