import argparse
import os
parser = argparse.ArgumentParser(description='DRL warper')
parser.add_argument('video_type', type=str,help="ERP or PTP")
parser.add_argument('is_training', type=int,help="is training (1) or testing (0)")
parser.add_argument('gpu_idx', type=int,help="which gpu want to use")
parser.add_argument('model_name', type=str,help="the name of model")
args = parser.parse_args()

if (args.video_type == "ERP"):
    os.system("cp utils_equi.py utils.py")
elif(args.video_type == "PTP" and args.is_training == 1):
    os.system("cp utils_pers_train.py utils.py")
elif(args.video_type == "PTP" and args.is_training == 0):
    os.system("cp utils_pers_test.py utils.py")

os.system(f"xvfb-run -a python DQN_server.py {args.gpu_idx} {args.model_name}")