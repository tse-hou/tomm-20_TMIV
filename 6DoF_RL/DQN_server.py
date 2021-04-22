# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
from tqdm import tqdm
from maze_env import Maze
from RL_brain import DeepQNetwork
import pandas as pd
import time

"""setting"""
# Modify by Tsehou 20201023, to let program see the specific GPU
gpus = [3]  # Here I set CUDA to only see one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpus])


# %%
# exp1
# [20200115] (1) test remove texture, (2) cancel the state encoder
# # input: D, r_n
# exp7 (based in exp6)
# [20200220] (1) change the 2nd conv blks to res blk, (2) add a conv block before flatten
# total_parameters 48129544
# input: I, D, r_n
# reward = next utiliry - cur utiliry
# exp8 (based in exp3)
# [20200220] (1) remove batch normalization layer, (2) expend fc layer
# total_parameters 48129544
# input: I, D, r_n
# reward = next utiliry - cur utiliry


# %%
test = "PTP_QJ18_4"
# BEST_OR_LAST = 'best'
BEST_OR_LAST = "last"
train = "PTP_QJ18_4"

# %%
def run_maze():
    step = 0
    opt_rate = -100
    if RL.is_train:
        fout = open(f"./opt_hist/{train}_adam.csv", "w")
        fout.write(str("opt_1,opt_3,opt_5\n"))
    else:
        fout = open("./opt_hist/" + env.test_folder + ".csv", "w")
        fout.write(str("Dataset,Frame,Synthesized View,p1,p2,p3\n"))
    ftout = open("./opt_hist/time.csv", "w")
    top_1 = 0
    top_3 = 0
    top_5 = 0
    done_count = 0
    for episode in tqdm(range(RL.episode)):
        # initial observation
        observation = env.reset()
        start_time = time.time()
        while True:
            # fresh env
            env.render()
            # RL choose action based on observation

            action = RL.choose_action(observation, episode)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            if RL.is_train:
                if done:
                    top_1 = top_1 + env.hit_top_1
                    top_3 = top_3 + env.hit_top_3
                    top_5 = top_5 + env.hit_top_5
                    done_count = done_count + 1

                RL.store_transition(
                    observation[0], action, reward, observation_, observation[1:]
                )

                if (step > 1000) and (step % 10 == 0):
                    is_update = RL.learn()
                    if is_update:
                        temp_top_1_rate = top_1 / done_count
                        temp_top_3_rate = top_3 / done_count
                        temp_top_5_rate = top_5 / done_count
                        if opt_rate <= temp_top_1_rate:
                            opt_rate = temp_top_1_rate
                            # Modify by Tsehou 20201023, for output weight of each model in different foldedr
                            # RL.save_weight(episode, './weights_exp/')
                            RL.save_weight(episode, f"./weights_exp/exp_response_apr/{train}/")

                            print(
                                "[*%d] OPT rate %.4f, %.4f, %.4f"
                                % (
                                    episode,
                                    temp_top_1_rate,
                                    temp_top_3_rate,
                                    temp_top_5_rate,
                                )
                            )
                        else:
                            print(
                                "[%d] OPT rate %.4f, %.4f, %.4f"
                                % (
                                    episode,
                                    temp_top_1_rate,
                                    temp_top_3_rate,
                                    temp_top_5_rate,
                                )
                            )

                        fout.write(
                            str(
                                "%d,%.4f,%.4f,%.4f\n"
                                % (
                                    episode,
                                    temp_top_1_rate,
                                    temp_top_3_rate,
                                    temp_top_5_rate,
                                )
                            )
                        )
                        fout.flush()
                        top_1 = 0
                        top_3 = 0
                        top_5 = 0
                        done_count = 0

            # swap observation
            observation[0] = observation_

            # break while loop when end of this episode
            if done:
                if not RL.is_train:
                    fout.write(
                        "%s,%d,%d,%d,%d,%d\n"
                        % (
                            env.ob.r_db,
                            env.ob.r_frame,
                            env.ob.r_tvs,
                            env.cur_state[0],
                            env.cur_state[1],
                            env.cur_state[2],
                        )
                    )
                break
            step += 1

        dur = time.time() - start_time
        ftout.write(str("%.4f\n" % (dur)))
    # end of game
    print("game over")
    fout.close()
    ftout.close()
    env.destroy()


# %%
# userstudy_and_obj_rand_zero
# userstudy_and_obj_rand
# obj_nsv
if __name__ == "__main__":
    # maze game
    is_train = True
    # is_train = False
    is_monitor = False
    env = Maze(
        is_train=is_train, test_folder="obj_nsv", is_monitor=is_monitor, reward_mag=100
    )
    if is_train:
        RL = DeepQNetwork(
            env.n_actions,
            env.max_passes,
            learning_rate=0.0001,
            reward_decay=1.0,
            e_greedy=0.5,
            replace_target_iter=500,
            memory_size=1000,
            batch_size=32,
            output_graph=True,
            is_train=is_train,
            weight_folder="",
            episode=300000,
            is_monitor=is_monitor,
        )
    else:
        RL = DeepQNetwork(
            env.n_actions,
            env.max_passes,
            learning_rate=0.001,
            reward_decay=1.0,
            e_greedy=1.0,
            replace_target_iter=200,
            memory_size=1000,
            output_graph=True,
            is_train=is_train,
            weight_folder="./weights_exp/" + test + f"/{BEST_OR_LAST}/",
            episode=210,
            is_monitor=is_monitor,
        )
    env.after(100, run_maze)
    env.mainloop()
    # RL.plot_cost()
    if is_train:
        RL.save_weight(20210422, f"./weights_exp/exp_response_apr/{train}/last/")

        df = pd.DataFrame({"loss": RL.cost_his})
        df.to_csv("./opt_hist/" + train + "_adam_loss.csv", index=False)
