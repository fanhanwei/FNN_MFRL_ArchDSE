import os
import torch
import random


def fold_maker(path):
    if not os.path.exists(path):
            os.makedirs(path)
    fold_number = 1
    while  os.path.exists(os.path.join(path, str(fold_number))):
        fold_number += 1
    sub_path = os.path.join(path, str(fold_number))
    os.makedirs(sub_path)
    return sub_path

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    
class record():
    def __init__(self, log_path, params, args):
        self.log_path = log_path
        # with open(os.path.join(self.log_path, "step_cpi.txt"),"w") as f:
        #     f.write("")
        # with open(os.path.join(self.log_path, "step_area.txt"),"w") as f:
        #     f.write("")
        # with open(os.path.join(self.log_path, "step_reward.txt"),"w") as f:
        #     f.write("")
        # with open(os.path.join(self.log_path, "reward_episode.txt"),"w") as f:
        #     f.write("")
        with open(os.path.join(self.log_path, "final_cpi.txt"),"w") as f:
            f.write("") 
        # with open(os.path.join(self.log_path, "updates.txt"),"w") as f:
        #     f.write("inital params: {}\n".format([int(x) for x in params]))
        #     f.write("args: {}\n\n".format(args))
        self.pool = []
        self.pool_extra = []
            
    def store(self, step, step_reward, metrics, area, y, updates, params, sigma, grads=[]):
        self.pool.append([step, y, [round(x,4) for x in grads], updates, params, metrics[0], metrics[1], metrics[2], area, step_reward, sigma])
        # with open(os.path.join(self.log_path, "step_reward.txt"),"a") as f:
        #     f.write("{}\n".format(step_reward))
        # with open(os.path.join(self.log_path, "step_cpi.txt"),"a") as f:
        #     f.write("{}\n".format(metrics[0]))
        # with open(os.path.join(self.log_path, "step_area.txt"),"a") as f:
        #     f.write("{}\n".format(area))        
    
    def store_extra(self, item):
        self.pool_extra.append(item)
    
    def write(self, episode, episode_cpi, episode_area, episode_reward):
        with open(os.path.join(self.log_path, "final_cpi.txt"),"a") as f:
            f.write("{}\n".format(episode_cpi))       
        # with open(os.path.join(self.log_path, "updates.txt"),"a") as f:
        #     f.write('\n--- episode: {} ---\n'.format(str(episode)))
        #     for item, (name, data) in zip(self.pool, self.pool_extra):
        #         f.write("step{}: \ny_preds: {}\ngrads: {}\nupdates: {}\nparams: {}\ncpi: {}, l1miss: {}, l2miss: {}, area: {}, step_reward: {}, sigma{}, importance: {}\n".format(
        #             item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], item[10], round(item[9]+episode_reward, 2))) 
        #         f.write("{}: {}  ".format(name, data))
        #         f.write("\n")
        #     f.write("end of episode:\nfinal cpi: {},  final area: {},  episode_reward: {}\n\n".format(episode_cpi, episode_area, episode_reward))
        self.pool = []
        self.pool_extra = []
        # with open(os.path.join(self.log_path, "reward_episode.txt"),"a") as f:
        #         f.write("{}\n".format(episode_reward))
        