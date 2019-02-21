import numpy as np
import torch
import os
import sys
from imageio import mimsave
#from skimage.transform import resize
import cv2

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)


def seed_everything(seed=1234):
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.deterministic = True

def handle_step(random_state, cnt, S_hist, S_prime, action, reward, finished, k_used, acts, episodic_reward, replay_buffer, checkpoint='', n_ensemble=1, bernoulli_p=1.0):
    # mask to determine which head can use this experience
    exp_mask = random_state.binomial(1, bernoulli_p, n_ensemble).astype(np.uint8)
    # at this observed state
    experience =  [S_prime, action, reward, finished, exp_mask, k_used, acts, cnt]
    batch = replay_buffer.send((checkpoint, experience))
    # update so "state" representation is past history_size frames
    S_hist.pop(0)
    S_hist.append(S_prime)
    episodic_reward += reward
    cnt+=1
    return cnt, S_hist, batch, episodic_reward

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """ from dopamine - Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
      Begin at 1. until warmup_steps steps have been taken; then
      Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
      Use epsilon from there on.
    Args:
      decay_period: float, the period over which epsilon is decayed.
      step: int, the number of training steps completed so far.
      warmup_steps: int, the number of steps taken before epsilon is decayed.
      epsilon: float, the final value to which to decay the epsilon parameter.
    Returns:
      A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus

def write_info_file(info, model_base_filepath, cnt):
    info_filename = model_base_filepath + "_%010d_info.txt"%cnt
    info_f = open(info_filename, 'w')
    for (key,val) in info.items():
        info_f.write('%s=%s\n'%(key,val))
    info_f.close()

def generate_gif(base_dir, step_number, frames_for_gif, reward, name='', results=[]):
    """
    from @fg91
        Args:
            step_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    from IPython import embed
    embed()
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = cv2.resize(frame_idx, (320, 220)).astype(np.uint8)

    if len(frames_for_gif[0].shape) == 2:
        name+='gray'
    else:
        name+='color'
    gif_fname = os.path.join(base_dir, "ATARI_step%010d_r%04d_%s.gif"%(step_number, int(reward), name))

    print("WRITING GIF", gif_fname)
    mimsave(gif_fname, frames_for_gif, duration=1/30)
    if len(results):
        txt_fname = gif_fname.replace('.gif', '.txt')
        ff = open(txt_fname, 'w')
        for ex in results:
            ff.write(ex+'\n')
        ff.close()


