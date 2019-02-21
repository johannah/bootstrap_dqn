import os
from collections import deque
import numpy as np
from atari_py.ale_python_interface import ALEInterface
import cv2

#from skimage.transform import resize
#from skimage.color import rgb2gray
#from imageio import imwrite
#def preprocess_frame(observ, output_size):
#    return resize(rgb2gray(observ),(output_size, output_size)).astype(np.float32, copy=False)

# opencv is ~3x faster than skimage
def cv_preprocess_frame(observ, output_size):
    gray = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)
    output = cv2.resize(gray, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    return output

class Environment(object):
    def __init__(self,
                 rom_file,
                 frame_skip=4,
                 num_frames=4,
                 frame_size=84,
                 no_op_start=30,
                 rand_seed=393,
                 dead_as_end=True,
                 max_episode_steps=18000,
                 autofire=False):
        self.max_episode_steps = max_episode_steps
        self.random_state = np.random.RandomState(rand_seed+15)
        self.ale = self._init_ale(rand_seed, rom_file)
        # normally (160, 210)
        self.actions = self.ale.getMinimalActionSet()

        self.frame_skip = frame_skip
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.no_op_start = no_op_start
        self.dead_as_end = dead_as_end

        self.total_reward = 0
        screen_width, screen_height = self.ale.getScreenDims()
        self.prev_screen = np.zeros(
            (screen_height, screen_width, 3), dtype=np.uint8)
        self.frame_queue = deque(maxlen=num_frames)
        self.end = True

    @staticmethod
    def _init_ale(rand_seed, rom_file):
        assert os.path.exists(rom_file), '%s does not exists.'
        ale = ALEInterface()
        ale.setInt('random_seed', rand_seed)
        ale.setBool('showinfo', False)
        ale.setInt('frame_skip', 1)
        ale.setFloat('repeat_action_probability', 0.0)
        ale.setBool('color_averaging', False)
        ale.loadROM(rom_file)
        return ale

    @property
    def num_actions(self):
        return len(self.actions)

    def _get_current_frame(self):
        # global glb_counter
        screen = self.ale.getScreenRGB()
        max_screen = np.maximum(self.prev_screen, screen)
        frame = cv_preprocess_frame(max_screen, self.frame_size)
        return frame

    def reset(self):
        self.steps = 0
        self.end = False
        self.plot_frames = []
        self.gray_plot_frames = []
        for _ in range(self.num_frames - 1):
            self.frame_queue.append(
                np.zeros((self.frame_size, self.frame_size), dtype=np.uint8))

        # steps are in steps the agent sees
        self.ale.reset_game()
        self.total_reward = 0
        self.prev_screen = np.zeros(self.prev_screen.shape, dtype=np.uint8)

        n = self.random_state.randint(0, self.no_op_start)
        for i in range(n):
            if i == n - 1:
                self.prev_screen = self.ale.getScreenRGB()
            self.ale.act(0)

        self.frame_queue.append(self._get_current_frame())
        self.plot_frames.append(self.prev_screen)
        a = np.array(self.frame_queue)
        out = np.concatenate((a[0],a[1],a[2],a[3]),axis=0).T
        self.gray_plot_frames.append(out)
        if self.ale.game_over():
            print("Unexpected game over in reset", self.reset())
        return np.array(self.frame_queue)

    def step(self, action_idx):
        """Perform action and return frame sequence and reward.
        Return:
        state: [frames] of length num_frames, 0 if fewer is available
        reward: float
        """
        assert not self.end
        reward = 0
        old_lives = self.ale.lives()

        for i in range(self.frame_skip):
            if i == self.frame_skip - 1:
                self.prev_screen = self.ale.getScreenRGB()
            r = self.ale.act(self.actions[action_idx])
            reward += r
        dead = (self.ale.lives() < old_lives)
        if self.dead_as_end and dead:
            lives_dead = True
        else:
            lives_dead = False

        if self.ale.game_over():
            self.end = True
            lives_dead = True
        self.steps +=1
        if self.steps >= self.max_episode_steps:
            self.end = True
            lives_dead = True
        self.frame_queue.append(self._get_current_frame())
        self.total_reward += reward
        a = np.array(self.frame_queue)
        self.prev_screen = self.ale.getScreenRGB()
        self.gray_plot_frames.append(np.concatenate((a[0],a[1],a[2],a[3]),axis=0))
        self.plot_frames.append(self.prev_screen)
        return np.array(self.frame_queue), reward, lives_dead, self.end


if __name__ == '__main__':

    import time
    env = Environment('roms/breakout.bin', 4, 4, 84, 30, 33, True)
    print('starting with game over?', env.ale.game_over())
    random_state = np.random.RandomState(304)

    state = env.reset()
    i = 0
    times = [0.0 for a in range(1000)]
    total_reward = 0
    for t in times:
        action = random_state.randint(0, env.num_actions)
        st = time.time()
        state, reward, end, do_reset = env.step(action)
        total_reward += reward
        times[i] = time.time()-st
        i += 1
        if end:
            print(i,"END")
        if do_reset:
            print(i,"RESET")
            state = env.reset()
    print('total_reward:', total_reward)
    print('total steps:', i)
    print('mean time', np.mean(times))
    print('max time', np.max(times))
    # with cv - 1000 steps ('mean time', 0.0008075275421142578) max   0.0008950233459472656
    # with skimage -       ('mean time', 0.0022023658752441406) max,  0.003056049346923828
