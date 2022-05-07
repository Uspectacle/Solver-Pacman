"""Pacman, classic arcade game"""




##############################
### Importation de modules ###
##############################

import numpy as np
import gym
from gym import spaces



############################
### Création de la carte ###
############################

"""
█████████████████████████████████
███ X          ███          X ███
███   ██████   ███   ██████   ███
███   ███               ███   ███
███   ███   █████████   ███   ███
███             P             ███
███   ███   █████████   ███   ███
███   ███               ███   ███
███   ██████   ███   ██████   ███
███ X          ███          X ███
█████████████████████████████████
"""
MAP = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
       [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
       [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
       [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
       [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
       [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
       [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]



##############################
### Classe des personnages ###
##########################
class Character: 

    def __init__(self, index):
        self.x = index[0]
        self.y = index[1]
        self.aim = 0

    def move(self):
        if self.aim == 0:
            self.x += -1
        if self.aim == 1:
            self.x += +1
        if self.aim == 2:
            self.y += -1
        if self.aim == 3:
            self.y += +1
        return
    
    def same(self, character):
        return self.x == character.x and self.y == character.y
    
    def copy(self):
        return Character([self.x, self.y])
    
    def aim_back(self):
        if self.aim == 0:
            return 1
        if self.aim == 1:
            return 0
        if self.aim == 2:
            return 3
        if self.aim == 3:
            return 2



####################
### Environement ###
####################

class PacmanEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    name = 'Pacman_v16'

    def __init__(self,
                map = np.array(MAP),
                ghosts_number = 1,
                time_max = 150,
                lifes = 1
                ):

        super(PacmanEnv, self).__init__()
        self.map = map
        self.ghosts_number = ghosts_number
        self.time_max = time_max
        self.lifes_init = lifes

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.float32(0), np.float32(1), (self.map.shape[0], self.map.shape[1], 4))

    def reset(self):
        self.time = 0
        self.score = 0
        self.lifes = self.lifes_init
        self.done = False
        self.map = self.map.transpose()

        possibles_index = np.argwhere(self.map == 1)
        index = possibles_index[np.random.choice(possibles_index.shape[0], 1 + self.ghosts_number, replace=False)]
        self.pacman_init = Character(index[0])
        self.ghosts_init = [Character(index[i+1]) for i in range(self.ghosts_number)]
        self.pacman = self.pacman_init.copy()
        self.ghosts = [self.ghosts_init[i].copy() for i in range(self.ghosts_number)]

        self.obs = np.zeros((self.map.shape[0], self.map.shape[1], 4))
        self.obs[:, :, 0] = np.where(self.map == 0, 1, 0)
        self.obs[:, :, 1] = np.where(self.map == 1, 1, 0)
        self.obs[index[0, 0], index[0, 1], 1] = 0
        return self._next_observation()
    
    def _next_observation(self):
        next_obs = np.zeros(self.obs.shape)
        next_obs[:, :, 0] = self.obs[:, :, 0] # murs
        next_obs[:, :, 1] = self.obs[:, :, 1] # pacgommes
        next_obs[self.pacman.x, self.pacman.y, 2] = 1 # pacman
        for ghost in self.ghosts :
            next_obs[ghost.x, ghost.y, 3] = 1 # fantomes
        self.obs = next_obs
        return self.obs
    
    def possible_aim(self, character):
        paim = [self.obs[character.x - 1, character.y, 0] == 0,
                self.obs[character.x + 1, character.y, 0] == 0, 
                self.obs[character.x, character.y - 1, 0] == 0, 
                self.obs[character.x, character.y + 1, 0] == 0]
        return [i for i in range(4) if paim[i]]

    def _take_action(self, action):
        self.pacman.aim = action
    
    def step(self, action):
        reward = 0

        prev_pacman = self.pacman.copy()
        self._take_action(action)
        if action in self.possible_aim(self.pacman):
            self.pacman.move()
            reward += 0 # 1
        else:
            reward += 0 # -1
        
        for i in range(self.ghosts_number):
            prev_ghost = self.ghosts[i].copy()
            paim = self.possible_aim(self.ghosts[i])
            if self.ghosts[i].aim_back() in paim:
                paim.remove(self.ghosts[i].aim_back())
            self.ghosts[i].aim = np.random.choice(paim)
            self.ghosts[i].move()

            crossing = self.pacman.same(prev_ghost) and prev_pacman.same(self.ghosts[i])
            if self.pacman.same(self.ghosts[i]) or crossing:
                reward += 0 # -3 # + self.time - self.time_max 
                self.lifes += -1
                if self.lifes == 0:
                    self.done = True
                else :
                    self.pacman = self.pacman_init.copy()
                    self.ghosts = [self.ghosts_init[i].copy() for i in range(self.ghosts_number)]
                    break
        
        if self.obs[self.pacman.x, self.pacman.y, 1] == 1:
            self.obs[self.pacman.x, self.pacman.y, 1] = 0
            self.score += 1
            reward += 1
            if np.sum(self.obs[:, :, 1]) == 0:
                print("It's a win")
                self.done = True
                reward += 0 # 10 # self.time - self.time_max 

        self.obs = self._next_observation()

        self.time += 1
        if self.time == self.time_max:
            self.done = True

        return self.obs, reward, self.done, {}

    def render(self, mode='humain', close = False, my_obs_bool=False, my_obs=None):
        if my_obs_bool:
            obs = my_obs
            # print("wiw")
        else: 
            obs = self.obs
        rendu = np.where(obs[:, :, 0] == 1, "///", "   ") # murs
        rendu = np.where(obs[:, :, 1] == 1, " . ", rendu) # pacgommes
        rendu = np.where(obs[:, :, 2] == 1, " P ", rendu) # pacman
        rendu = np.where(obs[:, :, 3] == 1, " G ", rendu) # fantomes

        text = "       action : " + ["^", "v", "<", ">"][self.pacman.aim] + " | score : " + str(self.score) + " | lifes : " + str(self.lifes) + " | time : " + str(self.time) + "\n"

        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                text += rendu[i, j]
            text += "\n"
        if self.done :
            text += "\nGAME OVER\n\n"
        return text