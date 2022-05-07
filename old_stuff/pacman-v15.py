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
       [0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0],
       [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
       [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
       [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
       [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]



##############################
### Classe des personnages ###
##########################
class Character: 



    def __init__(self, x, y, aim):
        self.x = x
        self.y = y
        self.aim = aim
    


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
        return Character(self.x, self.y, self.aim)
    
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
    name = 'Pacman_v11'


    def __init__(self,
                map = np.array(MAP),
                pacman_x = 5, pacman_y = 5,
                ghosts_x = [1, 9, 1, 9], ghosts_y = [1, 9, 9, 1],
                ghosts_aim = [0, 1, 2, 3],
                ghosts_number = 4,
                time_max = 150,
                lifes = 1
                ):

        super(PacmanEnv, self).__init__()
        self.map = map
        self.pacman_init = Character(pacman_x, pacman_y, None)
        self.ghosts_number = range(ghosts_number)
        self.ghosts_init = [Character(ghosts_x[i], ghosts_y[i], ghosts_aim[i]) for i in range(len(ghosts_x))]
        self.time_max = time_max
        self.lifes_init = lifes

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.float32(0), np.float32(1), (self.map.shape[0], self.map.shape[1], 4))
        return



    def change_ghosts_number(self, ghosts_number):
        self.ghosts_number = range(ghosts_number)
        return



    def reset(self):
        self.time = 0
        self.score = 0
        self.lifes = self.lifes_init
        self.done = False

        self.pacman = self.pacman_init.copy()
        self.ghosts = [self.ghosts_init[i].copy() for i in self.ghosts_number]

        self.obs = np.zeros((self.map.shape[0], self.map.shape[1], 4))
        self.obs[:, :, 0] = np.where(self.map == 0, 1, 0)
        self.obs[:, :, 1] = np.where(self.map == 1, 1, 0)
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
        return
    


    def step(self, action):
        reward = 0

        prev_pacman = self.pacman.copy()
        self._take_action(action)
        if action in self.possible_aim(self.pacman):
            self.pacman.move()
            reward += 1
        else:
            reward += -1
        
        for i in self.ghosts_number:
            prev_ghost = self.ghosts[i].copy()
            paim = self.possible_aim(self.ghosts[i])
            if self.ghosts[i].aim_back() in paim:
                paim.remove(self.ghosts[i].aim_back())
            self.ghosts[i].aim = np.random.choice(paim)
            self.ghosts[i].move()

            crossing = self.pacman.same(prev_ghost) and prev_pacman.same(self.ghosts[i])
            if self.pacman.same(self.ghosts[i]) or crossing:
                reward += -1 # + self.time - self.time_max 
                self.lifes += -1
                if self.lifes == 0:
                    self.done = True
                else :
                    self.pacman = self.pacman_init.copy()
                    self.ghosts = [self.ghosts_init[i].copy() for i in self.ghosts_number]
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



    def render(self, mode='humain', close = False):
        rendu = np.where(self.obs[:, :, 0] == 1, "///", "   ") # murs
        rendu = np.where(self.obs[:, :, 1] == 1, " . ", rendu) # pacgommes
        rendu = np.where(self.obs[:, :, 2] == 1, " P ", rendu) # pacman
        rendu = np.where(self.obs[:, :, 3] == 1, " G ", rendu) # fantomes

        text = "score : " + str(self.score) + " | lifes : " + str(self.lifes) + " | time : " + str(self.time) + "\n"

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[0]):
                text += rendu[i, j]
            text += "\n"
        text += "action : " + ["^", "v", "<", ">"][self.pacman.aim] + '\n'
        if self.done :
            text += "\nGAME OVER\n\n"
        return text
