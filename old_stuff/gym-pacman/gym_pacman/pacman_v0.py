"""Pacman, classic arcade game"""

from turtle import * # cause des 16 'erreurs' non problématiques
from random import choice
from math import sqrt

# Fonction utiles pour manipuler les vecteurs
def index_to_vect(index): # donne le vecteur associé à un index
    x = (DIM_X/2 - (index[0]+1)) * SIZE
    y = (index[1] - DIM_Y/2) * SIZE
    return [int(y), int(x)]

def vect_to_index(vect): # donne l'index d'un vecteur
    x = float((vect[0] + DIM_X*SIZE/2) // SIZE) * SIZE - DIM_X*SIZE/2
    y = float((vect[1] + DIM_Y*SIZE/2) // SIZE) * SIZE - DIM_Y*SIZE/2

    x = DIM_X/2 + x/SIZE
    y = DIM_Y/2 - 1 - y/SIZE
    return [int(y), int(x)]

def comb_vect(vect1, vect2, coef1=1, coef2=1): # combinaison linéaire de deux vecteurs
    return ([ coef1*vect1[0] + coef2*vect2[0], coef1*vect1[1] + coef2*vect2[1] ])

def distance(vect1, vect2): # donne la distance entre deux vecteurs
    d = comb_vect(vect2, vect1, coef2 = -1)
    return int(sqrt(d[0]*d[0] + d[1]*d[1]))

def valid(vect): # verifie la validité d'une position avant déplacement
    for add in [[0, 0], [1, 0], [0, 1], [1, 1]]:
        index = vect_to_index(comb_vect(vect, add, coef2 = SIZE-1))
        if map[index[0]][index[1]] == 0:
            return False
    return True

def square(vect): # dessine un carrée en position du vecteur
    path.up()
    path.goto(vect[0],vect[1])
    path.down()
    path.begin_fill()

    path.forward(SIZE)
    path.left(90)
    path.forward(SIZE)
    path.left(90)
    path.forward(SIZE)
    path.left(90)
    path.forward(SIZE)
    path.left(90)

    path.end_fill()

# definition des personnages
class Guy: 

    def __init__(self, name, index, aim, color):
        self.name = name  
        self.position = index_to_vect(index)
        if not valid(self.position):
            print("ERREUR : initial position of "+self.name+" invalid")
        self.aim = aim
        self.new_aim = self.aim
        self.color = color
    
    def change(self, new_aim): # change la direction
        self.new_aim = new_aim

    def move(self): # fait un pas dans la direction si possible
        new_position = comb_vect(self.position, self.new_aim, coef2=STEP) # tente de change de direction
        if valid(new_position):
            self.position = new_position
            self.aim = self.new_aim # valide la nouvelle direction 
        else:
            new_position = comb_vect(self.position, self.aim, coef2=STEP) # tente d'avancer sans changer de direction
            if valid(new_position):
                self.position = new_position
            elif self.name != "Pacman": # si c'est un fantome ne peut avancer in change aléatoirement de direction
                self.change(choice([[1, 0], [-1, 0], [0, 1], [0, -1]]))
                    
# affichage du terrain
def world():
    bgcolor("black") # dessine le font
    path.color("blue") # couleur du labyrinthe

    for i in range(DIM_X):
        for j in range(DIM_Y):
            if map[i][j] > 0: # dessine le labyrinthe
                square(index_to_vect([i,j]))

                if map[i][j] == 1: # dessine les pacgommes
                    path.up()
                    v = comb_vect(index_to_vect([i,j]), [1, 1], coef2=SIZE/2)
                    path.goto(v[0],v[1])
                    path.dot(SIZE/10, "white")


# definition de la boucle d'action principale
def move():
    state['time'] += 1
    clear()
    writer.undo()

    for p in personages:
        p.move() # bouge les personnages

        v = comb_vect(p.position, [1, 1], coef2=SIZE/2)

        if p.name=="Pacman":
            [i, j] = vect_to_index(v)
            if map[i][j] == 1: # pacman mange une pacgomme
                map[i][j] = 2
                state['score'] += 1
                square(index_to_vect([i,j]))
                if state['score'] == MAX_SCORE: # victoire de pacman
                    writer.write("YOU WIN : \nscore : "+str(state['score']))
                    return
        else:
            if distance(p.position, personages[0].position) < SIZE: # victoire des fantomes
                writer.write("GAME OVER : \n"+str(p.name)+" ate you\nscore : "+str(state['score']))
                return

        up() # dessine les personnages
        goto(v[0],v[1])
        dot(SIZE, p.color)

    writer.write("score : "+str(state['score'])+"\ntime : "+str(state['time'])) # affiche les informations

    update() # affiche les modifications
    ontimer(move, TIME_STEP) # attend le prochain pas de temps


# parametres
SIZE = 40 # resolution du rendu
STEP = int(SIZE/4) # pas de déplacement
if SIZE%STEP!=0:
    print("ERREUR: STEP must be multiple of SIZE")
TIME_STEP = 50 # pas de temps

state = {"score": 0, "time":0} # score initale

map = [ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

MAX_SCORE = sum(sum(map,[])) # = 160
DIM_X = len(map)    # = 20
DIM_Y = len(map[0]) # = 20

# initialisation 
path = Turtle(visible=False) 
writer = Turtle(visible=False)
setup((DIM_Y+1)*SIZE, (DIM_X+1)*SIZE, 370, 0)
hideturtle()
tracer(False)
writer.goto(8*SIZE, 8*SIZE)
writer.color('white')
writer.write("score : "+str(state['score'])+"\ntime : "+str(state['time']))

# Initialisation de Pacman et des fantomes
personages = [Guy("Pacman", [10, 10], [ 0,  0], "yellow"),
              Guy("Blinky", [ 1,  1], [ 1,  0],    "red"),
              Guy( "Pinky", [ 1, 15], [ 0,  1],   "pink"),
              Guy(  "Inky", [17,  1], [ 0, -1],   "cyan"),
              Guy( "Clyde", [17, 15], [-1,  0], "orange")]

# liste les controles
listen() 
onkey(lambda: personages[0].change([ 1,  0]), 'Right')
onkey(lambda: personages[0].change([-1,  0]), 'Left')
onkey(lambda: personages[0].change([ 0,  1]), 'Up')
onkey(lambda: personages[0].change([ 0, -1]), 'Down')

# boucle de jeu
world()
move()
done()