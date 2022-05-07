import time

f = open("last_game.txt")
f1 = f.readlines()

i = 0
f2 = ''
for x in f1:
    i += 1
    f2 +=x
    if i % 14 == 0:
        print(f2)
        f2 = ''
        time.sleep(0.5)

print(f1[-2])

f.close()