import numpy as numpy
from tabulate import tabulate

class State: # class for each unit on the game
    def __init__(self, _id):
        if _id != 0:
            self.value = 0
        elif _id == 0:
            self.value = 0
        self.id = _id
        self.left_bound = max(1, (self.id // 4) * 4)  # save the left border
        self.right_bound =  min(14, (self.id // 4) * 4 + 3) # save the right border
        self.nextS = [self.move('L'), self.move('R'), self.move('U'), self.move('D')]


    def move(self, u):  # get next S
        if u == 'L':  # move left
            if self.id - 1 >= self.left_bound:
                return self.id - 1
            elif self.id - 1 == 0:
                return 0
            else:
                return self.id
        if u == 'R':  # move right
            if self.id + 1 <= self.right_bound:
                return self.id + 1
            elif self.id + 1 == 15:
                return 0
            else:
                return self.id
        if u == 'U':  # move up
            if self.id - 4 >= 1:
                return self.id - 4
            elif self.id - 4 == 0:
                return 0
            else:
                return self.id
        if u == 'D':  # move down
            if self.id + 4 <= 14:
                return self.id + 4
            elif self.id + 4 == 15:
                return 0
            else:
                return self.id

    def update(self, S):  # S is the whole set of the States.
        V = 0
        for i in range(0, 4):
            V += S[self.nextS[i]].value
        # print("update id " + str(self.id))
        # print("for the " + str(self.nextS[i]))
        # print("old value" + str(self.value))
        # print("new value" + str(S[self.nextS[i]].value))
        self.value = -1+0.25 * V

def train(k=10):
    V  = []
    S_T = State(0)
    S = {0: S_T}
    for j in range(1, 15):
        S[j] = State(j)
    for loop in range(k):
        if loop>=1000 and loop%1000 == 0:
            print("Training "+str(loop)+"'s loop.......Remaining: "+str(k-loop)+ " loops")
        n = numpy.random.random()
        if n > 0.5:
            for j in range(1, 15):
                S[j].update(S)
        else:
            for j in range(14, 0, -1):
                S[j].update(S)
    for t in range(0,16):
        if t == 0 or t == 15:
            V.append("0")
        else:
            V.append(S[t].value)
    draw(V)


def draw(valueArray):
    for i in range(4):
        print("----------------------")
        print("| "+str(int(valueArray[i*4]))+" | "+str(int(valueArray[i*4+1])) +" | "+str(int(valueArray[i*4+2])) +" | "+str(int(valueArray[i*4+3])) +" |")
    print("----------------------")
    print("Accurate State Values List:")
    for i in range(1,8):
        print("State "+str(2*i-1)+": "+str(valueArray[2*i-1])+ "          State "+str(2*i)+": "+str(valueArray[2*i]))

if __name__ == '__main__':
    k = input("Specify the desired training loop count(0-10000):")
    train(int(k))
