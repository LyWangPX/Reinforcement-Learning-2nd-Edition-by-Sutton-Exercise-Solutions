import numpy as np 
import matplotlib.pyplot as plt 
def RandomWalk(x):
    #x is a vector, each element takes random walk independently, this function return a new vector where each element takes a step by the rule of  random walk
    dim=np.size(x)
    walk_set=[-1,1,0]
    for i in range(dim):
        x[i]=x[i]+np.random.choice(walk_set)
    return x

def eps_greedy(epsilon, Q):
    # This function return an action chosen by epsilon greedy algorithm given the current action value estimate is Q
    i=np.argmax(Q)
    dim=np.size(Q)
    action_space=range(0,dim,1)
    sample=np.random.uniform(0,1)
    if sample<=1-epsilon:
        return i
    else:
       np.delete(action_space,i)
       return np.random.choice(action_space)

def multi_task(max_iter,task_number,epsilon,arm_number,step_size):
    rows, cols = task_number,arm_number
    my_matrix =np.array( [([0]*cols) for i in range(rows)] )
    constQ=np.array( [([0]*cols) for i in range(rows)] )
    variaQ=np.array( [([0]*cols) for i in range(rows)] )
    q=np.array( [([0]*cols) for i in range(rows)] )
    constN=np.array( [([0]*cols) for i in range(rows)] )
    variaN=np.array( [([0]*cols) for i in range(rows)] )
    constR=np.zeros(max_iter)
    variaR=np.zeros(max_iter)
    for i in range(max_iter):
        for j in range(task_number):
            #random walk of each arm
            task_q=q[j,:]
            task_q=RandomWalk(task_q)
            q[j,:]=task_q
            #constant stepsize
            
            task_constQ=constQ[j,:]
            task_constN=constN[j,:]
            action_const=eps_greedy(epsilon,task_constQ)
            
            RewardConst=task_q[action_const]
            constR[i]=constR[i]+RewardConst
            task_constN[action_const]=task_constN[action_const]+1
            alpha=step_size
            difference=RewardConst-task_constQ[action_const]
            task_constQ[action_const]=task_constQ[action_const]+alpha*difference
            constQ[j,:]=task_constQ
            constN[j,:]=task_constN
            #Changing stepsize
            task_variaQ=variaQ[j,:]
            task_variaN=variaN[j,:]
            action_varia=eps_greedy(epsilon,task_variaQ)
            reward_varia=task_q[action_varia]
            task_variaN[action_varia]=task_variaN[action_varia]+1
            if i==0:
                beta=1
            else:
                beta=1/task_variaN[action_varia]
            task_variaQ[action_varia]=task_variaQ[action_varia]+beta*(reward_varia-task_variaQ[action_varia])
            
            variaN[j,:]=task_variaN
            variaQ[j,:]=task_variaQ
            variaR[i]=variaR[i]+reward_varia
        variaR[i]=variaR[i]/task_number
        constR[i]=constR[i]/task_number
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.plot(variaR,color='r')
    plt.plot(constR,color='b')
    plt.xticks(np.arange(0,max_iter+1,100))
    #plt.xticks(np.arange(len(constR)), np.arange(100, len(constR)+1) )
    # plt.grid()
    plt.show()
    plt.close()
    print(q)
    print(constQ)
    print(variaQ)

max_iter=1000
task_number=500
epsilon=0.1
arm_number=10
step_size=0.1
multi_task(max_iter,task_number,epsilon,arm_number,step_size)