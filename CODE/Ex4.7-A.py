import numpy as np
import pickle
import matplotlib.pyplot as plt

"""
This implementation is a pure replication of Example 4.2 in the book
The modified version as the answer to Exercise 4.7 will be posted as Ex4.7-B.py later.    
"""


def poisson_calculator(Lambda=3):
    """
    input:
        lambda: the expectation number of poisson random distribution
    return:
        A dictionary: {value:possibility}
    """
    result = {}
    for i in range(0, 21):
        result[i] = max(np.finfo(float).eps, abs((np.power(Lambda, i) / (np.math.factorial(i))) * np.exp(-Lambda)))
    return result


def P_calculate(all_possibility):
    """

    :param all_possibility: Dict :{(customer_A, returned_car_A,customer_B, returned_car_B):Joint Possibility}
    :return: P: Dict: {((state_value_A,state_value_B),a): reward_dict}
             reward_dict = {reward: Joint Possibility}
    S_T --> (day)sell --> (night)returned ---> policy action -->S_{T+1}
    """
    for state_value_A in range(21):  # car left at the end of the day at A
        print("State " + str(state_value_A))
        for state_value_B in range(21):  # car left at the end of the day at B
            P = {}
            for action in range(-5, 6):  # action range is from -5 to 5
                temp = {}
                #problem: action=-5 A=1 B=10
                if action <= state_value_A and -action <= state_value_B and action + state_value_B <= 20 and -action + state_value_A <= 20:
                    for customer_A in range(21):  # total customers come at the end of the day at A
                        for customer_B in range(21):  # total customers come at the end of the day at B
                            for returned_car_A in range(21):  # total cars returned at the end of the day at A
                                for returned_car_B in range(21):  # total cars returned at the end of the day at B
                                    value_A_Changed = min(20, state_value_A - min(customer_A,
                                                                                  state_value_A- action) + returned_car_A - action)
                                    value_B_Changed = min(20, state_value_B - min(customer_B,
                                                                                  state_value_B+ action) + returned_car_B + action)
                                    reward = 10 * min(customer_A, state_value_A - action ) + \
                                             10 * min(customer_B, state_value_B + action) - \
                                             abs(action) * 2  # the reason for action here is the current action change the next stroes
                                    temp[((value_A_Changed, value_B_Changed),reward)] = temp.get(
                                        (value_A_Changed, value_B_Changed),
                                        0)
                                    temp[((value_A_Changed, value_B_Changed),reward)]+= all_possibility[
                                        (customer_A, returned_car_A, customer_B, returned_car_B)]
                    P[action] = temp
            with open('P' + str(state_value_A)+str('_')+str(state_value_B), 'wb') as f:
                pickle.dump(P, f, protocol=-1)


def policy_evaluation(V, pi, Theta):
    counter = 1
    while True:
        Delta = 0
        print("Calculating loop " + str(counter))
        for i in range(21):
            print("----Calculating " + str(i))
            for j in range(21):
                with open('P' + str(i)+str('_')+str(j), 'rb') as f:
                    p = pickle.load(f)
                a = pi[(i, j)]
                p = p[a]
                old_value = V[(i, j)]
                V[(i, j)] = 0
                for keys, values in p.items():
                    (states, reward) = keys
                    possibility = values
                    V[(i, j)] += (reward + 0.9 * V[states]) * possibility
                Delta = max(Delta, abs(V[(i, j)] - old_value))
        print("Delta = " + str(Delta))
        if Delta < Theta:
            return V
        counter += 1


def policy_improvement(V, pi={}):
    counter = 1
    while True:
        print("Calculating policy loop " + str(counter))
        policy_stable = True
        for keys, old_action in pi.items():
            with open('P' + str(keys[0])+str('_')+str(keys[1]), 'rb') as f:
                p = pickle.load(f)
            possible_q = [0] * 11
            [state_value_A, state_value_B] = keys
            for possible_action in range(-5, 6):
                index = possible_action + 5
                if possible_action <= state_value_A and -possible_action <= state_value_B and possible_action + state_value_B <= 20 and -possible_action + state_value_A <= 20:
                    # print(possible_action)
                    # print(state_value_A, state_value_B)
                    p_a = p[possible_action]
                    for p_keys, values in p_a.items():
                        [states, reward]=p_keys
                        possibility = values
                        possible_q[index] += (reward + 0.9 * V[states]) * possibility
                else:
                    possible_q[index] = -999
            pi[keys] = np.argmax(possible_q) - 5
            if pi[keys] != old_action:
                policy_stable = False
        if policy_stable:
            return pi
        counter += 1


def init():
    customer_A = poisson_calculator(3) # This is the all possible customers from side A and its corresponding possibility
    customer_B = poisson_calculator(4) # This is the all possible customers from side B and its corresponding possibility
    return_A = poisson_calculator(3) # This is the all possible cars returned from side A and its corresponding possiblity
    return_B = poisson_calculator(2) # This is the all possible cars returned from side B and its corresponding possiblity
    all_possibility_A = {}
    all_possibility_B = {}
    all_possibility = {}
    for i in range(21):
        for j in range(21):
            all_possibility_A[(i, j)] = max(np.finfo(float).eps, abs(np.multiply(customer_A[i], return_A[j])))
            all_possibility_B[(i, j)] = max(np.finfo(float).eps, abs(np.multiply(customer_B[i], return_B[j])))
            # min here is to prevent underflow of float. np.finfo(float).eps is exactly the EPS of this machine
            # they are the joint possibility that customers and returned cars both happen.
    for i in range(21):
        for j in range(21):
            for m in range(21):
                for n in range(21):
                    all_possibility[(i, j, m, n)] = max(np.finfo(float).eps,
                                                        abs(all_possibility_A[i, j] * all_possibility_B[m, n]))
    with open('all_possibility', 'wb') as f:
        pickle.dump(all_possibility, f, protocol=-1)
    # with open('all_possibility', 'rb') as f:
    #     all_possibility=pickle.load(f)
    P_calculate(all_possibility)


def train():
    V = {}
    for i in range(21):
        for j in range(21):
            V[(i, j)] = 10 * np.random.random()
    pi = {}
    for i in range(21):
        for j in range(21):
            pi[(i, j)] = 0
    for q in range(5):
        print("Big loop "+str(q))
        V = policy_evaluation(V, pi, Theta=0.01)
        pi = policy_improvement(V, pi)
        with open('pi'+str(q), 'wb') as f:
            pickle.dump(pi, f, protocol=-1)
        with open('V'+str(q), 'wb') as v:
            pickle.dump(V, v, protocol=-1)
        print("================")
        for i in range(21):
            print("i = " + str(i))
            for j in range(21):
                print("  " + str(pi[i, j]))

def main():
    init()
    train()


if __name__ == "__main__":
    main()
