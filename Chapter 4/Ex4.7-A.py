from dataclasses import dataclass
from functools import lru_cache

import numpy as np


MAX_CARS = 20
RENTAL_COST = 10
CAR_MOVE_OVERNIGHT_COST = 2
MAX_MOVES_OVERNIGHT = 5
EXPECTED_RENTALS = (3, 4)
EXPECTED_RETURNS = (3, 2)
ONE_FREE_R2_TO_R1_SHUTTLE = True    # Specific to ex 4.7
LIMITED_PARKING_SPACE = True        # Specific to ex 4.7
MAX_FREE_PARKING = 10
PARKING_COST = 4


@dataclass(eq=True, frozen=True)
class State:
    n_cars_r1: int
    n_cars_r2: int


@lru_cache()
def _factorial(n):
    return _factorial(n - 1) * n if n else 1


@lru_cache()
def _poisson(n, lambd):
    return ((lambd ** n) / _factorial(n)) * np.exp(-lambd)


@lru_cache()
def _rental_dynamics(n_cars_avail: int, rental_lamb: int, return_lamb: int):
    # Compute the individual probabilities of getting n returns and n rentals
    return_prob = np.array([_poisson(n, return_lamb) for n in range(MAX_CARS)])
    rental_prob = np.array([_poisson(n, rental_lamb) for n in range(MAX_CARS)])

    state_reward_probs = {}
    for n_returns in range(MAX_CARS):
        for n_rentals in range(MAX_CARS):
            # The probability of both n_returns and n_rentals occuring is the product of each's probability
            prob = return_prob[n_returns] * rental_prob[n_rentals]

            # First we get back cars from returns (capped to SIZE)
            next_n_cars_avail = min(n_cars_avail + n_returns, MAX_CARS)

            # Then we rent cars throughout the day
            reward = RENTAL_COST * min(next_n_cars_avail, n_rentals)
            next_n_cars_avail = max(0, next_n_cars_avail - n_rentals)

            # Add this probability to the current dictionary
            key = (next_n_cars_avail, reward)
            state_reward_probs[key] = state_reward_probs.get(key, 0.) + prob

    # Restructure the data as numpy arrays for efficiency
    n_cars, rewards, probs = map(
        np.array, zip(*[(nc, r, p) for (nc, r), p in state_reward_probs.items()])
    )

    return n_cars, rewards, probs


@lru_cache()
def _day_dynamics(state: State):
    # Simulate all possible outcomes for each rental
    r1_n_cars, r1_rewards, r1_probs = _rental_dynamics(state.n_cars_r1, EXPECTED_RENTALS[0], EXPECTED_RETURNS[0])
    r2_n_cars, r2_rewards, r2_probs = _rental_dynamics(state.n_cars_r2, EXPECTED_RENTALS[1], EXPECTED_RETURNS[1])

    # Combine the outcomes of both rentals (numpy version, faster)
    rewards = (r1_rewards[:, None] + r2_rewards[None, :]).flatten()
    probs = (r1_probs[:, None] * r2_probs[None, :]).flatten()
    comb_r1_n_cars = np.repeat(r1_n_cars, len(r2_n_cars))
    comb_r2_n_cars = np.tile(r2_n_cars, len(r1_n_cars))

    # # Combine the outcomes of both rentals (python version, simpler)
    # outcomes = []
    # for r1_n_cars_, r1_reward, r1_prob in zip(r1_n_cars, r1_rewards, r1_probs):
    #     for r2_n_cars_, r2_reward, r2_prob in zip(r2_n_cars, r2_rewards, r2_probs):
    #         prob = r1_prob * r2_prob
    #         reward = r1_reward + r2_reward
    #         outcomes.append((r1_n_cars_, r2_n_cars_, reward, prob))
    #
    # # Restructure the data as numpy arrays for efficiency
    # comb_r1_n_cars, comb_r2_n_cars, rewards, probs = map(np.array, zip(*outcomes))

    return comb_r1_n_cars, comb_r2_n_cars, rewards, probs


@lru_cache()
def dynamics(state: State, action):
    # Move the cars across locations
    new_state = State(state.n_cars_r1 - action, state.n_cars_r2 + action)

    # Simulate the possible outcomes of the day
    r1_n_cars, r2_n_cars, rewards, probs = _day_dynamics(new_state)

    # Cost of moving cars overnight
    if ONE_FREE_R2_TO_R1_SHUTTLE and action > 0:
        overnight_cost = -CAR_MOVE_OVERNIGHT_COST * (action - 1)
    else:
        overnight_cost = -CAR_MOVE_OVERNIGHT_COST * abs(action)

    # Cost of the parking space
    parking_cost = 0
    if LIMITED_PARKING_SPACE:
        parking_cost = np.zeros_like(r1_n_cars)
        parking_cost -= (r1_n_cars > MAX_FREE_PARKING) * PARKING_COST
        parking_cost -= (r2_n_cars > MAX_FREE_PARKING) * PARKING_COST

    # Add the costs to the rewards
    rewards = rewards.copy() + overnight_cost + parking_cost

    return r1_n_cars, r2_n_cars, rewards, probs


def state_action_value(state: State, action: int, values, discount=0.9):
    # Simulate the next states according to the policy
    r1_n_cars, r2_n_cars, rewards, probs = dynamics(state, action)

    # Get the estimated value of each state
    new_states_value = values[r1_n_cars, r2_n_cars]
    # Discount the values, add the rewards, weight by the probabilities and return the sum of the whole
    return (probs * (rewards + discount * new_states_value)).sum()


def possible_actions(state: State):
    # Positive: from #1 to #2
    # Negative: from #2 to #1
    return list(range(-min(state.n_cars_r2, MAX_MOVES_OVERNIGHT), min(state.n_cars_r1, MAX_MOVES_OVERNIGHT) + 1))


def possible_states():
    for i in range(MAX_CARS + 1):
        for j in range(MAX_CARS + 1):
            yield State(n_cars_r1=i, n_cars_r2=j)


def policy_state_values(policy, discount=0.9, eps=0.01):
    values = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

    for k in range(1, 10 ** 100):
        stable = True

        for state in possible_states():
            action = policy[state.n_cars_r1, state.n_cars_r2]
            new_value = state_action_value(state, action, values, discount)

            delta = abs(new_value - values[state.n_cars_r1, state.n_cars_r2])
            if delta >= eps:
                stable = False

            values[state.n_cars_r1, state.n_cars_r2] = new_value

        if stable:
            print(f"Converged in {k} steps")
            return values


def greedy_action(state: State, values, discount=0.9):
    action_values = {
        action: state_action_value(state, action, values, discount)
        for action in possible_actions(state)
    }

    max_value = max(action_values.values())
    return next(k for k, v in action_values.items() if v == max_value)


def policy_iteration_state_values():
    policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.int32)

    for k in range(10 ** 100):
        print(f"--- Policy {k} ---")
        print("Actions:")
        print(policy)
        print("Values:")
        values = policy_state_values(policy)
        print(values)
        print("\n")

        improved_policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.int32)
        for state in possible_states():
            improved_policy[state.n_cars_r1, state.n_cars_r2] = greedy_action(state, values)

        if np.array_equal(policy, improved_policy):
            return policy, values
        policy = improved_policy


def plot_policy_values(policy, values):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Optimal policy for the car rental problem")

    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Policy actions")
    cmap = plt.get_cmap('RdYlGn', 2 * MAX_MOVES_OVERNIGHT + 1)
    im = ax.imshow(
        policy, origin="lower",
        vmin=-MAX_MOVES_OVERNIGHT - 0.5, vmax=MAX_MOVES_OVERNIGHT + 0.5, cmap=cmap
    )
    ax.set_xlabel("#Cars at second location")
    ax.set_ylabel("#Cars at first location")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical", ticks=range(-MAX_MOVES_OVERNIGHT, MAX_MOVES_OVERNIGHT + 1))

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_title("Policy values")
    x = np.arange(0, MAX_CARS + 1)
    y = np.arange(0, MAX_CARS + 1)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, values)
    ax.set_xlabel("#Cars at second location")
    ax.set_ylabel("#Cars at first location")
    ax.zaxis.set_major_formatter("{x:.0f}")

    plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=2)

    policy, values = policy_iteration_state_values()

    plot_policy_values(policy, values)
