using Random, Plots, Distributions, Statistics

# Example solution for exercise 8.4 using Julia 1.4.1
#
# In this exercise, I first reproduced example 8.2 with Dyna-Q and Dyna-Q+.
# Then I made the experiment based on this example. Let's call the experiment
# with name Dyna-Q+exp, The result shows Dyna-Q+exp is better than Dyna-Q and
# almost as good as Dyna-Q+ in the first 1000 steps. But after environment
# change, it can't catch up with Dyna-Q+.
#
# I think this is probably because Dyna-Q+ updates bonus directly in Q values,
# this makes it can quickly do exploration again after environment change. But
# Dyna-Q+exp only consider bonus when choose maximal Q, this will take longer
# time to do exploration than Dyna-Q+. In other words, the bonus is not
# 'accumulated' in Q value for Dyna-Q+exp.
#
# By @burmecia at GitHub, 12 May 2020
#
# Python reference code for example 8.2:
# https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter08/maze.py
#

α, ε, γ = 1.0, 0.1, 0.95

# n-step planning
n = 10

# maximum step numbers
max_steps = 3000

# grid world map, origin (1,1) is at the bottom left
rows, cols = 6, 9
start, goal = [1, 4], [6, 9]
barrier = [[3, c] for c = 1:8]  # barrier
new_barrier = [[3, c] for c = 2:9]  # new barrier

# full action list
actions = ("N", "E", "S", "W")
act_len = length(actions)

# global Q array: (state, action)
Q = Array{Float64}

# model: element is (state, reward, timestamp)
Model = Array{Tuple{Tuple{Int64,Int64},Float64,Int64}}

# choose action using ε-greedy policy
function choose_action(s, steps, k, experiment)
    if rand() >= ε
        a = Q[s..., :]
        if experiment
            # Q(S, a) + bonus
            a += [k * sqrt(steps - i[3]) for i in Model[s..., :]]
        end
        max_q = maximum(a)
        sample(filter(x -> x[2] == max_q, collect(enumerate(a))))[1]
    else
        rand(1:act_len)
    end
end

# take action, observe the next state and reward
function next_state(s, a)
    s′ = copy(s)
    if a == 1
        s′[1] += 1  # north
    elseif a == 2
        s′[2] += 1  # east
    elseif a == 3
        s′[1] -= 1  # south
    elseif a == 4
        s′[2] -= 1  # west
    end

    # if agent hits barrier, leave its position unchanged
    if s′ in barrier
        s′ = copy(s)
    end

    # if agent is off the grid, leave its position unchanged
    s′[1] = clamp(s′[1], 1, rows)
    s′[2] = clamp(s′[2], 1, cols)

    s′, s′ == goal ? 1.0 : 0.0
end

function game(; k = 0.0, experiment = false)
    global Q, Model, barrier

    # initialise Q, model and barrier
    Q = zeros(Float64, rows, cols, act_len)
    Model = [((0, 0), 0, 1) for r = 1:rows, c = 1:cols, a = 1:act_len]
    barrier = [[3, c] for c = 1:8]

    rewards = zeros(Float64, max_steps)
    steps, s = 1, start

    while steps < max_steps
        # take action, observe result
        a = choose_action(s, steps, k, experiment)
        s′, r = next_state(s, a)

        # upate Q[sa]
        sa = CartesianIndex(Tuple([s..., a]))
        Q[sa] += α * (r + γ * maximum(Q[s′..., :]) - Q[sa])

        # model learning
        # Note from footnote on page 168 from the book:
        # 1. actions that had never been tried before from a state were
        #    allowed to be considered
        # 2. the initial model for such actions was that they would lead back
        #    to the same state with a reward of zero
        if k > 0 && !experiment
            for i = 1:act_len
                if Model[s..., i][1][1] == 0
                    Model[s..., i] = (Tuple(s), 0, 1)
                end
            end
        end
        Model[sa] = (Tuple(s′), r, steps)

        # planning, sample experience from the model
        observed = filter(
            x -> Model[x...][1][1] > 0,
            collect(Iterators.product(1:rows, 1:cols, 1:act_len)),
        )
        for i in sample(observed, n)
            sa = CartesianIndex(i)
            s2′, r2, t = Model[sa]

            # adjust reward with elapsed time since last vist
            r2 += experiment ? 0.0 : k * sqrt(steps - t)

            Q[sa] += α * (r2 + γ * maximum(Q[s2′..., :]) - Q[sa])
        end

        s = s′

        # start a new episode if reached goal
        if s == goal
            s = start
        end

        # update acumulative rewards
        steps += 1
        rewards[steps] = rewards[steps-1] + r

        # set new barrier after 1000 steps
        if steps == 1000
            barrier = new_barrier
        end
    end

    rewards
end

function main()
    runs, k = 20, 1e-4
    rewards_dyna_q = zeros(Float64, runs, max_steps)
    rewards_dyna_qplus = zeros(Float64, runs, max_steps)
    rewards_dyna_qplus_exp = zeros(Float64, runs, max_steps)

    for i = 1:runs
        println("run $(i)")
        rewards_dyna_q[i, :] = game()
        rewards_dyna_qplus[i, :] = game(k = k)
        rewards_dyna_qplus_exp[i, :] = game(k = k, experiment = true)
    end

    rewards_dyna_q = mean(rewards_dyna_q, dims = 1)
    rewards_dyna_qplus = mean(rewards_dyna_qplus, dims = 1)
    rewards_dyna_qplus_exp = mean(rewards_dyna_qplus_exp, dims = 1)

    p = plot(
        xlabel = "Time steps",
        ylabel = "Cumulative rewards",
        legend = :topleft,
    )
    plot!(p, rewards_dyna_q[1, :], label = "Dyna-Q")
    plot!(p, rewards_dyna_qplus[1, :], label = "Dyna-Q+")
    plot!(p, rewards_dyna_qplus_exp[1, :], label = "Dyna-Q+ experiment")
end

# uncomment the main function below to start learning
#main()
