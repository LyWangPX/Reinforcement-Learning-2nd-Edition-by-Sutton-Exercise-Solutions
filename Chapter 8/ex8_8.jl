using Random, Plots, Distributions, Statistics

# Example solution for exercise 8.8 using Julia 1.4.1
#
# My solution has similar shape with the book, but different start state value
# under the greedy policy. I am not sure where goes wrong, probably in the
# reward calculation? But my results are similar to all the other people's
# results which I found online (see below reference implementation). So just
# take my solution as one of the references, don't treat it absolutely correct.
#
# By @burmecia at GitHub, 15 May 2020
#
# Other reference implementation:
#
# 1. https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter08/trajectory_sampling.py
# 2. https://github.com/JuliaReinforcementLearning/ReinforcementLearningAnIntroduction.jl/blob/b5c718a891a4b3db4fae177b8b33ca506df1ecea/notebooks/Chapter08_Trajectory_Sampling.ipynb
# 3. https://github.com/enakai00/rl_book_solutions/blob/master/Chapter08/Exercise_8_8_Solution.ipynb

stat_len = 10000
T = 200000

ε = 0.1
act_len = 2
term_prob = 0.1
tick_num = 100

# make transition matrix, (state, action) -> (next_state, reward)
# element is b tuples: [(next_state, reward),...]
function make_transition_matrix(b)
    trans = [
        map(s -> (s, randn()), rand(1:stat_len, b))
        for s = 1:stat_len, a = 1:act_len
    ]
    # terminal state, transit to itself with zero reward
    trans[stat_len, 1] = [(stat_len, 0) for _ = 1:b]
    trans[stat_len, 2] = [(stat_len, 0) for _ = 1:b]
    trans
end

# take action, observe the next state and reward
function next_state(s, a, trans)
    if rand() < term_prob
        return stat_len, 0
    end
    sample(trans[s, a])
end

# evaluate start state value under greedy policy
function start_state_value(Q, trans)
    n = 200
    returns = zeros(Float64, n)

    for i = 1:n
        s = 1
        while s != stat_len
            a = argmax(Q[s, :])
            s, r = next_state(s, a, trans)
            returns[i] += r
        end
    end

    mean(returns)
end

function run_uniform(b, trans)
    Q = zeros(Float64, stat_len, act_len)
    interval = T ÷ tick_num
    vπ_s0 = zeros(Float64, tick_num + 1)

    for t = 0:T
        s = t % stat_len + 1
        a = (t % (stat_len * act_len)) ÷ stat_len + 1
        next_states = trans[s, a]

        Q[s, a] =
            (1 - term_prob) *
            mean(map(ns -> (ns[2] + maximum(Q[ns[1], :])), next_states))

        if t % interval == 0
            vπ_s0[t ÷ interval + 1] = start_state_value(Q, trans)
        end
    end

    vπ_s0
end

function run_on_policy(b, trans)
    Q = zeros(Float64, stat_len, act_len)
    interval = T ÷ tick_num
    vπ_s0 = zeros(Float64, tick_num + 1)
    s = 1

    for t = 0:T
        if rand() < ε
            a = rand(1:act_len)
        else
            a = argmax(Q[s, :])
        end
        next_states = trans[s, a]

        Q[s, a] =
            (1 - term_prob) *
            mean(map(ns -> (ns[2] + maximum(Q[ns[1], :])), next_states))

        s, r = next_state(s, a, trans)
        if s == stat_len
            s = 1
        end

        if t % interval == 0
            vπ_s0[t ÷ interval + 1] = start_state_value(Q, trans)
        end
    end

    vπ_s0
end

function experiment(b_list, p, subplot)
    task_cnt = 100
    interval = T ÷ tick_num
    x = 0:interval:T

    for b in b_list
        uniform = zeros(Float64, task_cnt, length(x))
        on_policy = zeros(Float64, task_cnt, length(x))

        println("start subplot $(subplot), b=$(b)")
        for i = 1:task_cnt
            if i % 10 == 0
                println("  task $(i)")
            end
            trans = make_transition_matrix(b)
            uniform[i, :] = run_uniform(b, trans)
            on_policy[i, :] = run_on_policy(b, trans)
        end

        uniform = mean(uniform, dims = 1)
        on_policy = mean(on_policy, dims = 1)

        plot!(
            p,
            x,
            uniform[1, :],
            subplot = subplot,
            title = "$(stat_len) states",
            label = "b=$(b), uniform",
        )
        plot!(
            p,
            x,
            on_policy[1, :],
            subplot = subplot,
            label = "b=$(b), on-policy",
        )
    end
end

function main()
    global stat_len, T

    p = plot(
        xlabel = "Computation time, in expected updates",
        ylabel = "Start state value",
        legend = :best,
        size = (600, 800),
        layout = (2, 1),
    )

    stat_len, T = 1000, 20000
    experiment([1, 3, 10], p, 1)
    stat_len, T = 10000, 200000
    experiment([1, 3], p, 2)

    display(p)
end

# uncomment the main function below to start learning
#main()
