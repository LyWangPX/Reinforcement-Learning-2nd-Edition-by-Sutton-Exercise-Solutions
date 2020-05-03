using Random, Plots, Distributions, Statistics, StatsBase

# Example solution for exercise 5.12 using Julia 1.4.1
# I only implemented the 1st race track, but it is trival to modify the
# race track data for the 2nd one.
#
# By @burmecia at GitHub, 29th April 2020
#
# Another useful solution in Python:
# https://towardsdatascience.com/solving-racetrack-in-reinforcement-learning-using-monte-carlo-control-bdee2aa4f04e

# number of race track rows and columns
rows, cols = 32, 17

ε = 0.1
γ = 1

# action can be horizontal and vertical with value 0, 1 or -1
# represented by 1 to 9
actions = [
    (0, 0),
    (0, 1),
    (0, -1),
    (1, 0),
    (1, 1),
    (1, -1),
    (-1, 0),
    (-1, 1),
    (-1, -1),
]
act_len = length(actions)

# 5 velocity values 0 to 4, represented by 1 to 5
vel_len = 5

# initialise Q, C and π
# state is 4-tuple: (row, col, velocity_horizontal, verlocity_vertical)
Q = rand(rows, cols, vel_len, vel_len, act_len) .* 400 .- 500
C = zeros(Float64, rows, cols, vel_len, vel_len, act_len)
π = ones(Int64, rows, cols, vel_len, vel_len)
for r = 1:rows, c = 1:cols, h = 1:vel_len, v = 1:vel_len
    π[r, c, h, v] = argmax(Q[r, c, h, v, :])
end

# set up the 1st race track map, origin (1,1) is at the bottom left,
# boundaries are marked with 1
track = zeros(Int8, rows, cols)
track[32, 1:3] .= 1
track[31, 1:2] .= 1
track[30, 1:2] .= 1
track[29, 1] = 1
track[1:18, 1] .= 1
track[1:10, 2] .= 1
track[1:3, 3] .= 1
track[1:26, 10:end] .= 1
track[26, 10] = 0
start_cols = 4:9  # start line columns
fin_cells = Set([
    (27, cols),
    (28, cols),
    (29, cols),
    (30, cols),
    (31, cols),
    (32, cols),
])  # finish cells

# valid actions for each velocity combination
# Both velocity components are restricted to be nonnegative and less than 5,
# and they cannot both be zero.
valid_acts = [
    map(
        x -> x[1],
        filter(
            a ->
                (h + a[2][1]) in 0:4 &&
                (v + a[2][2]) in 0:4 &&
                !((h + a[2][1]) == 0 && (v + a[2][2]) == 0),
            collect(enumerate(actions)),
        ),
    ) for h = 0:vel_len-1, v = 0:vel_len-1
]

# pre-allocated state, action, probility tracjectory array
S = Array{Tuple{Int64,Int64,Int64,Int64}}(undef, 10^6)
A = Array{Int64}(undef, 10^6)
B = Array{Float64}(undef, 10^6)

function make_trajectory(ε; noise=true)
    t = 1

    # start state
    S[t] = (1, rand(start_cols), 1, 1)

    while true
        s = S[t]

        # get all valid actions for current velocity
        acts = valid_acts[s[3], s[4]]
        num_acts = length(acts)

        # choose next action using ε-greedy policy if πa is valid
        # otherwise choose randomly, and save its probility
        πa = π[s...]
        πa_valid = πa in acts
        if rand() >= ε
            if πa_valid
                a = πa
                b = 1 - ε + ε / num_acts
            else
                a = sample(acts)
                b = 1 / num_acts
            end
        else
            a = sample(acts)
            b = (πa_valid ? ε : 1) / num_acts
        end

        # add some noise
        # with probability 0.1 at each time step the velocity increments are
        # both zero, independently of the intended increments
        if noise && rand() < 0.1
            a = 1
            b = 0.1
        end

        A[t] = a
        B[t] = b
        act = actions[a]

        # next state
        vel = (s[3] + act[1], s[4] + act[2])
        next_s = (s[1] + vel[2] - 1, s[2] + vel[1] - 1, vel[1], vel[2])

        # check if car hits finish line
        path = Set([
            (min(s[1] + i, next_s[1]), min(s[2] + i, next_s[2]))
            for i = 0:maximum(vel)
        ])
        if !isempty(intersect(path, fin_cells))
            return t
        end

        #@show s, a, act, b, next_s, path
        if !checkbounds(Bool, track, next_s[1], next_s[2]) ||
           track[next_s[1], next_s[2]] == 1
            # if car hits boundary, go back to start
            s = (1, rand(start_cols), 1, 1)
        else
            # otherwise go to the next state
            s = next_s
        end

        # append state to tracjectory
        t += 1
        S[t] = s
    end
end

function run_episode(T)
    G, W, R = 0.0, 1.0, -1

    for t = T:-1:1
        s = S[t]
        st = CartesianIndex(Tuple(collect(s)))
        sa = CartesianIndex(Tuple([Tuple(st)..., A[t]]))

        G = γ * G + R
        C[sa] += W
        Q[sa] += W * (G - Q[sa]) / C[sa]

        acts = valid_acts[s[3], s[4]]
        π[st] = acts[argmax(Q[st, :][acts])]
        if A[t] != π[st]
            return t
        end

        W /= B[t]
    end

    0
end

# print several trajectories following the optimal policy
function output_trajectories()
    for i = 1:3
        T = make_trajectory(0.0, noise=false)
        println("\noptimal trajectory #$(i):")
        println("S: ", S[1:T])
        println("A: ", A[1:T])
        println("R: ", -1 * T)
    end
end

function main()
    episode_num = 10^5
    rewards = []

    for i = 1:episode_num
        T = make_trajectory(ε)
        t = run_episode(T)
        println("episode $(i): $(T), $(t), $(T-t)")

        if i % 9 == 0
            T = make_trajectory(0.0)
            push!(rewards, -1 * T)
        end
    end

    plot(rewards)

    output_trajectories()
end

# uncomment the main function below start learning
# take a cup of coffee, it will run for several minutes

#main()
