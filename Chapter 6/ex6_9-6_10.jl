using Random, Plots, Distributions, Statistics

# Example solution for example 6.5, exercise 6.9 and 6.10 using Julia 1.4.1
#
# By @burmecia at GitHub, 2nd May 2020
#

α, ε, γ = 0.5, 0.1, 1

# grid world map, origin (1,1) is at the bottom left
rows, cols = 7, 10
start, goal = [4, 1], [4, 8]

# full action list
actions = ("N", "E", "S", "W", "NE", "SE", "SW", "NW", "Z")

# wind
wind = zeros(Int64, cols)
wind[4:9] .= 1
wind[7:8] .= 2
wind_offset = 0:0

# ============================================================
# uncomment individual block below to run different exercises
#
# example 6.5
act_len = 4

# exercise 6.9
#act_len = 8

# exercise 6.9a
#act_len = 9

# exercise 6.10
#act_len = 8
#wind_offset = -1:1
# ============================================================

# initialise Q
# state is a tuple: (row, col)
Q = zeros(Float64, rows, cols, act_len)

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
    elseif a == 5
        s′[1] += 1  # north east
        s′[2] += 1
    elseif a == 6
        s′[1] -= 1  # south east
        s′[2] += 1
    elseif a == 7
        s′[1] -= 1  # south west
        s′[2] -= 1
    elseif a == 8
        s′[1] += 1  # north west
        s′[2] -= 1
    elseif a == 9
        # do nothing
    end

    # row shift by wind
    w = wind[clamp(s[2], 1, length(wind))]
    s′[1] += w + (w > 0 ? rand(wind_offset) : 0)

    s′
end

# choose action using ε-greedy policy
function choose_action(s; ε = ε)
    if rand() >= ε
        argmax(Q[s[1], s[2], :])
    else
        rand(1:act_len)
    end
end

# take action, observe next state and get next action using ε-greedy policy
function next_state_action(s, a; ε = ε)
    # take action, get next state
    s′ = next_state(s, a)

    # if agent is off the grid, leave its position unchanged
    s′[1] = clamp(s′[1], 1, rows)
    s′[2] = clamp(s′[2], 1, cols)

    # choose A′ from S′
    a′ = choose_action(s′, ε = ε)

    (s′, a′)
end

function main()
    episode_num = 2 * 10^5
    r = -1

    for i = 1:episode_num
        if i % 1000 == 0 println("episode $(i)") end

        s = start
        a = choose_action(s)
        while s != goal
            s′, a′ = next_state_action(s, a)

            # upate Q[sa]
            sa = CartesianIndex(Tuple([s..., a]))
            s′a′ = CartesianIndex(Tuple([s′..., a′]))
            Q[sa] += α * (r + γ * Q[s′a′] - Q[sa])

            s, a = s′, a′
        end
    end

    println("finished.\n")
    println("an optimal tracjectory:\n")

    s = start
    a = choose_action(s, ε = 0)
    while s != goal
        println("S: $(s), A: $(actions[a])")
        s, a = next_state_action(s, a, ε = 0)
    end
end

# uncomment the main function below to start learning

#main()
