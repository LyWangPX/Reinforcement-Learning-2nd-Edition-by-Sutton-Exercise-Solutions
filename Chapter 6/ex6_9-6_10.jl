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

# global Q array
Q = Array{Float64}

episode_num = 300

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
function choose_action(s, act_len; ε = ε)
    if rand() >= ε
        argmax(Q[s[1], s[2], :])
    else
        rand(1:act_len)
    end
end

# take action, observe next state and get next action using ε-greedy policy
function next_state_action(s, a, act_len; ε = ε)
    # take action, get next state
    s′ = next_state(s, a)

    # if agent is off the grid, leave its position unchanged
    s′[1] = clamp(s′[1], 1, rows)
    s′[2] = clamp(s′[2], 1, cols)

    # choose A′ from S′
    a′ = choose_action(s′, act_len, ε = ε)

    (s′, a′)
end

# output an optimal tracjectory using greedy policy
function output_optimal_tractory()
    s, a = start, choose_action(start, act_len, ε = 0)
    while s != goal
        println("S: $(s), A: $(actions[a])")
        s, a = next_state_action(s, a, act_len, ε = 0)
    end
end

function game(act_len)
    global Q

    # initialise Q
    Q = zeros(Float64, rows, cols, act_len)

    hist = zeros(Int64, episode_num)
    r = -1

    for i = 1:episode_num
        step = 0
        s, a = start, choose_action(start, act_len)
        while s != goal
            s′, a′ = next_state_action(s, a, act_len)

            # upate Q[sa]
            sa = CartesianIndex(Tuple([s..., a]))
            s′a′ = CartesianIndex(Tuple([s′..., a′]))
            Q[sa] += α * (r + γ * Q[s′a′] - Q[sa])

            s, a = s′, a′
            step += 1
        end

        # cap extream values in the first several episodes
        hist[i] = min(step, 300)
    end

    reshape(hist, 1, :)
end

function main()
    global wind_offset

    hist_4 = Array{Float64,2}(undef, 0, episode_num)
    hist_8 = Array{Float64,2}(undef, 0, episode_num)
    hist_9 = Array{Float64,2}(undef, 0, episode_num)

    for i = 1:200
        hist_4 = vcat(hist_4, game(4))
        hist_8 = vcat(hist_8, game(8))
        hist_9 = vcat(hist_9, game(9))
    end

    # exercise 6.10
    wind_offset = -1:1
    hist_8_rand_wind = Array{Float64,2}(undef, 0, episode_num)
    for i = 1:200
        hist_8_rand_wind = vcat(hist_8_rand_wind, game(8))
    end

    hist_4 = mean(hist_4, dims = 1)
    hist_8 = mean(hist_8, dims = 1)
    hist_9 = mean(hist_9, dims = 1)
    hist_8_rand_wind = mean(hist_8_rand_wind, dims = 1)

    p = plot(
        title = "4 vs 8 vs 9 action space with random wind",
        xlabel = "Episodes (averaged over 200 experiments)",
        ylabel = "Steps",
        legend = :topright,
    )
    plot!(p, hist_4[1, :], label = "actions=4")
    plot!(p, hist_8[1, :], label = "actions=8")
    plot!(p, hist_9[1, :], label = "actions=9")
    plot!(p, hist_8_rand_wind[1, :], label = "actions=8, random wind")
end

# uncomment the main function below to start learning
#main()
