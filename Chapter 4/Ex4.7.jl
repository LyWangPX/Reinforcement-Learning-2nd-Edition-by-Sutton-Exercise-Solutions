using Random, Plots, Distributions, Statistics

n = 20
k = n + 1
b = 10  # Poisson upper bound
γ = 0.9
θ = 0.1
V = zeros(Float64, k, k)
PI = zeros(Int64, k, k)

# day rental probility at location A and B for one day
A_rp =
    [(j - i, i, pdf(Poisson(3), i) * pdf(Poisson(4), j)) for i = 0:b, j = 0:b]
B_rp =
    [(j - i, i, pdf(Poisson(3), i) * pdf(Poisson(2), j)) for i = 0:b, j = 0:b]

# day rental at the end of day with a given number of cars
# return element: (car number delta, number of cars rent out, probility)
function day_rental(rp, start)
    broadcast((x, y) -> (clamp(y + x[1], 0, n), min(y, x[2]), x[3]), rp, start)
end

# value evaluation
function value_eval(A, B, orig_act)
    global k, γ, V
    v = 0.0

    for a in A, b in B
        # calculate actual action
        act = orig_act >= 0 ? min(a[1], orig_act, n - b[1]) :
            -min(b[1], abs(orig_act), n - a[1])

        # next state
        i2 = a[1] - act + 1
        j2 = b[1] + act + 1

        # extra parking cost for car number > 10
        extra = 0.0
        if a[1] - act > 10
            extra += 4
        end
        if b[1] + act > 10
            extra += 4
        end

        # free bus transport 1 car from A to B
        if act > 1
            act -= 1
        end

        v +=
            a[3] *
            b[3] *
            ((a[2] + b[2]) * 10 - abs(act) * 2 - extra + γ * V[i2, j2])
    end

    v
end

function policy_evaluation()
    global n, k, γ, θ, V, PI, A_rp, B_rp

    while true
        delta = 0.0
        for i = 1:k, j = 1:k
            A = day_rental(A_rp, i - 1)
            B = day_rental(B_rp, j - 1)

            v = value_eval(A, B, PI[i, j])

            delta = max(delta, abs(v - V[i, j]))
            V[i, j] = v
        end
        @show delta
        if delta < θ
            break
        end
    end
end

function policy_improvement()
    global n, k, γ, V, PI, A_rp, B_rp
    stable = true

    for i = 1:k, j = 1:k
        A = day_rental(A_rp, i - 1)
        B = day_rental(B_rp, j - 1)
        max_v, max_act = 0.0, 0

        for act = -5:5
            v = value_eval(A, B, act)

            if v >= max_v
                max_act = act
                max_v = v
            end
        end

        if PI[i, j] != max_act
            stable = false
        end
        PI[i, j] = max_act
    end

    stable
end

# policy iteration
policy_evaluation()
while !policy_improvement()
    policy_evaluation()
end

# draw plot for V and π
heatmap(V)
heatmap(PI)
