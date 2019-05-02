using Yao, LinearAlgebra, YaoBlocks

Ugate(nbit::Int, i::Int) = put(nbit, i=>chain(Rz(0), Rx(0), Rz(0)));
Ugate(i::Int) = put(i=>chain(Rz(0), Rx(0), Rz(0)));

struct InitStateMatrix
    m1::Matrix{ComplexF64}
    m2::Matrix{ComplexF64}
end

function build_state(mats::InitStateMatrix)
    # return zero_state(2) |> chain(2, put(1=>chain(Rz(1), Rx(2))), put(2=>chain(Rz(1), Rx(2))))
    return zero_state(3) |> chain(3, put(3, (2)=>matblock(mats.m1)), put(3, (3)=>matblock(mats.m2)))
end

# @show probs(build_state())

function overlap(mats::InitStateMatrix)
    return dot(zero_state(2) |> chain(2, put(2, (1)=>matblock(mats.m1)), put(2, (2)=>matblock(mats.m2)), control(1, 2=>X), put(1=>H)) |> probs,[1, 1, 1, -1])
end

function cost(trainX, trainY, circuit, c)
    actual_out = [dot(i |> build_state |> circuit |> probs, c) for i in trainX]
    return sum((trainY-actual_out).^2)/length(trainY)
end

function generate_gates_set(n::Int64)
    rtn = []
    for i in 1:n
        for j in 1:n
            if i != j
                push!(rtn, control(n, i, j=>X))
            end
        end
    end

    for i in 1:n
        push!(rtn, Ugate(n, i))
    end

    return rtn
end

# 检测生成的circuit是否合法
function validate_circuit(circuit)
    old = circuit[1]
    for i in 2:length(circuit)
        now = circuit[i]
        if isa(old, ControlBlock) && old == now
            return false
        end
        if (!isa(old, ControlBlock)) && (!isa(now, ControlBlock)) && occupied_locs(old) == occupied_locs(now)
            return false
        end
        old = now
    end

    return true
end

function generate_init_circuit(n::Int64, d::Int64)
    gate_set = generate_gates_set(n)
    rtn = chain(n)
    while length(rtn) < d
        push!(rtn, rand(gate_set))
        if !validate_circuit(rtn)
            pop!(rtn)
        end
    end
    params = rand(nparameters(rtn)) * 2π .- π
    dispatch!(rtn, params)

    return rtn
end

function random_modify(circuit, circuit_indexes, gate_set)
    circuit2 = deepcopy(circuit)
    index = rand(circuit_indexes)
    # @show index
    circuit2[index] = rand(gate_set)
    if validate_circuit(circuit2)
        params = rand(nparameters(circuit2[index])) * 2π .- π
        dispatch!(circuit2[index], params)
        return circuit2
    end
    return random_modify(circuit, circuit_indexes, gate_set)
end

function gradient(circuit, circuit_indexes, trainX, trainY, delta)
    n = nqubits(circuit)
    grad = zeros(Float64, nparameters(circuit))

    count = 1
    for i in 1:length(circuit)
        params = parameters(circuit[i])
        for k in 1:nparameters(circuit[i])
            if i in circuit_indexes
                params[k] += delta
                dispatch!(circuit[i], params)
                cost_pos = cost(trainX, trainY, circuit, c)

                params[k] -= 2 * delta
                dispatch!(circuit[i], params)
                cost_neg = cost(trainX, trainY, circuit, c)

                params[k] += delta
                dispatch!(circuit[i], params)

                # @show (cost_pos - cost_neg)

                grad[count] = (cost_pos - cost_neg)/(2 * delta)
            end
            count += 1
        end
    end
    return grad
end




using Flux.Optimise

c = [1, -1, 1, -1, 1, -1, 1, -1];
const cost_threshold = 2e-7
# circuit = build_circuit()
trainX = [InitStateMatrix(rand_unitary(2), rand_unitary(2)) for _ in 1:10]
trainY = map(overlap, trainX)
opt = Descent(.9)
# opt = ADAM(0.03, (0.9, 0.999))

function train(circuit, circuit_indexes, trainX, trainY, cost_threshold)
    history = Float64[]
    last_cost = now_cost = 0
    for i in 1:100
        last_cost = now_cost
        now_cost = cost(trainX, trainY, circuit, c)
        push!(history, now_cost)
        ps = parameters(circuit)
        # @show (i, ps, now_cost)
        if (abs(now_cost - last_cost) < cost_threshold / 3) || now_cost < cost_threshold
            # @show now_cost
            break
        end
        Optimise.update!(opt, ps, gradient(circuit, circuit_indexes, trainX, trainY, .001))
        popdispatch!(circuit, ps)
        # if i == 200
        #     println("Warning: failed to converge")
        # end
    end
    return (now_cost, history)
end

(tmp, history) = train(generate_init_circuit(3, 9), [ 1:length(circuit); ], trainX, trainY, cost_threshold)
using Plots
function plot_history(history)
    pyplot() # Choose a backend
    fig1 = plot(history; legend=nothing)
    title!("training history")
    xlabel!("steps"); ylabel!("cost")
end
plot_history(history)
