using Yao, LinearAlgebra, YaoBlocks

Ugate(nbit::Int, i::Int) = put(nbit, i=>chain(Rz(0), Rx(0), Rz(0)));
Ugate(i::Int) = put(i=>chain(Rz(0), Rx(0), Rz(0)));

function build_circuit()
    # circuit = chain(2, control(1, 2=>H), Ugate(1))
    circuit = chain(2, control(1, 2=>H), Ugate(1))
    return circuit
end

struct InitStateMatrix
    m1::Matrix{ComplexF64}
    m2::Matrix{ComplexF64}
end

function build_state(mats::InitStateMatrix)
    # return zero_state(2) |> chain(2, put(1=>chain(Rz(1), Rx(2))), put(2=>chain(Rz(1), Rx(2))))
    return zero_state(2) |> chain(2, put(2, (1)=>matblock(mats.m1)), put(2, (2)=>matblock(mats.m2)))
end

# @show probs(build_state())

function overlap(x::InitStateMatrix)
    return dot(apply!(build_state(x), chain(2, control(1, 2=>X), put(1=>H))) |> probs,[1, 1, 1, -1])
end

function cost(trainX, trainY, circuit, c)
    actual_out = [dot(i |> circuit |> probs, c) for i in map(build_state, trainX)]
    return sum((trainY-actual_out).^2)/length(trainY)
end

function gradient(circuit, trainX, trainY, delta)
    n = nqubits(circuit)
    grad = zeros(Float64, nparameters(circuit))
    params = parameters(circuit)

    count = 1
    for k in 1:nparameters(circuit)
        params[k] += delta
        dispatch!(circuit, params)
        cost_pos = cost(trainX, trainY, circuit, c)

        params[k] -= 2 * delta
        dispatch!(circuit, params)
        cost_neg = cost(trainX, trainY, circuit, c)

        params[k] += delta
        dispatch!(circuit, params)

        # @show (cost_pos - cost_neg)

        grad[count] = (cost_pos - cost_neg)/(2 * delta)
        count += 1
    end
    return grad
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
    last = circuit[1]
    for i in 2:length(circuit)
        now = circuit[i]
        if isa(last, ControlBlock) && last == now
            return false
        end
        if (!isa(last, ControlBlock)) && (!isa(now, ControlBlock)) && occupied_locs(last) == occupied_locs(now)
            return false
        end
        last = now
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
    return rtn
end

function random_modify(circuit, gate_set)
    circuit2 = deepcopy(circuit)
    index = rand(1:length(circuit2))
    # @show index
    circuit2[index] = rand(gate_set)
    if validate_circuit(circuit2)
        return circuit2
    end
    return random_modify(circuit, gate_set)
end


using Flux.Optimise

c = [1, 1, 1, -1];
# circuit = build_circuit()
trainX = [InitStateMatrix(rand_unitary(2), rand_unitary(2)) for _ in 1:8]
trainY = map(overlap, trainX)
opt = Descent(3.0)

function train(circuit)
    history = Float64[]
    last_cost = now_cost = 0
    for i in 1:200
        last_cost = now_cost
        now_cost = cost(trainX, trainY, circuit, c)
        push!(history, now_cost)
        ps = parameters(circuit)
        # @show (i, ps, now_cost)
        if abs(now_cost - last_cost) < 1e-8 || now_cost < 1e-7
            # @show now_cost
            break
        end
        Optimise.update!(opt, ps, gradient(circuit, trainX, trainY, .001))
        popdispatch!(circuit, ps)
    end
    return (now_cost, history)
end

function find_algo(annealing_param, d::Int64)
    gate_set = generate_gates_set(2)
    circuit = generate_init_circuit(2, d)
    (best_cost, history) = train(circuit)
    count = 0
    circuit_backup = circuit
    while best_cost > 1e-7 && count < 1000
        count += 1;
        circuit_backup = deepcopy(circuit)
        circuit = random_modify(circuit, gate_set)
        (new_cost, history) = train(circuit)

        accept_prob = exp(-annealing_param * (new_cost - best_cost))
        if new_cost < best_cost
            @show (count, new_cost)
            best_cost = new_cost
            continue
        elseif rand() > accept_prob
            @show "fallback"
            circuit = circuit_backup
        end
        @show (count, accept_prob)
    end
    return circuit
end

@show find_algo(10.0, 2)
# (now_cost, history) = train(circuit);

# using Plots
#
# function plot_history(history)
#     pyplot() # Choose a backend
#     fig1 = plot(history; legend=nothing)
#     title!("training history")
#     xlabel!("steps"); ylabel!("cost")
# end
