using Yao, LinearAlgebra, YaoBlocks
import Random.shuffle

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

    if length(circuit) > 3 && length(param_indexes(circuit)) == 0
        return false
    end
    if occupied_locs(circuit[length(circuit)]) != (1, )
        return rand() < .2
    end

    return true
end

function param_indexes(circuit, indexes)
    rtn = []
    for i in indexes
        if nparameters(circuit[i]) > 0
            push!(rtn, i)
        end
    end
    return rtn
end

function param_indexes(circuit)
    return param_indexes(circuit, [ 1:length(circuit); ])
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

function random_modify!(circuit2, circuit_indexes, gate_set)
    for _ in 1:rand(1:5)
        index = rand(circuit_indexes)
        # @show index
        circuit2[index] = rand(gate_set)

        while !validate_circuit(circuit2)
            circuit2[index] = rand(gate_set)
        end
        params = rand(nparameters(circuit2[index])) * 2π .- π
        dispatch!(circuit2[index], params)
    end
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
trainX = [InitStateMatrix(rand_unitary(2), rand_unitary(2)) for _ in 1:5]
testX = [InitStateMatrix(rand_unitary(2), rand_unitary(2)) for _ in 1:5]
trainY = map(overlap, trainX)
testY = map(overlap, testX)
opt = Descent(1.2)
d = 8
# opt = ADAM(0.03, (0.9, 0.999))

function train(circuit, circuit_indexes, trainX, trainY, testX, testY, cost_threshold)
    history = Float64[]
    last_cost = new_cost = 0
    for i in 1:60
        last_cost = new_cost
        new_cost = cost(trainX, trainY, circuit, c)
        # push!(history, new_cost)
        ps = parameters(circuit)
        # @show (i, ps, new_cost)
        if (abs(new_cost - last_cost) < new_cost * .005) || new_cost < cost_threshold
            # @show new_cost
            break
        end

        # Optimise only one gate each step
        # This prevents local-minimal
        for index in shuffle(circuit_indexes)
            Optimise.update!(opt, ps, gradient(circuit, [ index ], trainX, trainY, .001))
        end

        popdispatch!(circuit, ps)
        if i == 60
            printstyled(IOContext(stdout, :color => true), "Warning: failed to converge\n", color=:red)
        end
    end
    return (new_cost, history)
end

function compress_subsequence(annealing_param, circuit, trainX, cost_threshold)
    rtn = deepcopy(circuit)

    # generate random circuit_indexes
    index1 = rand([ 1:length(circuit) - 1; ])
    index2 = rand([ index1 + 1:min(index1 + 4, length(circuit)); ]) # try to compress 2-5 gates
    circuit_indexes = [ index1+1:index2; ]
    max_steps = 10 * (index2 - index1)

    gate_set = generate_gates_set(nqubits(circuit))

    rtn[index1] = Ugate(nqubits(rtn), rand([ 1:nqubits(rtn); ]))
    while !validate_circuit(rtn)
        rtn[index1] = Ugate(nqubits(rtn), rand([ 1:nqubits(rtn); ]))
    end

    newTrainY = [dot(i |> build_state |> circuit |> probs, c) for i in trainX]
    newTestY = [dot(i |> build_state |> circuit |> probs, c) for i in testX]
    train(rtn, param_indexes(rtn, circuit_indexes), trainX, newTrainY, testX, newTestY, cost_threshold)
    best_cost = cost(testX, newTestY, rtn, c)
    step = 0

    while best_cost > cost_threshold && step < max_steps
        step += 1;

        circuit_backup = deepcopy(rtn)
        random_modify!(rtn, circuit_indexes, gate_set)
        train(rtn, param_indexes(rtn, circuit_indexes), trainX, newTrainY, testX, newTestY, cost_threshold)
        new_cost = cost(testX, newTestY, rtn, c)

        accept_prob = exp(-annealing_param * (new_cost - best_cost))
        if new_cost < best_cost
            # @show (step, new_cost)
            best_cost = new_cost
            continue
        elseif rand() < accept_prob
            # accept
            # best_cost = new_cost
            # println("accepted")
        else
            # println("fallback")
            rtn = circuit_backup
        end
        # @show (step, best_cost, accept_prob)
    end

    sucs = best_cost < cost_threshold
    return (rtn, sucs)
end

function find_algo(annealing_param, d::Int64, trainX, trainY, max_steps::Int64, cost_threshold)
    gate_set = generate_gates_set(3)
    circuit = generate_init_circuit(3, d)
    train(circuit, param_indexes(circuit), trainX, trainY, testX, testY, cost_threshold)
    best_cost = cost(testX, testY, circuit, c)

    step = 0
    circuit_backup = circuit
    while best_cost > cost_threshold && step < max_steps
        step += 1;

        circuit_backup = deepcopy(circuit)
        random_modify!(circuit, [ 1:length(circuit); ], gate_set)

        train(circuit, param_indexes(circuit), trainX, trainY, testX, testY, cost_threshold)
        new_cost = cost(testX, testY, circuit, c)

        accept_prob = exp(-annealing_param * (new_cost - best_cost))
        if new_cost < best_cost
            @show (step, new_cost)
            best_cost = new_cost
            continue
        elseif rand() < accept_prob
            # accept
            # best_cost = new_cost
            println("accepted")
        else
            # println("fallback")
            circuit = circuit_backup
        end


        if mod(step, 5) == 0
            printstyled(IOContext(stdout, :color => true), "tring to compress...\n", color=:blue)

            (new_circuit, sucs) = compress_subsequence(annealing_param, circuit, trainX, cost_threshold)

            if sucs
                @show circuit
                printstyled(IOContext(stdout, :color => true), "compress succeeded!\n", color=:green)

                circuit = new_circuit
                best_cost = cost(testX, testY, circuit, c)

            else
                printstyled(IOContext(stdout, :color => true), "compress failed!\n", color=:cyan)
            end
            @show (step, d, best_cost)
            if mod(step, 100) == 0
                @show circuit
            end
        end
    end
    return (circuit, best_cost < cost_threshold)
end

function find_nearest_pi(x::Float64)
    rtn = mod(x, 2*π);
    rtn = rtn > π ? rtn - 2*π : rtn;
    rtn = π / 4 * round(4 * rtn / π)
end

function enhance_circuit_to_pi(circuit)
    params = parameters(circuit)
    params = map(find_nearest_pi, params)
    dispatch!(circuit, params)
end

# (tmp, history) = train(generate_init_circuit(3, 9), trainX, trainY, cost_threshold)
# using Plots
# function plot_history(history)
#     pyplot() # Choose a backend
#     fig1 = plot(history; legend=nothing)
#     title!("training history")
#     xlabel!("steps"); ylabel!("cost")
# end
# plot_history(history)

# annealing_param 必须是100000数量级
@show (circuit, ) = find_algo(70000.0, d, trainX, trainY, 500000, cost_threshold)
enhance_circuit_to_pi(circuit)
@show cost(trainX, trainY, circuit, c)
