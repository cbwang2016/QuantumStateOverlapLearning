# todo
# using Yao, LinearAlgebra, YaoBlocks
#
# Ugate(nbit::Int, i::Int) = put(nbit, i=>chain(Rz(0), Rx(0), Rz(0)));
# Ugate(i::Int) = put(i=>chain(Rz(0), Rx(0), Rz(0)));
#
# struct InitStateMatrix
#     m1::Matrix{ComplexF64}
#     m2::Matrix{ComplexF64}
# end
#
# function build_state(mats::InitStateMatrix)
#     # return zero_state(2) |> chain(2, put(1=>chain(Rz(1), Rx(2))), put(2=>chain(Rz(1), Rx(2))))
#     return zero_state(3) |> chain(3, put(3, (2)=>matblock(mats.m1)), put(3, (3)=>matblock(mats.m2)))
# end
#
# # @show probs(build_state())
#
# function overlap(mats::InitStateMatrix)
#     return dot(zero_state(2) |> chain(2, put(2, (1)=>matblock(mats.m1)), put(2, (2)=>matblock(mats.m2)), control(1, 2=>X), put(1=>H)) |> probs,[1, 1, 1, -1])
# end
#
# function cost(trainX, trainY, circuit, c)
#     actual_out = [dot(i |> circuit |> probs, c) for i in map(build_state, trainX)]
#     return sum((trainY-actual_out).^2)/length(trainY)
# end
#
# function gradient(circuit, trainX, trainY, delta)
#     n = nqubits(circuit)
#     grad = zeros(Float64, nparameters(circuit))
#     params = parameters(circuit)
#
#     count = 1
#     for k in 1:nparameters(circuit)
#         params[k] += delta
#         dispatch!(circuit, params)
#         cost_pos = cost(trainX, trainY, circuit, c)
#
#         params[k] -= 2 * delta
#         dispatch!(circuit, params)
#         cost_neg = cost(trainX, trainY, circuit, c)
#
#         params[k] += delta
#         dispatch!(circuit, params)
#
#         # @show (cost_pos - cost_neg)
#
#         grad[count] = (cost_pos - cost_neg)/(2 * delta)
#         count += 1
#     end
#     return grad
# end
#
# function generate_gates_set(n::Int64)
#     rtn = []
#     push!(rtn, control(n, 2, 1=>X))
#     push!(rtn, control(n, 3, 1=>X))
#     push!(rtn, control(n, 3, 2=>X))
#
#     for i in 1:n
#         push!(rtn, Ugate(n, i))
#     end
#
#     return rtn
# end
#
# # 检测生成的circuit是否合法
# function validate_circuit(circuit)
#     last = circuit[1]
#     for i in 2:length(circuit)
#         now = circuit[i]
#         if isa(last, ControlBlock) && last == now
#             return false
#         end
#         if (!isa(last, ControlBlock)) && (!isa(now, ControlBlock)) && occupied_locs(last) == occupied_locs(now)
#             return false
#         end
#         last = now
#     end
#
#     return true
# end
#
# function generate_init_circuit(n::Int64, d::Int64)
#     gate_set = generate_gates_set(n)
#     rtn = chain(n)
#     while length(rtn) < d
#         push!(rtn, rand(gate_set))
#         if !validate_circuit(rtn)
#             pop!(rtn)
#         end
#     end
#     return rtn
# end
#
# function random_modify(circuit, gate_set)
#     circuit2 = deepcopy(circuit)
#     index = rand(1:length(circuit2))
#     # @show index
#     circuit2[index] = rand(gate_set)
#     if validate_circuit(circuit2)
#         return circuit2
#     end
#     return random_modify(circuit, gate_set)
# end
#
#
#
#
#
# using Flux.Optimise
#
# c = [1, -1, 1, -1, 1, -1, 1, -1];
# const cost_threshold = 2e-7
# # circuit = build_circuit()
# trainX = [InitStateMatrix(rand_unitary(2), rand_unitary(2)) for _ in 1:8]
# trainY = map(overlap, trainX)
# opt = Descent(0.8)
# # opt = ADAM()
#
# function train(circuit, trainX, trainY, cost_threshold)
#     history = Float64[]
#     last_cost = now_cost = 0
#     for i in 1:100
#         last_cost = now_cost
#         now_cost = cost(trainX, trainY, circuit, c)
#         push!(history, now_cost)
#         ps = parameters(circuit)
#         # @show (i, ps, now_cost)
#         if (abs(now_cost - last_cost) < cost_threshold / 3) || now_cost < cost_threshold
#             # @show now_cost
#             break
#         end
#         Optimise.update!(opt, ps, gradient(circuit, trainX, trainY, .001))
#         popdispatch!(circuit, ps)
#         # if i == 200
#         #     println("Warning: failed to converge")
#         # end
#     end
#     return (now_cost, history)
# end
#
# function find_algo(annealing_param, d::Int64, trainX, trainY, max_steps::Int64, cost_threshold)
#     gate_set = generate_gates_set(3)
#     circuit = generate_init_circuit(3, d)
#     (best_cost, history) = train(circuit, trainX, trainY, cost_threshold)
#     step = 0
#     circuit_backup = circuit
#     while best_cost > cost_threshold && step < max_steps
#         step += 1;
#         # if mod(step, 60) == 59 && d > 8
#         #
#         #     println("尝试压缩circuit")
#         #     newTrainY = [dot(i |> circuit |> probs, c) for i in map(build_state, trainX)]
#         #     (new_circuit, sucs) = find_algo(annealing_param, d - 1, trainX, newTrainY, 100, best_cost / 2)
#         #     if sucs
#         #         println("成功压缩circuit！！")
#         #         circuit = new_circuit
#         #         best_cost = cost(trainX, trainY, circuit, c)
#         #         while length(circuit) < d
#         #             push!(circuit, rand(gate_set))
#         #             if !validate_circuit(circuit)
#         #                 pop!(circuit)
#         #             end
#         #         end
#         #         continue
#         #     else
#         #         println("压缩circuit失败")
#         #     end
#         # end
#
#         circuit_backup = deepcopy(circuit)
#         circuit = random_modify(circuit, gate_set)
#         (new_cost, history) = train(circuit, trainX, trainY, cost_threshold)
#
#         accept_prob = exp(-annealing_param * (new_cost - best_cost))
#         if new_cost < best_cost
#             @show (step, new_cost)
#             best_cost = new_cost
#             continue
#         elseif rand() < accept_prob
#             # accept
#             # best_cost = new_cost
#             println("accepted")
#         else
#             # println("fallback")
#             circuit = circuit_backup
#         end
#         @show (step, d, best_cost, accept_prob)
#     end
#     return (circuit, best_cost < cost_threshold)
# end
#
# function find_nearest_pi(x::Float64)
#     rtn = mod(x, 2*π);
#     rtn = rtn > π ? rtn - 2*π : rtn;
#     rtn = π / 4 * round(4 * rtn / π)
# end
#
# function enhance_circuit_to_pi(circuit)
#     params = parameters(circuit)
#     params = map(find_nearest_pi, params)
#     dispatch!(circuit, params)
# end
#
# # (tmp, history) = train(generate_init_circuit(3, 9), trainX, trainY, cost_threshold)
# # using Plots
# # function plot_history(history)
# #     pyplot() # Choose a backend
# #     fig1 = plot(history; legend=nothing)
# #     title!("training history")
# #     xlabel!("steps"); ylabel!("cost")
# # end
# # plot_history(history)
#
# @show (circuit, ) = find_algo(5000.0, 9, trainX, trainY, 600000, cost_threshold)
# enhance_circuit_to_pi(circuit)
# @show cost(trainX, trainY, circuit, c)
