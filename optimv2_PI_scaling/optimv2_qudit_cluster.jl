
using LinearAlgebra
using Optim 
using LineSearches
using Combinatorics
using JLD2
using SpecialFunctions
using ADTypes: AutoForwardDiff, AutoReverseDiff
using ReverseDiff

include("base_KL.jl")

### ---------------------- Optimisation Code ---------------------- ### 

const n = parse(Int64, ARGS[1])
const q = parse(Int64, ARGS[2])
const d = parse(Int64, ARGS[3])
const iter = parse(Int64, ARGS[4])
const t = parse(Int64, ARGS[5])

const λ = partitions_into_q_parts(n,q);
const μ = partitions_into_q_parts(2t,q);
const ν = partitions_into_q_parts(2t,q);
const vλ = size(λ)[1];
const vμ = size(μ)[1];
const vν = size(ν)[1];
const binoms_diff_cached = binoms_diff_precompute(n, q, t, λ, μ, ν, vλ, vμ, vν);
const nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν);
const optim_soltol = 1e-15;
const pos_cached = partition_pos_precompute(n, q, d, t, λ, μ, ν, vλ, vμ, vν);

#Define cost function
cost(c0) = abs(rules1(n,q,d,c0)) + rule4_5(n,q,d,t,c0,λ,μ,ν,vλ,vμ,vν);

callback(state) = (abs(state.f_x) < optim_soltol ? (return true) : (return false) );

const codeword_length = size(partitions_into_q_parts(n,q))[1];
const num_var_params = d * codeword_length;

res_loop_minimum = []; res_loop_minimizer = []; thread_arr = [];



println("start optimization.")

x0 = normalize(rand(num_var_params)) # this is for real 
# x0 = normalize(rand(ComplexF64,num_var_params)) # for Vlad: this is complex 

res = optimize(cost, x0, LBFGS(linesearch=LineSearches.BackTracking()), autodiff = AutoReverseDiff(),
# res = optimize(cost, x0, LBFGS(linesearch=LineSearches.BackTracking()), 
            Optim.Options(iterations=200000,
                        g_tol=1e-20,
                        f_abstol=1e-20,
                        f_reltol=1e-20,
                        allow_f_increases=true,
                        show_trace=false,
                        callback=callback,
                        time_limit = 60 * 60 * 24) # put a time limit of 1 day (units of seconds))
                )

minval = res.minimum;
minx0 = res.minimizer;
stoppedby = res.stopped_by;
optimiters = res.iterations;

println("saving. minimum was $minval")

save("data/n$(n)_q$(q)_d$(d)_t$(t)/iter$(iter).jld2","minval",minval,"minx0",minx0,"n",n,"t",t,"d",d,"iter",iter,"q",q,"stoppedby",stoppedby,"optimiters",optimiters,"res",res)

println("done.")
    