using LinearAlgebra
using Optim 
using LineSearches
using Combinatorics
using JLD2
using SpecialFunctions
using ADTypes: AutoForwardDiff, AutoReverseDiff
using ReverseDiff
using ProgressMeter
using PyPlot 

include("base_KL.jl")

const n = 37;
const q = 2;
const d = 2;
const iter = parse(Int64, ARGS[1])
const t = 3;

const λ = partitions_into_q_parts(n,q);
const μ = partitions_into_q_parts(2t,q);
const ν = partitions_into_q_parts(2t,q);
const vλ = size(λ)[1];
const vμ = size(μ)[1];
const vν = size(ν)[1];
const binoms_diff_cached = binoms_diff_precompute(n, q, t, λ, μ, ν, vλ, vμ, vν);
const nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν);
const pos_cached = partition_pos_precompute(n, q, d, t, λ, μ, ν, vλ, vμ, vν);

callback(state) = (abs(state.f_x) < optim_soltol ? (return true) : (return false) );

function renormalize_fixed(x, fixed, values)
    norm(values) > 1 ? error("Norm of fixed values must be less than 1") : nothing 
    y = copy(x)
    y[fixed] .= values
    unfixed = setdiff(1:length(x), fixed)
    y[unfixed] .*= sqrt((1 - sum(values.^2)) / sum(y[unfixed].^2))
    return y
end

function unique_length(x_in,tol) 
    x = deepcopy(x_in)
    xlength = length(x) 
    distancevec = []; 
    for i in eachindex(x) 
        j = i+1 
        while j ≤ xlength 
            distance = norm(x[i]-x[j])
            push!(distancevec,distance)
            if distance < tol 
                popat!(x,j)
                xlength = length(x) 
            else
                j += 1 
            end 
        end 
    end 
    return length(x), x, distancevec
end;


function sol_point_n(xvec, ivec, reps, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
     
    codeword_length = size(partitions_into_q_parts(n,q))[1]
    num_var_params = Int(d * codeword_length/2)

    function cost(x) 

        np = length(x) 

        for i in eachindex(xvec)
            x[xvec[i]] = ivec[i]
        end

        c0 = [x;x]

        for i in 1:np
            c0[np+i] = (-1)^(i-1) * c0[n-i+2]
        end

        rules1(n,q,d,c0) + rule4_5(n,q,d,t,c0,λ,μ,ν,vλ,vμ,vν)

    end 

    res_loop_minimum = zeros(reps); res_loop_minimizer = zeros(num_var_params,reps)

    @showprogress Threads.@threads for k in 1:reps

        cs = rand(num_var_params)

        res = Optim.optimize(cost, cs, LBFGS(linesearch=LineSearches.BackTracking()),
                    Optim.Options(iterations=15000,
                                g_tol=gtol,
                                f_abstol=ftol,
                                f_reltol=ftol,
                                allow_f_increases=true,
                                show_trace=false,
                                callback=callback)
                        )

        res_loop_minimum[k] = res.minimum 
        res_loop_minimizer[:,k] = res.minimizer

    end

    return res_loop_minimum, res_loop_minimizer

end

reps = 1000; 
ε = 1e-24;
ftol = 1e-24; 
gtol = 1e-24; 

const optim_soltol = 1e-24;

xvec = [1;2;3;4]; 
ivec = [0;0;0;0];
sol_point_vec = sol_point_n(xvec, ivec, reps, n, q, d, t, λ, μ, ν, vλ, vμ, vν); 


ϵ_success = 1e-18;
success_indices = (1:reps)[sol_point_vec[1] .≤ ϵ_success];
sol_point_vec_success = [ sol_point_vec[2][:,i] for i in success_indices ];
successratio = length(sol_point_vec_success)/reps
nsols, xsols, distvec = unique_length(sol_point_vec_success,1e-8);
nsols

# nsols, xsols, distvec = unique_length(xsols,1.5)

save("data/ParamFixing/SymmetryFix/ParamFixed_xvec$(xvec)_ivec$(ivec)_reps$(reps)_iter$(iter).jld2","xvec",xvec,"ivec",ivec,"sol_point_vec",sol_point_vec,"successratio",successratio,"nsols",nsols,"xsols",xsols,"distvec",distvec,"ϵ_success",ϵ_success)

