
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
const optim_soltol = 1e-25;
const pos_cached = partition_pos_precompute(n, q, d, t, λ, μ, ν, vλ, vμ, vν);

#Define cost function
cost(c0) = abs(rules1(n,q,d,c0)) + rule4_5(n,q,d,t,c0,λ,μ,ν,vλ,vμ,vν);

callback(state) = (abs(state.f_x) < optim_soltol ? (return true) : (return false) );

const codeword_length = size(partitions_into_q_parts(n,q))[1];
const num_var_params = d * codeword_length;

mw(λ) = mod(sum(λ[i]*(i-1) for i in eachindex(λ)), length(λ));

function zeros_precompute(n, q, d, λ, vλ)
    zeros_pos = zeros(ComplexF64, d*vλ)
    for i in 1:d
        for k in 1:vλ
            if (mw(λ[k,:]) == (i-1)) && (mw(vec_λnew[i,k,:]) == 0)
                zeros_pos[(k+(i-1)*vλ)] = 1
            else
                zeros_pos[(k+(i-1)*vλ)] = 0
            end
        end
    end
    return zeros_pos
end

function vknew(n, q, d, λ, vλ) 
    # Preallocate outside loops
    memocache = Dict{Tuple{Int,Int}, Int}()
    kvec = zeros(Int, d, vλ)
    λnewvec = zeros(Int, d, vλ, q)
    for i in 1:d
        for k in 1:vλ
            λnew = λ[k,[i:q ; 1:(i-1)]]
            if (mw(λ[k,:]) == (i-1)) && (mw(λnew) == 0) 
                knew = partition_find_fast(λnew, n, q, memocache)
                kvec[i,k] = Int(knew)
                @views λnewvec[i,k,:] .= λnew
            end
        end
    end   
    return [kvec, λnewvec]
end

const vec_knew = vknew(n, q, d, λ, vλ)[1];
const vec_λnew = vknew(n, q, d, λ, vλ)[2];
const zeros_cached = zeros_precompute(n, q, d, λ, vλ);

#Cost function of free variables, optimized with precomputed zeros vector
function padded_costr(n, q, d, t, free_vars, codewords_copy, λ, μ, ν, vλ, vμ, vν)
    
    counter = 1
    for k in 1:vλ
        if zeros_cached[k] == 1
            codewords_copy[k] = free_vars[counter]
            counter += 1
        end
    end

    for i in 2:d
        for k in 1:vλ
            if zeros_cached[(k+(i-1)*vλ)] == 1
                codewords_copy[(k+(i-1)*vλ)] = codewords_copy[Int(vec_knew[i,k])]
            end
        end
    end   

    return abs(rules1(n,q,d,codewords_copy)) + rule4_5(n,q,d,t,codewords_copy,λ,μ,ν,vλ,vμ,vν)

end

function ruskai_optim(n, q, d, t, λ, μ, ν, vλ, vμ, vν)

    #Preallocate before loops
    num_var = Int(vλ / q)

    # codewords_copy = zeros(d*vλ)
    codewords_copy = zeros(BigFloat,d*vλ)

    costcl(free_vars) = padded_costr(n, q, d, t, free_vars, codewords_copy, λ, μ, ν, vλ, vμ, vν)
        
    free_cb = normalize(rand(BigFloat,num_var))
    # free_cb = normalize(rand(num_var))

    res = Optim.optimize(costcl, free_cb, LBFGS(linesearch=LineSearches.BackTracking()), 
                Optim.Options(iterations=100000,
                            g_tol=1e-20,
                            f_reltol=0.0,
                            f_abstol=0.0,
                            allow_f_increases=true,
                            show_trace=false,
                            callback=callback,
                            time_limit = 60 * 60 * 24 * 1) # time limit of 1 days (units of seconds)
                    )

    return res

end

println("start optimization.")

res = ruskai_optim(n, q, d, t, λ, μ, ν, vλ, vμ, vν)

minval = res.minimum;
minx0 = res.minimizer;
stoppedby = res.stopped_by
optimiters = res.iterations;

println("saving. minimum was $minval. exit code was $([k for (k, v) in pairs(stoppedby) if v])")

save("data/Ruskai_BigFloat/n$(n)_q$(q)_d$(d)_t$(t)/iter$(iter).jld2","minval",minval,"minx0",minx0,"n",n,"t",t,"d",d,"iter",iter,"q",q,"stoppedby",stoppedby,"optimiters",optimiters,"res",res)