
# TODO: check that my optimizer finds qubit codes that agree with Vlad's mathematica script 

# TODO: optimize functions. eg. remove allocations, remove repeated computations, try reduce # for loops. 
#           - rules is extremely fast (50ns). 
#           - rules4_5 still relatively slow (7μs). Its basically non-allocating, so I don't immedaitely see how to speed it up. 

# TODO: try and find qutrit solutions 

using LinearAlgebra
using Optim 
using LineSearches
using Combinatorics
using PyPlot
using BenchmarkTools
using JLD2
using Base.Threads
using SpecialFunctions


function partitions_into_q_parts(n, q)
    # Start with an array of q parts filled with zeros
    partitions = zeros(Int64,0,q)
    
    # Recursive function to generate partitions
    function generate_partition(partition, remaining, parts_left)
        # Base case: if no more parts are left and no remaining value, store partition
        if parts_left == 0
            if remaining == 0
                partitions = vcat(partitions, copy(partition))
            end
            return
        end
        
        # Loop to assign values to the current part, ranging from 0 to the remaining value
        for i in 0:remaining
            partition[parts_left] = i
            generate_partition(partition, remaining - i, parts_left - 1)
        end
    end
    
    # Initialize the partition array
    generate_partition(zeros(Int, 1,q), n, q)
        
    return partitions
end;

heaviside(x) = x ≥ 0 ? 1 : 0;
checknonneg(λ) = all(x -> x ≥ 0, λ);
#partbinom(n,λ) = (checknonneg(λ) == 1) ? (factorial(n)/prod(factorial(λ[i]) for i in eachindex(λ))) : 0;
logpartbinom(n,λ) = (checknonneg(λ) == 1) ? (logfactorial(n) - sum(logfactorial(λ[i]) for i in eachindex(λ))) : 0;

function rules1(n,q,d,codewords)

    local_num_var_params = length(codewords)
    local_codeword_length = Int(local_num_var_params/d)

    outval = 0 
    @inbounds for i in 1:d 
        @inbounds for j in 1:d 
            if i ≥ j 
                @views outval += abs2( abs(codewords[(1+(i-1)*local_codeword_length):local_codeword_length*i]' * codewords[(1+(j-1)*local_codeword_length):local_codeword_length*j]) - (i==j) )
            end
        end
    end
    return outval

end

#rule4binoms_diff(n,t,λ,λ_minus_μ,λ_minus_μ_plus_ν) = ( checknonneg(λ_minus_μ) == 1 ? partbinom(n-2t,λ_minus_μ)/( sqrt(partbinom(n,λ)*partbinom(n,λ_minus_μ_plus_ν)) ) : 0 )
rule4binoms_diff(n,t,λ,λ_minus_μ,λ_minus_μ_plus_ν) = ( checknonneg(λ_minus_μ) == 1 ? exp( logpartbinom(n-2t,λ_minus_μ) - ( 1/2*(logpartbinom(n,λ)+logpartbinom(n,λ_minus_μ_plus_ν)) ) ) : 0 )

function x_min_y!(out,x,y)
    for i in eachindex(x)
        out[i] = x[i] - y[i] 
    end 
end;
function x_min_y_plus_z!(out,x,y,z)
    for i in eachindex(x)
        out[i] = x[i] - y[i] + z[i]
    end 
end;

function binoms_diff_precompute(n, q, t, λ, μ, ν, vλ, vμ, vν)
    
    λ_minus_μ = zeros(Int, q)
    λ_minus_μ_plus_ν = zeros(Int, q)
    
    binoms_diff = zeros(Float64,vμ,vν,vλ)
    for k in 1:vμ
        μview = view(μ, k, :)
        for l in 1:vν
            νview = view(ν, l, :)
            for iλ in 1:vλ
                λview = view(λ, iλ, :)
                
                x_min_y!(λ_minus_μ, λview, μview)
                x_min_y_plus_z!(λ_minus_μ_plus_ν, λview, μview, νview)
                
                if checknonneg(λ_minus_μ) != 0 && checknonneg(λ_minus_μ_plus_ν) != 0
                    binoms_diff[k,l,iλ] = rule4binoms_diff(n, t, λview, λ_minus_μ, λ_minus_μ_plus_ν)
                end
            end
            
        end
    end
    return binoms_diff
    
end

function nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν)

    λ_minus_μ = zeros(Int, q)
    λ_minus_μ_plus_ν = zeros(Int, q)
        
    nonneg_matr = zeros(Int64,vμ,vν,vλ)
    for k in 1:vμ
        μview = view(μ, k, :)  
        for l in 1:vν
            νview = view(ν, l, :)
            
            for iλ in 1:vλ
                λview = view(λ, iλ, :)
                
                x_min_y!(λ_minus_μ, λview, μview)
                x_min_y_plus_z!(λ_minus_μ_plus_ν, λview, μview, νview)
                
                if checknonneg(λ_minus_μ) != 0 && checknonneg(λ_minus_μ_plus_ν) != 0
                    nonneg_matr[k,l,iλ] = 1
                end
            end
            
        end
    end
    
    return nonneg_matr

end


function rule4_5(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν)

    # Preallocate outside loops
    local_num_var_params = length(codewords)
    local_codeword_length = Int(local_num_var_params/d)
    outval1 = 0.0
    outval2 = 0.0
        
    @inbounds for i in 1:d-1
        @views codewords_i = codewords[(1+(i-1)*local_codeword_length):local_codeword_length*i]
            
        for j in i+1:d
            @views codewords_j = codewords[(1+(j-1)*local_codeword_length):local_codeword_length*j]
            
            for k in 1:vμ
                for l in 1:vν
                    λsum_rule4 = 0.0
                    λsum_rule5 = 0.0
                    
                    for iλ in 1:vλ                        
                        if nonneg_cached[k,l,iλ] != 0

                            code_i_iλ = codewords_i[iλ]
                            code_j_iλ_k_l = codewords_j[iλ - k + l]
                            λsum_rule4 += code_i_iλ' * code_j_iλ_k_l * binoms_diff_cached[k,l,iλ]
                            
                            code_i_iλ_k_l = codewords_i[iλ - k + l]
                            code_j_iλ = codewords_j[iλ]
                            λsum_rule5 += (code_i_iλ' * code_i_iλ_k_l - code_j_iλ' * code_j_iλ_k_l) * binoms_diff_cached[k,l,iλ]

                        end
                    end  
                    outval1 += abs2(λsum_rule4)
                    outval2 += abs2(λsum_rule5)
                end
            end
        end
    end
    
    return outval1 + outval2

end

# # check n = 7 qubit Ruskai codes  
# n = 7; 
# q = 2; 
# d = 2; 
# t = 1;
# λ = partitions_into_q_parts(n,q);
# μ = partitions_into_q_parts(2t,q);
# ν = partitions_into_q_parts(2t,q);
# vλ = size(λ)[1];
# vμ = size(μ)[1];
# vν = size(ν)[1];
# binoms_diff_cached = binoms_diff_precompute(n, q, t, λ, μ, ν, vλ, vμ, vν); # TODO: make const 
# nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν); # TODO: make const 
# ruskai_codewords_c0 = [sqrt(15);0;-sqrt(7);0;sqrt(21);0;sqrt(21);0]/8;
# ruskai_codewords_c1 = [0;sqrt(21);0;sqrt(21);0;-sqrt(7);0;sqrt(15)]/8;
# codewords = [ruskai_codewords_c0;ruskai_codewords_c1];

# rules1(n,q,d,codewords)
# rule4_5(n,q,d,t,codewords,λ,μ,ν,vλ,vμ,vν)

### ---------------------- Optimisation Code ---------------------- ### 

const n = 61;
const q = 2;
const d = 2;
const t = 4;
const λ = partitions_into_q_parts(n,q);
const μ = partitions_into_q_parts(2t,q);
const ν = partitions_into_q_parts(2t,q);
const vλ = size(λ)[1];
const vμ = size(μ)[1];
const vν = size(ν)[1];
const binoms_diff_cached = binoms_diff_precompute(n, q, t, λ, μ, ν, vλ, vμ, vν);
const nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν);
const optim_soltol = 1e-15;

# Define cost function  
cost(x) = abs(rules1(n,q,d,x)) + rule4_5(n,q,d,t,x,λ,μ,ν,vλ,vμ,vν)

callback(state) = (abs(state.value) < optim_soltol ? (return true) : (return false) );

const codeword_length = size(partitions_into_q_parts(n,q))[1]
const num_var_params = d * codeword_length
#= x0 = rand(num_var_params)
cost(x0) =#
# Optimize
res_loop_minimum = []; res_loop_minimizer = []; thread_arr = [];
println("START")
@threads for i in 1:256
    x0 = normalize(rand(num_var_params)) # this is for real 
    #x0 = normalize(rand(ComplexF64,num_var_params)) # for Vlad: this is complex 
    res = optimize(cost, x0, LBFGS(linesearch=LineSearches.BackTracking()), #LBFGS(linesearch=LineSearches.BackTracking()),
                Optim.Options(iterations=100000,
                            g_tol=1e-30,
                            f_tol=1e-30,
                            allow_f_increases=true,
                            show_trace=true,
                            callback=callback)
                    )
    println("iteration $i: min val = $(res.minimum)")
    push!(res_loop_minimum, res.minimum)
    push!(res_loop_minimizer, res.minimizer)
    push!(thread_arr, threadid())
end
minval,minloc = findmin(res_loop_minimum)
minx0 = res_loop_minimizer[minloc]

@show minval
println(thread_arr)



#= ### ---------------------- Benchmarking ---------------------- ### 
# check n = 7 qubit case 
n = 7; 
q = 2; 
d = 2; 
t = 1;
λ = partitions_into_q_parts(n,q);
μ = partitions_into_q_parts(2t,q);
ν = partitions_into_q_parts(2t,q);
vλ = size(λ)[1];
vμ = size(μ)[1];
vν = size(ν)[1];
binoms_diff_cached = binoms_diff_precompute(n, q, t, λ, μ, ν, vλ, vμ, vν);
nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν);

# ruskai_codewords_c0 = [sqrt(15);0;-sqrt(7);0;sqrt(21);0;sqrt(21);0]/8;
# ruskai_codewords_c1 = [0;sqrt(21);0;sqrt(21);0;-sqrt(7);0;sqrt(15)]/8;
# codewords = [ruskai_codewords_c0;ruskai_codewords_c1];

# Initial guess 
codeword_length = size(partitions_into_q_parts(n,q))[1]
num_var_params = d * codeword_length
x0 = rand(num_var_params)

@time rules1(n,q,d,x0)
@btime rules1($n,$q,$d,$x0)

@time rule4_5(n,q,d,t,x0,λ,μ,ν,vλ,vμ,vν)
@btime rule4_5($n,$q,$d,$t,$x0,$λ,$μ,$ν,$vλ,$vμ,$vν)

function rule4_5_loop(n,q,d,t,x0,λ,μ,ν,vλ,vμ,vν)
    for i in 1:10000000
        rule4_5(n,q,d,t,x0,λ,μ,ν,vλ,vμ,vν)
    end
end
rule4_5_loop(n,q,d,t,x0,λ,μ,ν,vλ,vμ,vν)
@profview rule4_5_loop(n,q,d,t,x0,λ,μ,ν,vλ,vμ,vν)

### check symmetry question 
x0_0 = normalize(rand(8))

function state1gen(x0_0)
    num_el = length(x0_0)
    x0_1 = zeros(num_el)
    for i in eachindex(x0_0)
        x0_1[i] = (-1)^(i+1) * x0_0[num_el-i+1]
    end 
    return x0_1 
end
x0_1 = state1gen(x0_0)


rules1(n,q,d,[x0_0;x0_1])
rule4_5(n,q,d,t,[x0_0;x0_1],λ,μ,ν,vλ,vμ,vν) =#