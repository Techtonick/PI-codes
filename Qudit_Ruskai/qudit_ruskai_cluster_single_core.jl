# The main difference from qudit_ruskai.jl is that here we parallelize operations to make the code friendly for running on a cluster

using LinearAlgebra
using Optim 
using LineSearches
using Combinatorics
#using Plots; plotlyjs(size = (700, 700))
using PlotlyJS
using BenchmarkTools
using JLD2
using Base.Threads
using SpecialFunctions
#gr(size = (700, 700))
using Transducers
using Folds
using OnlineStats
using FLoops
using MicroCollections
using BangBang 
using FileIO

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
logpartbinom(n,λ) = (checknonneg(λ) == 1) ? (logfactorial(n) - sum(logfactorial(λ[i]) for i in eachindex(λ))) : 0;
mw(λ) = mod(sum(λ[i]*(i-1) for i in eachindex(λ)), length(λ));

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
    
    binoms_diff = zeros(Float64,vλ,vν,vμ)
    for k in 1:vμ
        μview = view(μ, k, :)
        for l in 1:vν
            νview = view(ν, l, :)
            for iλ in 1:vλ
                λview = view(λ, iλ, :)
                
                x_min_y!(λ_minus_μ, λview, μview)
                x_min_y_plus_z!(λ_minus_μ_plus_ν, λview, μview, νview)
                
                if checknonneg(λ_minus_μ) != 0 && checknonneg(λ_minus_μ_plus_ν) != 0
                    binoms_diff[iλ,l,k] = rule4binoms_diff(n, t, λview, λ_minus_μ, λ_minus_μ_plus_ν)
                end
            end
            
        end
    end
    return binoms_diff
    
end

function nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν)

    λ_minus_μ = zeros(Int, q)
    λ_minus_μ_plus_ν = zeros(Int, q)
        
    nonneg_matr = zeros(Int64,vλ,vν,vμ)
    for k in 1:vμ
        μview = view(μ, k, :)  
        for l in 1:vν
            νview = view(ν, l, :)
            
            for iλ in 1:vλ
                λview = view(λ, iλ, :)
                
                x_min_y!(λ_minus_μ, λview, μview)
                x_min_y_plus_z!(λ_minus_μ_plus_ν, λview, μview, νview)
                
                if checknonneg(λ_minus_μ) != 0 && checknonneg(λ_minus_μ_plus_ν) != 0
                    nonneg_matr[iλ,l,k] = 1
                end
            end
            
        end
    end
    
    return nonneg_matr

end

#Function for finding the ordering index of an input partition
function partition_find_fast(partition, n, q, memo)
    """
    Fast version using memorization to avoid recomputing partition counts.
    """
    
    # Validate input
    if length(partition) != q
        return 0
    end
    
    if sum(partition) != n
        return 0
    end
    
    # Convert to vector if it's a matrix row
    if isa(partition, Matrix)
        partition = vec(partition)
    end
    
    # Memorization cache for partition counting
    # memo = Dict{Tuple{Int,Int}, Int}()
    
    function count_partitions(remaining_sum, remaining_parts)
        if remaining_parts == 0
            return remaining_sum == 0 ? 1 : 0
        end
        
        key = (remaining_sum, remaining_parts)
        if haskey(memo, key)
            return memo[key]
        end
        
        count = 0
        for i in 0:remaining_sum
            count += count_partitions(remaining_sum - i, remaining_parts - 1)
        end
        
        memo[key] = count
        return count
    end
    
    position = 1
    
    for pos in q:-1:1
        current_value = partition[pos]
        partition_view = @view partition[1:pos-1]
        remaining_sum = sum(partition_view)
        remaining_parts = pos - 1
        
        for i in 0:(current_value-1)
            temp_remaining = remaining_sum + (current_value - i)
            position += count_partitions(temp_remaining, remaining_parts)
        end
    end
    
    return position
end

function partition_pos_precompute(n, q, d, t, λ, μ, ν, vλ, vμ, vν)
    memocache = Dict{Tuple{Int,Int}, Int}()
    λ_minus_μ_plus_ν = zeros(Int64,q)
    pos = zeros(Int64,vλ,vν,vμ)
    for k in 1:vμ
        μview = @view μ[k,:]
        for l in 1:vν
            νview = @view ν[l,:]
            for iλ in 1:vλ                        
                if nonneg_cached[iλ,l,k] != 0
                    λview = @view λ[iλ,:]
                    x_min_y_plus_z!(λ_minus_μ_plus_ν, λview, μview, νview)
                    pos[iλ,l,k] = partition_find_fast(λ_minus_μ_plus_ν, n, q, memocache)
                end
            end
        end
    end
    return pos
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
                        if nonneg_cached[iλ,l,k] != 0

                            code_i_iλ = codewords_i[iλ]
                            code_j_iλ_k_l = codewords_j[pos_cached[iλ,l,k]]
                            λsum_rule4 += @fastmath code_i_iλ' * code_j_iλ_k_l * binoms_diff_cached[iλ,l,k]
                            
                            code_i_iλ_k_l = codewords_i[pos_cached[iλ,l,k]]
                            code_j_iλ = codewords_j[iλ]
                            λsum_rule5 += @fastmath (code_i_iλ' * code_i_iλ_k_l - code_j_iλ' * code_j_iλ_k_l) * binoms_diff_cached[iλ,l,k]

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

function replace_element(arr, new_element, pos)
    outarr = []
    for i in eachindex(arr)
        if i == pos
            push!(outarr, new_element)
        else
            push!(outarr, arr[i])
        end
    end
    return outarr
end

#Supplemental function for the cost function
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

#Precompute positions of zeros everywhere
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

#Define cost function with fixing codeword coefficients according to Ruskai codes (padded)
function costr(n, q, d, t, codewords, codewords_copy, λ, μ, ν, vλ, vμ, vν)
    for i in 1:d
        for k in 1:vλ
            if (mw(λ[k,:]) == (i-1)) && (mw(vec_λnew[i,k,:]) == 0)
                codewords_copy[(k+(i-1)*vλ)] = codewords[Int(vec_knew[i,k])]
            else
                codewords_copy[(k+(i-1)*vλ)] = 0
            end
        end
    end   
    return rules1(n,q,d,codewords_copy) + rule4_5(n,q,d,t,codewords_copy,λ,μ,ν,vλ,vμ,vν)
end

#Cost function of free variables, optimized with precomputed zeros vector
function padded_costr(n, q, d, t, free_vars, codewords_copy, λ, μ, ν, vλ, vμ, vν)
    counter = 1
    for k in 1:vλ
        if zeros_cached[k] == 1
            codewords_copy[k] = free_vars[counter]
            counter += 1
        else
            codewords_copy[k] = 0
        end
    end

    for i in 2:d
        for k in 1:vλ
            if zeros_cached[(k+(i-1)*vλ)] == 1
                codewords_copy[(k+(i-1)*vλ)] = codewords_copy[Int(vec_knew[i,k])]
            else
                codewords_copy[(k+(i-1)*vλ)] = 0
            end
        end
    end   

    return rules1(n,q,d,codewords_copy) + rule4_5(n,q,d,t,codewords_copy,λ,μ,ν,vλ,vμ,vν)
end

callback(state) = (abs(state.value) < optim_soltol ? (return true) : (return false) );

function ruskai_optim(n, q, d, t, λ, μ, ν, vλ, vμ, vν)
    #Preallocate before loops
    num_var = Int(vλ / q)
    codewords_copy = zeros(ComplexF64, d*vλ)
    costcl(free_vars) = padded_costr(n, q, d, t, free_vars, codewords_copy, λ, μ, ν, vλ, vμ, vν)
        free_cb = normalize(rand(ComplexF64, num_var))
        #@time begin
            res = Optim.optimize(costcl, free_cb, LBFGS(linesearch=LineSearches.BackTracking()),
                        Optim.Options(iterations=10000,
                                    g_tol=1e-8,
                                    f_tol=1e-8,
                                    allow_f_increases=true,
                                    show_trace=false,
                                    callback=callback)
                            )
        #end   
    minval = res.minimum
    minc = res.minimizer
    return [minval, minc]
end

const n = parse(Int64, ARGS[1])
const t = parse(Int64, ARGS[2])
const repcount = parse(Int64, ARGS[3])

const q = 3;
const d = 3;
const optim_soltol = 1e-18;
const λ = partitions_into_q_parts(n,q);
const μ = partitions_into_q_parts(2t,q);
const ν = partitions_into_q_parts(2t,q);
const vλ = size(λ)[1];
const vμ = size(μ)[1];
const vν = size(ν)[1];
const binoms_diff_cached = binoms_diff_precompute(n, q, t, λ, μ, ν, vλ, vμ, vν);
const nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν);
const vec_knew = vknew(n, q, d, λ, vλ)[1];
const vec_λnew = vknew(n, q, d, λ, vλ)[2];
const pos_cached = partition_pos_precompute(n, q, d, t, λ, μ, ν, vλ, vμ, vν);
const zeros_cached = zeros_precompute(n, q, d, λ, vλ);

results = ruskai_optim(n, q, d, t, λ, μ, ν, vλ, vμ, vν);
res_minimum = results[1];
res_minimizer = results[2];

filestring = "data_n$(n)_t$(t)_repnumber$(repcount).jld2" 
save(filestring,"t",t,"n",n,"repnumber",repcount,"minimum",res_minimum,"minimizer",res_minimizer,"optim_soltol",optim_soltol)

#FOR n=25, q=d=3, t=2:
#LBFGS takes 5s for one step of optim
#GradientDescent takes 9s for one step 

#n=10, t=1, optim steps = 10000, LBFGS, yes precomputed grad!, btime = 16.113 seconds
#n=10, t=1, optim steps = 10000, LBFGS, no precomputed grad!, btime = 5.533 seconds

#FOR n=25, q=d=3, t=2:
#LBFGS takes 0.277 s for one step of optim after removing padding and passing only free variables (cost function evaluation still padded, so suboptimal)
#Average speedup of ~18 times!

#BENCHMARKING

#= @profview ruskai_optim(range_n[1], q, d, range_t[1], aλ[1], aμ[1], aν[1], avλ[1], avμ[1], avν[1])
@btime ruskai_optim(range_n[1], q, d, range_t[1], aλ[1], aμ[1], aν[1], avλ[1], avμ[1], avν[1])

cdwrds_copy = zeros(ComplexF64, d*avλ[1])
num_var = Int(avλ[1] / q)
cs = zeros(ComplexF64, d*avλ[1])
cb = zeros(ComplexF64, num_var)
counter = 1
@views cb .= normalize(rand(ComplexF64, num_var))
@views vec_knew = arr_vec_knew[1]
counter = 1
for l in 1:avλ[1]
    if vec_knew[1,l] ≠ 0
        cs[l] = cb[Int(counter)]
        counter += 1
    end
end =#


"END"

#= function test_b()
    num_var = Int(vλ / q)
    cb = ones(num_var) #+ im*ones(num_var) # this is for complex
    cs = zeros(ComplexF64, d*vλ)
    counter = 1
    for l in 1:vλ
        if vec_knew[1,l] ≠ 0
            cs[l] = cb[Int(counter)]
            counter += 1
        end
    end
    return cs
end

cs = test_b()
cs_copy = zeros(ComplexF64, d*vλ)

function testf(reps)
    for i in 1:reps 
        costr(n, q, d, t, cs, cs_copy, λ, μ, ν, vλ, vμ, vν)
    end
end

#@profview testf(10000)
#@time testf(10000)

@btime costr(n, q, d, t, cs, cs_copy, λ, μ, ν, vλ, vμ, vν) =#

#0.115673 seconds (1000 allocations: 15.625 KiB)

#println(costr(n, q, d, t, cs, cs_copy, λ, μ, ν, vλ, vμ, vν))

#= @show partitions_into_q_parts(4,3)
memocache = Dict{Tuple{Int,Int}, Int}();
vecin = [2;2;0];
n = 4;
d = 3;
@show partition_find_fast(vecin, n, d, memocache)
@btime partition_find_fast(vecin, n, d, memocache)

function test(nreps)
    for i in 1:nreps
        partition_find_fast(vecin, n, d, memocache)
    end
end

test(1000000)
@profview test(1000000) =#

#@show vec_knew
#= function test_c()
    num_var = Int(vλ / q)
    #@var bo5x0x0
    #@var bo2x3x0
    #@var bo3x1x1
    #@var bo0x4x1
    #@var bo1x2x2
    #@var bo2x0x3
    #@var bo0x1x4
    #cb = 1:num_var # this is for complex
    #cb = [bo5x0x0, bo2x3x0, bo3x1x1, bo0x4x1, bo1x2x2, bo2x0x3, bo0x1x4]
    cs = zeros(ComplexF64, d*vλ)
    counter = 1
    for l in 1:vλ
        if vec_knew[1,l] ≠ 0
            cs = replace_element(cs,cb[Int(counter)],l)
            counter += 1
        end
    end
    return cs
end =#


#@var a[1:vλ*d]

#println(costr(n, q, d, t, cs, λ, μ, ν, vλ, vμ, vν)[1:21])
#println(costr(n, q, d, t, cs, λ, μ, ν, vλ, vμ, vν)[22:42])
#println(costr(n, q, d, t, cs, λ, μ, ν, vλ, vμ, vν)[43:63])

#println(expand(costr(n, q, d, t, cs, λ, μ, ν, vλ, vμ, vν)))
#println(costr(n, q, d, t, ones(d*vλ), λ, μ, ν, vλ, vμ, vν))
#println("CS IS BELOW")
#println(" ")
#println(cs)
#println(costrtest(n, q, d, t, cs, λ, μ, ν, vλ, vμ, vν))