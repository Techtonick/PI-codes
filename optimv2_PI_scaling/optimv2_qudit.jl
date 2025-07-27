
# TODO: optimize functions.
#           - rules is extremely fast (50ns). 
#           - rules4_5 also now quite fast 

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

# n = 2; # number of physical qudits 
# q = 2; # physical qudit dimension 
# d = 2; # logical qudit dimension

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

#Old, incorrect:
#= function rule4_5(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν)

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

end =#

#Function for finding the ordering index of an input partition
function partition_find_fast(partition, n, q, memo)
    """
    Fast version using memoization to avoid recomputing partition counts.
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
    
    # Memoization cache for partition counting
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
    pos = zeros(Int64,vμ,vν,vλ)
    for k in 1:vμ
        μview = @view μ[k,:]
        for l in 1:vν
            νview = @view ν[l,:]
            for iλ in 1:vλ                        
                if nonneg_cached[k,l,iλ] != 0
                    λview = @view λ[iλ,:]
                    x_min_y_plus_z!(λ_minus_μ_plus_ν, λview, μview, νview)
                    pos[k,l,iλ] = partition_find_fast(λ_minus_μ_plus_ν, n, q, memocache)
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
                        if nonneg_cached[k,l,iλ] != 0

                            code_i_iλ = codewords_i[iλ]
                            code_j_iλ_k_l = codewords_j[pos_cached[k,l,iλ]]
                            λsum_rule4 += code_i_iλ' * code_j_iλ_k_l * binoms_diff_cached[k,l,iλ]
                            
                            code_i_iλ_k_l = codewords_i[pos_cached[k,l,iλ]]
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

### ---------------------- Optimisation Code ---------------------- ### 

const n = 37;
const q = 2;
const d = 2;
const t = 3;
const λ = partitions_into_q_parts(n,q);
const μ = partitions_into_q_parts(2t,q);
const ν = partitions_into_q_parts(2t,q);
const vλ = size(λ)[1];
const vμ = size(μ)[1];
const vν = size(ν)[1];
const binoms_diff_cached = binoms_diff_precompute(n, q, t, λ, μ, ν, vλ, vμ, vν);
const nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν);
const optim_soltol = 1e-15;
const pos_cached = partition_pos_precompute(n, q, d, t, λ, μ, ν, vλ, vμ, vν)

#Define cost function
cost(c0) = abs(rules1(n,q,d,c0)) + rule4_5(n,q,d,t,c0,λ,μ,ν,vλ,vμ,vν)  

# Define cost function with fixing  
function costf(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs, fixed_vals)
    for i in eachindex(fixed_locs)
        codewords[fixed_locs[i]] = fixed_vals[i]
    end
    return abs(rules1(n,q,d,codewords)) + rule4_5(n,q,d,t,codewords,λ,μ,ν,vλ,vμ,vν)
end

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

#minval = 3.187052164707976e-8 for n=36,t=3
#minval = 7.274001751452536e-10 for n=37,t=3

#@show minx0

#= @threads for i = 1:10
    arr[i] = Threads.threadid()
end

println(arr) =#

"END"

#= function stupid()
    num_var = Int(vλ / q)
    cb = im*ones(num_var) + ones(num_var) # this is for complex
    cs = zeros(ComplexF64, d*vλ)
    counter = 1
    for l in 1:vλ
        if vec_knew[1,l] ≠ 0
            cs[l] = cb[Int(counter)]
            counter += 1
        end
    end
    return cs
end =#

#num_var=vλ*d
#= x0 = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
     0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1,
     0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0]
@show cost(x0)

function testf(reps)
    for i in 1:reps 
        cost(x0)
    end
end =#

#@profview testf(1000)
#@time testf(1000)

#126.9558791020091

"END"


# save("solns/qudit/n$(n)_q$(q)_d$(d)_t$(t)/file1.jld2","minval",minval,"minx0",minx0)


#= # Optimize using LBFGS 
xa = 1
ya = 3
za = 5
res_loop_minimum = []; res_loop_minimizer = [];
@btime for i in 1:1
    fixed_locs = [xa, ya, za]
    fixed_vals = [0,0,0]
    costcl(codewords) = costf(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs, fixed_vals)
    x0 = normalize(rand(num_var_params - 3)) # this is for real
    x0 = [x0[1:xa-1];0;x0[xa:ya-1];0;x0[ya:za-1];0;x0[za:end]]
    #costcs = cc -> cost([cc[1:xa-1];0;cc[xa+1:ya-1];0;cc[ya+1:za-1];0;cc[za+1:end]])
    res = optimize(costcl, x0, LBFGS(linesearch=LineSearches.BackTracking()),
                Optim.Options(iterations=5000,
                            g_tol=1e-6,
                            f_tol=1e-6,
                            allow_f_increases=true,
                            show_trace=false,
                            callback=callback)
                    )
    # println("iteration $i: min val = $(res.minimum)")
    push!(res_loop_minimum, res.minimum)
    push!(res_loop_minimizer, res.minimizer )
end
minval,minloc = findmin(res_loop_minimum)
minx0 = res_loop_minimizer[minloc]

@show minval
@show minx0 =#


#= ### ---------------------- Benchmarking ---------------------- ### 
# check n = 7 qubit case 
n = 7; 
q = 2; 
d = 2; 
t = 1;
λ = partitions_into_q_parts(n,q)
μ = partitions_into_q_parts(2t,q);
ν = partitions_into_q_parts(2t,q);
vλ = size(λ)[1];
vμ = size(μ)[1];
vν = size(ν)[1];
const binoms_diff_cached = binoms_diff_precompute(n, q, t, λ, μ, ν, vλ, vμ, vν);
const nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν);

# Initial guess 
codeword_length = size(partitions_into_q_parts(n,q))[1]
num_var_params = d * codeword_length
x0 = rand(num_var_params)

@time rules1(n,q,d,x0)
@btime rules1($n,$q,$d,$x0)

@time rule4_5(n,q,d,t,x0,λ,μ,ν,vλ,vμ,vν)
@code_warntype rule4_5(n,q,d,t,x0,λ,μ,ν,vλ,vμ,vν)
@btime rule4_5($n,$q,$d,$t,$x0,$λ,$μ,$ν,$vλ,$vμ,$vν)

function rule4_5_loop(n,q,d,t,x0,λ,μ,ν,vλ,vμ,vν)
    for i in 1:50000000
        rule4_5(n,q,d,t,x0,λ,μ,ν,vλ,vμ,vν)
    end
end
rule4_5_loop(n,q,d,t,x0,λ,μ,ν,vλ,vμ,vν)
@profview rule4_5_loop(n,q,d,t,x0,λ,μ,ν,vλ,vμ,vν)


### ---------------------- Debugging ---------------------- ### 

# check yingkai example 6.4. this requires n = 18, q = 2, d = 3, t = 1 
zeroL = [1/3;0;0;0;0;0;0;0;0;sqrt(7)/3;0;0;0;0;0;0;0;0;1/3]
oneL = [0;0;0;sqrt(3)/3;0;0;0;0;0;0;0;0;sqrt(6)/3;0;0;0;0;0;0]
twoL = [0;0;0;0;0;0;sqrt(6)/3;0;0;0;0;0;0;0;0;sqrt(3)/3;0;0;0]
yingkai_test = [zeroL;oneL;twoL]
cost(yingkai_test)

# check against solution generated with vlad. this requires n = 19, q = 2, d = 2, t = 2 
zeroL = [0.1;0.3;0.5;0.7;0.9;1.1;1.3;1.5;1.7;1.9;2.0;1.8;1.6;1.4;1.2;1.0;0.8;0.6;0.4;0.2]
oneL = zeroL .+ 2
vlad_test = [zeroL;oneL]
cost(vlad_test)

# check ordering 
test_index = 20; 
λ[test_index,:]
oneL[test_index]

rules1(n,q,d,vlad_test)
rule4_5(n,q,d,t,vlad_test,λ,μ,ν,vλ,vμ,vν) =#

# check yingkai example 6.4. this requires n = 18, q = 2, d = 3, t = 1 
#= zeroL = [1/3;0;0;0;0;0;0;0;0;sqrt(7)/3;0;0;0;0;0;0;0;0;1/3]
oneL = [0;0;0;sqrt(3)/3;0;0;0;0;0;0;0;0;sqrt(6)/3;0;0;0;0;0;0]
twoL = [0;0;0;0;0;0;sqrt(6)/3;0;0;0;0;0;0;0;0;sqrt(3)/3;0;0;0]
yingkai_test = [zeroL;oneL;twoL]
cost(yingkai_test) =#

# check against solution generated with vlad. this requires n = 19, q = 2, d = 2, t = 2 
#= zeroL = [0.1;0.3;0.5;0.7;0.9;1.1;1.3;1.5;1.7;1.9;2.0;1.8;1.6;1.4;1.2;1.0;0.8;0.6;0.4;0.2]
oneL = zeroL .+ 2
vlad_test = [zeroL;oneL]
cost(vlad_test) =#