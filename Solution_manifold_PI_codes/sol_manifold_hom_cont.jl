using LinearAlgebra
using Optim 
using LineSearches
using Combinatorics
using Plots
using BenchmarkTools
using JLD2
using HomotopyContinuation
gr(size = (800, 800))

n = 7; # init number of physical qudits 
q = 2; # init physical qudit dimension 
d = 2; # init logical qudit dimension
t = 1; # init number of errors

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
partbinom(n,λ) = (checknonneg(λ) == 1) ? (factorial(n)/prod(factorial(λ[i]) for i in eachindex(λ))) : 0;

function rules1(n,q,d,codewords)

    local_num_var_params = length(codewords)
    local_codeword_length = Int(local_num_var_params/d)

    outarr = []
    @inbounds for i in 1:d 
        @inbounds for j in 1:d 
            if i ≥ j 
                push!(outarr, codewords[(1+(i-1)*local_codeword_length):local_codeword_length*i]' * codewords[(1+(j-1)*local_codeword_length):local_codeword_length*j] - (i==j))
            end
        end
    end
    return outarr

end

rule4binoms_diff(n,t,λ,λ_minus_μ,λ_minus_μ_plus_ν) = ( checknonneg(λ_minus_μ) == 1 ? partbinom(n-2t,λ_minus_μ)/( sqrt(partbinom(n,λ)*partbinom(n,λ_minus_μ_plus_ν)) ) : 0 )

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
    outarr4 = []
    outarr5 = []
        
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
                    push!(outarr4, λsum_rule4)
                    push!(outarr5, λsum_rule5)
                end
            end
        end
    end
    
    return [outarr4, outarr5]

end

### ---------------------- Solutions Manifold ---------------------- ###

#= n = 7;
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
nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν); =#

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

function sol_points(step, x, y)
    outarr_x = []
    outarr_y = []
    global n 
    global q 
    global d 
    global t 
    global λ 
    global μ 
    global ν 
    global vλ
    global vμ
    global vν
    global binoms_diff_cached
    global nonneg_cached
    @var a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8
    avec = [a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8]
    avec = replace_element(avec, 0, x)
    avec = replace_element(avec, 0, y)
    for i in -1:step:1
        avec[x] = i
        for j in -1:step:1
            if 0.16 <= i^2 + j^2 <= 0.36
                avec[y] = j
                c = [avec[1], avec[2], avec[3], avec[4], avec[5], avec[6], avec[7], avec[8], avec[8], -avec[7], avec[6], -avec[5], avec[4], -avec[3], avec[2], -avec[1]]
                rulesvector = [rules1(n,q,d,c),rule4_5(n,q,d,t,c,λ,μ,ν,vλ,vμ,vν)]
                rulesvector = collect(Iterators.flatten(rulesvector))
                rulesvector = collect(Iterators.flatten(rulesvector))
                F = System(rulesvector)
                result = solve(F;start_system=:total_degree)
                if real_solutions(result) != []
                    push!(outarr_x, avec[x])
                    push!(outarr_y, avec[y])
                end
            end
        end
    end
    return [outarr_x, outarr_y]
end

#= @var a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8
avec = [0, a_2, a_3, a_4, a_5, a_6, a_7, sqrt(2)/3 -0.000001]
c = [avec[1], avec[2], avec[3], avec[4], avec[5], avec[6], avec[7], avec[8], avec[8], -avec[7], avec[6], -avec[5], avec[4], -avec[3], avec[2], -avec[1]]
rulesvector = [rules1(n,q,d,c),rule4_5(n,q,d,t,c,λ,μ,ν,vλ,vμ,vν)]
rulesvector = collect(Iterators.flatten(rulesvector))
rulesvector = collect(Iterators.flatten(rulesvector))
F = System(rulesvector)
result = solve(F)
real_solutions(result) =#

#= xa = 1
ya = 8
solpts = sol_points(0.025,xa,ya)

xvector = solpts[1]
yvector = solpts[2]
scatter(xvector, yvector, legend=false, ylimits=(-1,1), xlimits=(-1,1))
xlabel!("a_$xa")
ylabel!("a_$ya")
title!("Algebraic variety 2d projection. Fixed coefficients: $xa, $ya") =#

n = 19;
q = 2;
d = 2;
t = 2;
λ = partitions_into_q_parts(n,q);
μ = partitions_into_q_parts(2t,q);
ν = partitions_into_q_parts(2t,q);
vλ = size(λ)[1];
vμ = size(μ)[1];
vν = size(ν)[1];
binoms_diff_cached = binoms_diff_precompute(n, q, t, λ, μ, ν, vλ, vμ, vν);
nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν);

#@var a[1:(n+1)]

function zeros_everywhere(arr, i, j, special)
    l = length(arr)
    out = arr
    for k in 1:l
        if k != i && k != j && k != special
            out = replace_element(out, 0, k)
        end
    end
    return out
end

function replace_zeros(a, start)
    outarr = []
    l = length(a)
    for i in (start+1):(l-1)
        for j in (i+1):l
            push!(outarr, zeros_everywhere(a, i, j, start))
        end
    end
    return outarr
end

function pi_sym(a)
    n = length(a)
    b = []
    for i in 1:n
        push!(b, (-1)^(i-1)*a[n-(i-1)])
    end
    return b
end

function finding_sols(start)
    global n 
    global q 
    global d 
    global t 
    global λ 
    global μ 
    global ν 
    global vλ
    global vμ
    global vν
    global binoms_diff_cached
    global nonneg_cached
    @var a[1:(n+1)]
    avector = replace_zeros(a, start)
    l = length(avector)
    for i in 1:l
        a0 = avector[i]
        c = [a0, pi_sym(a0)]
        c = collect(Iterators.flatten(c))
        rulesvector = [rules1(n,q,d,c),rule4_5(n,q,d,t,c,λ,μ,ν,vλ,vμ,vν)]
        rulesvector = collect(Iterators.flatten(rulesvector))
        rulesvector = collect(Iterators.flatten(rulesvector))
        F = System(rulesvector)
        result = solve(F)
        println(real_solutions(result))
        println(a0)
    end
end

#= for i in 2:18
    finding_sols(i)
end =#

@var a[1:(n+1)]
#@var b[1:(n+1)]
a0 = a
#b0 = b
#a0 = replace_element(a0, 0, 2)
a0 = replace_element(a0, 0, 3)
a0 = replace_element(a0, 0, 4)
a0 = replace_element(a0, 0, 5)
a0 = replace_element(a0, 0, 6)
a0 = replace_element(a0, 0, 7)
a0 = replace_element(a0, 0, 9)
a0 = replace_element(a0, 0, 11)
a0 = replace_element(a0, 0, 12)
a0 = replace_element(a0, 0, 13)
a0 = replace_element(a0, 0, 14)
a0 = replace_element(a0, 0, 15)
a0 = replace_element(a0, 0, 16)
a0 = replace_element(a0, 0, 18)
a0 = replace_element(a0, 0, 19)
a0 = replace_element(a0, 0, 20)
c = [a0, pi_sym(a0)]
c = collect(Iterators.flatten(c))
#= @var x 
@var y
y = conj(x)
@show [x,y] =#
#= c = [-0.4622107685291028 - 0.45597706603690147im, 0.6397718874547429 + 2.4390470319685527im, 2.0393150228597787 + 0.4806265114176439im, 
-2.53840868705064 + 0.7885844153027655im, -1.2109931296850276 + 1.751691326205311im, -1.4873195463578766 - 1.8189192052890415im,
-0.9856474787693799 - 0.7385035645159848im, 1.2818364886320763 - 1.6082571285947425im, 1.4025106773967477 + 1.8413126216224789im,
1.223391185485244 - 1.5837683410542889im, -2.1381407852762013 + 1.6920156980840477im, 2.261879675944938 + 1.9744823040126922im,
-1.989634203845159 - 1.6136287260568611im, -2.1347842297273725 + 1.6177054793948402im, 0.8499636507033455 - 1.7006669255487874im, -0.23798072459510927 - 0.8204878969725549im] =#
rulesvector = [rules1(n,q,d,c),rule4_5(n,q,d,t,c,λ,μ,ν,vλ,vμ,vν)]
rulesvector = collect(Iterators.flatten(rulesvector))
rulesvector = collect(Iterators.flatten(rulesvector))
#@show rulesvector
F = System(rulesvector)
#result = solve(F;start_system=:total_degree,stop_early_cb = is_success,threading = true)
#result = solve(F;start_system=:total_degree,stop_early_cb = is_real,threading = true)
result = solve(F;threading = true)
println(c)
sols = real_solutions(result)
@show sols
#@show results(result; only_real=false, real_tol=1e-10)
#sols = solutions(result; only_nonsingular = false, only_real=false)
#sols = collect(Iterators.flatten(sols))
#@show sols
#@show [rules1(n,q,d,sols),rule4_5(n,q,d,t,sols,λ,μ,ν,vλ,vμ,vν)]


#= vect = [-1.258344159677198 - 0.19048338416869068im, 0.4591437883268832 - 0.6297448325522669im, 0.21298676142505588 - 0.1649603295694236im, 
0.17092924831216028 - 1.4330176920864248im, 0.5538421048045136 + 0.5669422735153568im, 0.15217941042861552 + 0.09269379253051002im,
 1.3557101284102198 + 0.1786913629408307im, -0.41639772068573616 + 0.5783593185085645im, 0.41639772068573916 - 0.5783593185085674im, 
 1.3557101284102182 + 0.17869136294082966im, -0.1521794104286177 - 0.09269379253050755im, 0.5538421048045135 + 0.5669422735153569im, 
 -0.17092924831216136 + 1.433017692086426im, 0.2129867614250559 - 0.16496032956942427im, -0.4591437883268796 + 0.6297448325522628im, -1.2583441596771994 - 0.19048338416869245im] =#

#real_solutions(result)
#nonsingular(result)
#results(result; only_real=true)[1]

#= # Store codewords separately
minx0array = reshape(minx0, (codeword_length,d)) =#

#= # Plot
xvector = partitions_into_q_parts(n, q)
xvector = [xvector[i,:] for i in 1:codeword_length]
xvector = string.(xvector)
y1vector = abs.(minx0array[:, 1])
y2vector = abs.(minx0array[:, 2])
y3vector = abs.(minx0array[:, 3])
#y4vector = abs.(minx0array[:, 4])
plot(xvector, y1vector, xticks=(0.5:1:codeword_length-0.5, xvector), label="First codeword")
plot!(xvector, y2vector, xticks=(0.5:1:codeword_length-0.5, xvector), label="Second codeword")
plot!(xvector, y3vector, xticks=(0.5:1:codeword_length-0.5, xvector), label="Third codeword")
#plot!(xvector, y4vector, xticks=(0.5:1:codeword_length-0.5, xvector), label="Fourth codeword")
xlabel!("Partition label")
ylabel!("Coefficient abs value")
title!("Abs value plot. #qutrits: $n ; Cost: $minval") =#