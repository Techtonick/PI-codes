using LinearAlgebra
using Optim 
using LineSearches
using Combinatorics
#using Plots; plotlyjs(size = (700, 700))
using PlotlyJS
using BenchmarkTools
using JLD2
#using HomotopyContinuation
#gr(size = (700, 700))

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

### ---------------------- Solutions Manifold ---------------------- ###

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

function zeros_everywhere(arr, i, j, k)
    l = length(arr)
    out = arr
    for m in 1:l
        if m != i && m != j && m != k
            out = replace_element(out, 0, m)
        end
    end
    return out
end

function replace_zeros(a)
    outarr = []
    l = length(a)
    for i in (start+1):(l-2)
        for j in (i+1):(l-1)
            for k in (j+1):l
                push!(outarr, zeros_everywhere(a, i, j, k))
            end
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

function insertat!(arr, sorted_list, sorted_items)
    out = arr
    for i in eachindex(sorted_list)
        out = insert!(out, i, sorted_items[i])
    end
    return out
end

function renormalize(vec, fixed)
    if norm(fixed) <= 1
        renorm = sqrt(1 - norm(fixed)^2)
        if (renorm != 0) && (norm(vec) != 0)
            sf = norm(vec)/renorm
            outarr = vec/sf
        else 
            outarr = zeros(length(vec))
        end
        return outarr
    else
        return "ERROR: make norm(fixed) <= 1, stupid"
    end
end

# Define cost function with fixing  
function costf(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs, fixed_vals)
    for i in eachindex(fixed_locs)
        codewords[fixed_locs[i]] = fixed_vals[i]
    end
    return abs(rules1(n,q,d,codewords)) + rule4_5(n,q,d,t,codewords,λ,μ,ν,vλ,vμ,vν)
end

callback(state) = (abs(state.value) < optim_soltol ? (return true) : (return false) );

function sol_points(step, x, y, reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
    outarr_x = []
    outarr_y = []
    sols = []
    codeword_length = size(partitions_into_q_parts(n,q))[1]
    num_var_params = d * codeword_length
    cost(c0) = rules1(n,q,d,c0) + rule4_5(n,q,d,t,c0,λ,μ,ν,vλ,vμ,vν)
    if x < y
        for i in 0:step:0.85
            for j in 0:step:0.85
                if i^2 + j^2 <= 1
                    res_loop_minimum = []; res_loop_minimizer = [];
                    fixed_locs = [x, y]
                    fixed_vals = [i, j]
                    costcl(codewords) = costf(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs, fixed_vals)
                    #costclosure = cc -> cost([cc[1:x-1];i;cc[x+1:y-1];j;cc[y+1:z-1];l;cc[z+1:end]])
                    #@show cost([0, 0, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32])
                    #vectorrr = [9, 2, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32]
                    println("i is: $i, j is: $j")
                    #println([vectorrr[1:x-1];i;vectorrr[x+1:y-1];j;vectorrr[y+1:end]])
                    #@show costclosure(vectorrr)
                    lmin = 2*ε
                    if sols != []
                        cs = sols[end]
                        #@time begin
                            #println("start: $cs")
                            res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                        Optim.Options(iterations=5000,
                                                    g_tol=1e-7,
                                                    f_tol=1e-7,
                                                    allow_f_increases=true,
                                                    show_trace=false,
                                                    callback=callback)
                                            )
                        #end   
                        push!(res_loop_minimum, res.minimum)
                        push!(res_loop_minimizer, res.minimizer)
                        lmin = res.minimum
                    end
                    if lmin > ε
                        for k in 1:reps
                            #@time begin
                                cs =  renormalize(rand(num_var_params - t - 1), [i,j]) # this is for real
                                cs = insert!(cs, x, 0)
                                cs = insert!(cs, y, 0)
                            #end
                            #@time begin
                                #println("start: $cs")
                                res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                            Optim.Options(iterations=5000,
                                                        g_tol=1e-6,
                                                        f_tol=1e-6,
                                                        allow_f_increases=true,
                                                        show_trace=false,
                                                        callback=callback)
                                                )
                            #end
                            #println("iteration $k: min val = $(res.minimum)")    
                            push!(res_loop_minimum, res.minimum)
                            push!(res_loop_minimizer, res.minimizer)
                            if res.minimum <= ε
                                break
                            else 
                                continue
                            end
                        end
                    else
                        println("SKIPPED LOOP!")
                    end
                    minval,minloc = findmin(res_loop_minimum)
                    minx0 = res_loop_minimizer[minloc]
                    if minval <= ε
                        push!(outarr_x, i)
                        push!(outarr_y, j)
                        push!(sols, minx0)
                    end
                end
            end
        end
        return [outarr_x, outarr_y]
    else 
        return "ERROR: make it x < y, stupid"
    end
end

function sol_points3D(step, x, y, z, reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
    outarr_x = []
    outarr_y = []
    outarr_z = []
    sols = []
    codeword_length = size(partitions_into_q_parts(n,q))[1]
    num_var_params = d * codeword_length
    cost(c0) = abs(rules1(n,q,d,c0)) + rule4_5(n,q,d,t,c0,λ,μ,ν,vλ,vμ,vν)
    if x < y < z
        for i in 0:step:0.7
            for j in 0:step:0.7
                for l in 0:step:0.7
                    if i^2 + j^2 + l^2 <= 1
                        res_loop_minimum = []; res_loop_minimizer = [];
                        fixed_locs = [x, y, z]
                        fixed_vals = [i, j, l]
                        costcl(codewords) = costf(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs, fixed_vals)
                        #costclosure = cc -> cost([cc[1:x-1];i;cc[x+1:y-1];j;cc[y+1:z-1];l;cc[z+1:end]])
                        #@show cost([0, 0, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32])
                        #vectorrr = [9, 2, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32]
                        println("i is: $i, j is: $j, l is: $l")
                        #println([vectorrr[1:x-1];i;vectorrr[x+1:y-1];j;vectorrr[y+1:end]])
                        #@show costclosure(vectorrr)
                        lmin = 2*ε
                        @time begin
                            if sols != []
                                cs = sols[end]
                                #@time begin
                                    #println("start: $cs")
                                    res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                                Optim.Options(iterations=5000,
                                                            g_tol=1e-6,
                                                            f_tol=1e-6,
                                                            allow_f_increases=true,
                                                            show_trace=false,
                                                            callback=callback)
                                                    )
                                #end   
                                push!(res_loop_minimum, res.minimum)
                                push!(res_loop_minimizer, res.minimizer)
                                lmin = res.minimum
                            end
                            if lmin > ε
                                for k in 1:reps
                                    #@time begin
                                        cs =  renormalize(rand(num_var_params - t - 1), [i,j,l]) # this is for real
                                        cs = insert!(cs, x, 0)
                                        cs = insert!(cs, y, 0)
                                        cs = insert!(cs, z, 0)
                                    #end
                                    #@time begin
                                        #println("start: $cs")
                                        res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                                    Optim.Options(iterations=5000,
                                                                g_tol=1e-6,
                                                                f_tol=1e-6,
                                                                allow_f_increases=true,
                                                                show_trace=false,
                                                                callback=callback)
                                                        )
                                    #end
                                    #println("iteration $k: min val = $(res.minimum)")    
                                    push!(res_loop_minimum, res.minimum)
                                    push!(res_loop_minimizer, res.minimizer)
                                    if res.minimum <= ε
                                        break
                                    else 
                                        continue
                                    end
                                end
                            else
                                println("SKIPPED LOOP!")
                            end
                        end
                        minval,minloc = findmin(res_loop_minimum)
                        minx0 = res_loop_minimizer[minloc]
                        if minval <= ε
                            push!(outarr_x, i)
                            push!(outarr_y, j)
                            push!(outarr_z, l)
                            push!(sols, minx0)
                        end
                    end
                end
            end
        end
        return [outarr_x, outarr_y, outarr_z]
    else 
        return "ERROR: make it x < y < z, stupid"
    end
end

function sharpener(point, num_iter, interval, direction_angle, reps, x, y, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
    outarr_x = []
    outarr_y = []
    sols = []
    codeword_length = size(partitions_into_q_parts(n,q))[1]
    num_var_params = d * codeword_length
    cost(c0) = rules1(n,q,d,c0) + rule4_5(n,q,d,t,c0,λ,μ,ν,vλ,vμ,vν)
    if x < y
        fin_point = [point[1] + interval*cos(direction_angle), point[2] + interval*sin(direction_angle)]
        in_point = [point[1], point[2]]
        for l in 1:num_iter
            i = (in_point[1] + fin_point[1]) / 2
            j = (in_point[2] + fin_point[2]) / 2
            res_loop_minimum = []; res_loop_minimizer = [];
            fixed_locs = [x, y]
            fixed_vals = [i, j]
            costcl(codewords) = costf(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs, fixed_vals)
            #costclosure = cc -> cost([cc[1:x-1];i;cc[x+1:y-1];j;cc[y+1:z-1];l;cc[z+1:end]])
            #@show cost([0, 0, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32])
            #vectorrr = [9, 2, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32]
            println("i is: $i, j is: $j")
            #println([vectorrr[1:x-1];i;vectorrr[x+1:y-1];j;vectorrr[y+1:end]])
            #@show costclosure(vectorrr)
            lmin = 2*ε
            if sols != []
                cs = sols[end]
                #@time begin
                    #println("start: $cs")
                    res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                Optim.Options(iterations=10000,
                                            g_tol=1e-10,
                                            f_tol=1e-10,
                                            allow_f_increases=true,
                                            show_trace=false,
                                            callback=callback)
                                    )
                #end   
                push!(res_loop_minimum, res.minimum)
                push!(res_loop_minimizer, res.minimizer)
                lmin = res.minimum
            end
            if lmin > ε
                for k in 1:reps
                    #@time begin
                        cs =  renormalize(rand(num_var_params - t - 1), [i,j]) # this is for real
                        cs = insert!(cs, x, 0)
                        cs = insert!(cs, y, 0)
                    #end
                    #@time begin
                        #println("start: $cs")
                        res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                    Optim.Options(iterations=10000,
                                                g_tol=1e-10,
                                                f_tol=1e-10,
                                                allow_f_increases=true,
                                                show_trace=false,
                                                callback=callback)
                                        )
                    #end
                    #println("iteration $k: min val = $(res.minimum)")    
                    push!(res_loop_minimum, res.minimum)
                    push!(res_loop_minimizer, res.minimizer)
                    if res.minimum <= ε
                        break
                    else 
                        continue
                    end
                end
            else
                println("SKIPPED LOOP!")
            end
            minval,minloc = findmin(res_loop_minimum)
            minx0 = res_loop_minimizer[minloc]
            if minval <= ε
                push!(outarr_x, i)
                push!(outarr_y, j)
                push!(sols, minx0)
                in_point = [i,j]
            else
                fin_point = [i,j]
            end
        end
        return [outarr_x, outarr_y, sols[end]]
    else 
        return "ERROR: make it x < y, stupid"
    end
end

function sol_zoom3D(zoom_scale, center, resolution, x, y, z, reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
    outarr_x = []
    outarr_y = []
    outarr_z = []
    sols = []
    codeword_length = size(partitions_into_q_parts(n,q))[1]
    num_var_params = d * codeword_length
    cost(c0) = abs(rules1(n,q,d,c0)) + rule4_5(n,q,d,t,c0,λ,μ,ν,vλ,vμ,vν)
    stx = (center[1] - 0.3/zoom_scale)
    finx = (center[1] + 0.3/zoom_scale)
    sty = (center[2] - 0.3/zoom_scale)
    finy = (center[2] + 0.3/zoom_scale)
    stz = (center[3] - 0.3/zoom_scale)
    finz = (center[3] + 0.3/zoom_scale)
    step = resolution/zoom_scale
    if x < y < z
        for i in stx:step:finx
            for j in sty:step:finy
                for l in stz:step:finz
                    if i^2 + j^2 + l^2 <= 1
                        res_loop_minimum = []; res_loop_minimizer = [];
                        fixed_locs = [x, y, z]
                        fixed_vals = [i, j, l]
                        costcl(codewords) = costf(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs, fixed_vals)
                        #costclosure = cc -> cost([cc[1:x-1];i;cc[x+1:y-1];j;cc[y+1:z-1];l;cc[z+1:end]])
                        #@show cost([0, 0, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32])
                        #vectorrr = [9, 2, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32]
                        println("i is: $i, j is: $j, l is: $l")
                        #println([vectorrr[1:x-1];i;vectorrr[x+1:y-1];j;vectorrr[y+1:end]])
                        #@show costclosure(vectorrr)
                        lmin = 2*ε
                        @time begin
                            if sols != []
                                cs = sols[end]
                                #@time begin
                                    #println("start: $cs")
                                    res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                                Optim.Options(iterations=5000,
                                                            g_tol=1e-6,
                                                            f_tol=1e-6,
                                                            allow_f_increases=true,
                                                            show_trace=false,
                                                            callback=callback)
                                                    )
                                #end   
                                push!(res_loop_minimum, res.minimum)
                                push!(res_loop_minimizer, res.minimizer)
                                lmin = res.minimum
                            end
                            if lmin > ε
                                for k in 1:reps
                                    #@time begin
                                        cs =  renormalize(rand(num_var_params - t - 1), [i,j,l]) # this is for real
                                        cs = insert!(cs, x, 0)
                                        cs = insert!(cs, y, 0)
                                        cs = insert!(cs, z, 0)
                                    #end
                                    #@time begin
                                        #println("start: $cs")
                                        res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                                    Optim.Options(iterations=5000,
                                                                g_tol=1e-6,
                                                                f_tol=1e-6,
                                                                allow_f_increases=true,
                                                                show_trace=false,
                                                                callback=callback)
                                                        )
                                    #end
                                    #println("iteration $k: min val = $(res.minimum)")    
                                    push!(res_loop_minimum, res.minimum)
                                    push!(res_loop_minimizer, res.minimizer)
                                    if res.minimum <= ε
                                        break
                                    else 
                                        continue
                                    end
                                end
                            else
                                println("SKIPPED LOOP!")
                            end
                        end
                        minval,minloc = findmin(res_loop_minimum)
                        minx0 = res_loop_minimizer[minloc]
                        if minval <= ε
                            push!(outarr_x, i)
                            push!(outarr_y, j)
                            push!(outarr_z, l)
                            push!(sols, minx0)
                        end
                    end
                end
            end
        end
        #CENTER SOLUTION:
        res_loop_minimum = []; res_loop_minimizer = [];
        i = center[1]
        j = center[2]
        l = center[3]
        fixed_locs = [x, y, z]
        fixed_vals = [i, j, l]
        costcl(codewords) = costf(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs, fixed_vals)
        #costclosure = cc -> cost([cc[1:x-1];i;cc[x+1:y-1];j;cc[y+1:z-1];l;cc[z+1:end]])
        #@show cost([0, 0, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32])
        #vectorrr = [9, 2, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32]
        println("i is: $i, j is: $j, l is: $l")
        #println([vectorrr[1:x-1];i;vectorrr[x+1:y-1];j;vectorrr[y+1:end]])
        #@show costclosure(vectorrr)
        lmin = 2*ε
        @time begin
            if sols != []
                cs = sols[end]
                #@time begin
                    #println("start: $cs")
                    res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                Optim.Options(iterations=5000,
                                            g_tol=1e-6,
                                            f_tol=1e-6,
                                            allow_f_increases=true,
                                            show_trace=false,
                                            callback=callback)
                                    )
                #end   
                push!(res_loop_minimum, res.minimum)
                push!(res_loop_minimizer, res.minimizer)
                lmin = res.minimum
            end
            if lmin > ε
                for k in 1:reps
                    #@time begin
                        cs =  renormalize(rand(num_var_params - t - 1), [i,j,l]) # this is for real
                        cs = insert!(cs, x, 0)
                        cs = insert!(cs, y, 0)
                        cs = insert!(cs, z, 0)
                    #end
                    #@time begin
                        #println("start: $cs")
                        res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                    Optim.Options(iterations=5000,
                                                g_tol=1e-6,
                                                f_tol=1e-6,
                                                allow_f_increases=true,
                                                show_trace=false,
                                                callback=callback)
                                        )
                    #end
                    #println("iteration $k: min val = $(res.minimum)")    
                    push!(res_loop_minimum, res.minimum)
                    push!(res_loop_minimizer, res.minimizer)
                    if res.minimum <= ε
                        break
                    else 
                        continue
                    end
                end
            else
                println("SKIPPED LOOP!")
            end
        end
        minval,minloc = findmin(res_loop_minimum)
        minx0 = res_loop_minimizer[minloc]
        if minval <= ε
            center_sol = minx0
        else
            center_sol = "Didn't converge"
        end
        return [outarr_x, outarr_y, outarr_z, sols, center_sol]
    else 
        return "ERROR: make it x < y < z, stupid"
    end
end

function a_solutions(sols)
    outarr = []
    for i in eachindex(sols)
        push!(outarr, sols[i][1:20])
    end
    return outarr
end

#= # Define cost function with fixing zeros
function costf0(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs)
    l = length(codewords)
    cl = trunc(Int, l/d)
    for i in 1:cl
        if (i != fixed_locs[1]) && (i != fixed_locs[2])
            codewords[i] = 0
        end
    end
    return abs(rules1(n,q,d,codewords)) + rule4_5(n,q,d,t,codewords,λ,μ,ν,vλ,vμ,vν)
end

function finding_sols(reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
    outarr = []
    sols = []
    codeword_length = size(partitions_into_q_parts(n,q))[1]
    #num_var_params = d * codeword_length
    for i in 1:(codeword_length-1)
        for j in (i+1):(codeword_length)
            res_loop_minimum = []; res_loop_minimizer = [];
            fixed_locs = [i, j]
            costcl0(codewords) = costf0(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs)
            println("i is: $i, j is: $j")
            @time begin
                for l in 1:reps
                    #@time begin
                        csr = normalize(rand(2))
                        cs1 = zeros(codeword_length)
                        cs1[i] = csr[1]
                        cs1[j] = csr[2]
                        cs2 = pi_sym(cs1)
                        cs2 = convert(Array{Float64}, cs2)
                        cs = [cs1[1:end];cs2[1:end]]
                    #end
                    #@time begin
                        #println("start: $cs")
                        res = Optim.optimize(costcl0, cs, #= LBFGS(linesearch=LineSearches.BackTracking()), =# NelderMead(),
                                    Optim.Options(iterations=10000,
                                                g_tol=1e-10,
                                                f_tol=1e-10,
                                                allow_f_increases=true,
                                                show_trace=false,
                                                callback=callback)
                                        )
                    #end
                    #println("iteration $k: min val = $(res.minimum)")    
                    push!(res_loop_minimum, res.minimum)
                    push!(res_loop_minimizer, res.minimizer)
                    if res.minimum <= ε
                        break
                    else 
                        continue
                    end
                end
            end
            minval,minloc = findmin(res_loop_minimum)
            minx0 = res_loop_minimizer[minloc]
            if minval <= ε
                push!(outarr, [i,j])
                push!(sols, minx0)
            end
        end
    end
    return [outarr, sols]
end =#

# Define cost function with fixing zeros
function costf0(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs)
    l = length(codewords)
    cl = trunc(Int, l/d)
    for i in 1:cl
        if (i != fixed_locs[1]) && (i != fixed_locs[2]) && (i != fixed_locs[3]) && (i != fixed_locs[4]) && (i != fixed_locs[5]) && (i != fixed_locs[6])
            codewords[i] = 0
        end
    end
    for i in (cl+1):2cl
        if (i != 2cl - fixed_locs[1] + 1) && (i != 2cl - fixed_locs[2] + 1) && (i != 2cl - fixed_locs[3] + 1) && (i != 2cl - fixed_locs[4] + 1) && (i != 2cl - fixed_locs[5] + 1) && (i != 2cl - fixed_locs[6] + 1)
            codewords[i] = 0
        end
    end
    return abs(rules1(n,q,d,codewords)) + rule4_5(n,q,d,t,codewords,λ,μ,ν,vλ,vμ,vν)
end

function finding_sols(reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
    outarr = []
    sols = []
    codeword_length = size(partitions_into_q_parts(n,q))[1]
    #num_var_params = d * codeword_length
    for i in 1:(codeword_length-5)
        for j in (i+1):(codeword_length-4)
            for k in (j+1):(codeword_length-3)
                for l in (k+1):(codeword_length-2)
                    for p in (l+1):(codeword_length-1)
                        for r in (p+1):codeword_length
                            res_loop_minimum = []; res_loop_minimizer = [];
                            fixed_locs = [i, j, k, l, p, r]
                            costcl0(codewords) = costf0(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs)
                            println("i is: $i, j is: $j, k is: $k, l is $l, p is $p, r is $r")
                            @time begin
                                for m in 1:reps
                                    #@time begin
                                        csr = normalize(rand(6))
                                        cs1 = zeros(codeword_length)
                                        cs1[i] = csr[1]
                                        cs1[j] = csr[2]
                                        cs1[k] = csr[3]
                                        cs1[l] = csr[4]
                                        cs1[p] = csr[5]
                                        cs1[r] = csr[6]
                                        cs2 = pi_sym(cs1)
                                        cs2 = convert(Array{Float64}, cs2)
                                        cs = [cs1[1:end];cs2[1:end]]
                                    #end
                                    #@time begin
                                        #println("start: $cs")
                                        res = Optim.optimize(costcl0, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                                    Optim.Options(iterations=5000,
                                                                g_tol=1e-10,
                                                                f_tol=1e-15,
                                                                allow_f_increases=true,
                                                                show_trace=false,
                                                                callback=callback)
                                                        )
                                    #end
                                    #println("iteration $k: min val = $(res.minimum)")    
                                    push!(res_loop_minimum, res.minimum)
                                    push!(res_loop_minimizer, res.minimizer)
                                    if res.minimum <= ε
                                        break
                                    else 
                                        continue
                                    end
                                end
                            end
                            minval,minloc = findmin(res_loop_minimum)
                            minx0 = res_loop_minimizer[minloc]
                            if minval <= ε
                                push!(outarr, [i,j,k,l,p,r])
                                push!(sols, minx0)
                            end
                        end
                    end
                end
            end
        end
    end
    return [outarr, sols]
end

const n = 7;
const q = 2;
const d = 2;
const t = 1;
const λ = partitions_into_q_parts(n,q);
const μ = partitions_into_q_parts(2t,q);
const ν = partitions_into_q_parts(2t,q);
const vλ = size(λ)[1];
const vμ = size(μ)[1];
const vν = size(ν)[1];
const binoms_diff_cached = binoms_diff_precompute(n, q, t, λ, μ, ν, vλ, vμ, vν);
const nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν);
const itstep = 0.02;
const reps = 20;
const ε = 1e-6;
const optim_soltol = 1e-6;
#const zoom_scale = 32
#const center = [0.3125, 0.0, 0.61875]

aydin_solution = [sqrt(3/10),0,0,0,0,sqrt(7/10),0,0]

#relayout!(p, title_text="Multiple Subplots with Titles")
#plot!(p, [aydin_solution[1]], [aydin_solution[6]], seriestype=:scatter, ms=3, color=:red, subplot=2)

#= t = 0:0.01:2π
plotto = plot(
    scatter(x=t, y=cos.(t), mode="lines"),
    Layout(yaxis_title="cos(t)", xaxis_title="t")
) =#

#= p = make_subplots(rows=3, cols=2, vertical_spacing=0.02)
add_trace!(p, scatter(x=0:2, y=10:12, mode="markers", name="(1,1)"), row=3, col=1)
add_trace!(p, scatter(x=t, y=cos.(t),name="(1,1)"), row=3, col=2)
add_trace!(p, scatter(x=2:4, y=100:10:120), row=3, col=1)
add_trace!(p, scatter(x=3:5, y=1000:100:1200), row=1, col=2)
relayout!(p, title_text="Stacked Subplots with Shared X-Axes")
p =#


plts = make_subplots(rows=8, cols=8, vertical_spacing=0.02)
for xa in 1:7
    for ya in (xa+1):8
        solpts = sol_points(itstep, xa, ya, reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
        xvector = solpts[1]
        xvector = [xvector[1:end];xvector[1:end];-xvector[1:end];-xvector[1:end]]
        yvector = solpts[2]
        yvector = [yvector[1:end];-yvector[1:end];yvector[1:end];-yvector[1:end]]
        add_trace!(plts, scatter(x=xvector, y=yvector, mode="markers", name="(Manifold for [$xa, $ya] fixed)"), row=xa, col=ya)
        #Plots.html(plt, "2d_plot_fine_$xa-$ya")
    end
end

for xa in 1:7
    for ya in (xa+1):8
        add_trace!(plts, scatter(x=[aydin_solution[xa]], y=[aydin_solution[ya]], mode="markers", name="(Aydin point for [$xa, $ya] fixed)"), row=xa, col=ya)
    end
end

relayout!(plts, title_text="Table of all algebraic variety 2D projections for n=7, t=1")
plts


#Plots.html(plts, "2d_plots_table")

#= sol = finding_sols(reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
#poss = reduce(hcat, [sol[1], sol[2]])
#poss = [sol[1], sol[2]]'
poss = sol[1]
sols = sol[2]

println("Positions: $poss")
println("Solutions: $sols") =#
#println("Aydin solution: $aydin_solution")


#= xa = 1;
ya = 8;
za = 10;
al_a = 17;

codeword_length = size(partitions_into_q_parts(n,q))[1]
res_loop_minimum = []; res_loop_minimizer = [];
fixed_locs = [xa, ya, za, al_a]
costcl0(codewords) = costf0(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs)
println("i is: $xa, j is: $ya, k is: $za, l is $al_a")
@time begin
    for m in 1:reps
        #@time begin
            csr = normalize(rand(5))
            cs1 = zeros(codeword_length)
            cs1[xa] = csr[1]
            cs1[ya] = csr[2]
            cs1[za] = csr[3]
            cs1[al_a] = csr[4]
            cs1[2] = csr[5]
            cs2 = pi_sym(cs1)
            cs2 = convert(Array{Float64}, cs2)
            cs = [cs1[1:end];cs2[1:end]]
        #end
        #@time begin
            #println("start: $cs")
            res = Optim.optimize(costcl0, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                        Optim.Options(iterations=100000,
                                    g_tol=1e-20,
                                    f_tol=1e-20,
                                    allow_f_increases=true,
                                    show_trace=false,
                                    callback=callback)
                            )
                            # @show res
        #end
        #println("iteration $k: min val = $(res.minimum)")    
        push!(res_loop_minimum, res.minimum)
        push!(res_loop_minimizer, res.minimizer)
        if res.minimum <= ε
            break
        else 
            continue
        end
    end
end
minval,minloc = findmin(res_loop_minimum)
minx0 = res_loop_minimizer[minloc]

println(minval)
println(minx0) =#

#= solpts = sol_points3D(itstep, xa, ya, za, reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
xvector = solpts[1]
xvector = [xvector[1:end];xvector[1:end];xvector[1:end];xvector[1:end];-xvector[1:end];-xvector[1:end];-xvector[1:end];-xvector[1:end]]
yvector = solpts[2]
yvector = [yvector[1:end];yvector[1:end];-yvector[1:end];-yvector[1:end];yvector[1:end];yvector[1:end];-yvector[1:end];-yvector[1:end]]
zvector = solpts[3]
zvector = [zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end]]
plt = plot(scatter(xvector, yvector, zvector, legend=false, ms=3, color=:blue, zlimits=(-1,1), ylimits=(-1,1), xlimits=(-1,1)))
xlabel!("a_$xa")
ylabel!("a_$ya")
zlabel!("a_$za")
title!("Algebraic variety 3d projection. Fixed coefficients: $xa, $ya, $za")
Plots.html(plt, "plot_$xa-$ya-$za") =#

#= solpts = sol_zoom3D(zoom_scale, center, itstep, xa, ya, za, reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
stx = (center[1] - 0.3/zoom_scale)
finx = (center[1] + 0.3/zoom_scale)
sty = (center[2] - 0.3/zoom_scale)
finy = (center[2] + 0.3/zoom_scale)
stz = (center[3] - 0.3/zoom_scale)
finz = (center[3] + 0.3/zoom_scale)
xvector = solpts[1]
xvector = [xvector[1:end];xvector[1:end];xvector[1:end];xvector[1:end];-xvector[1:end];-xvector[1:end];-xvector[1:end];-xvector[1:end]]
yvector = solpts[2]
yvector = [yvector[1:end];yvector[1:end];-yvector[1:end];-yvector[1:end];yvector[1:end];yvector[1:end];-yvector[1:end];-yvector[1:end]]
zvector = solpts[3]
zvector = [zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end]] =#

#= a_sols = a_solutions(solpts[4])
for i in eachindex(a_sols)
    println(a_sols[i])
end =#

#= center_sol = solpts[5]

println("Solution at center: $center_sol")

scatter(xvector, yvector, zvector, legend=false, ms=3, color=:blue, zlimits=(stz,finz), ylimits=(sty,finy), xlimits=(stx,finx))
xlabel!("a_$xa")
ylabel!("a_$ya")
zlabel!("a_$za")
title!("Edge point zoom = $zoom_scale. Fixed coefficients: $xa, $ya, $za") =#

#= const point = [0.62, 0.62]
const num_iter = 20000
const interval = 0.02
const direction_angle = pi/4 
const reps = 100;
const ε = 1e-13;
const optim_soltol = 1e-14; =#

#=  [0.2209466435936671, 4.610057677647839e-12, 2.4488743813063155e-12, 0.3897676570795215, 2.738612564998651e-14, 2.247427097679216e-12, 0.3897676570795215, 3.8984153714066156e-12] =#
#= [0.22094661538235436, 1.1295947806476828e-15, 1.5839557849635856e-14, 0.3897676569993496, 5.967128570698811e-13, 3.116293275277638e-13, 0.3897676569993496, 3.6850023076300423e-14] =#
#= [0.22094658829630118, 1.0918512087975133e-11, 4.5135864541562885e-11, 0.3897676570795215, 4.90837586481705e-11, 4.06478722175483e-11, 0.3897676570795215, 2.6318123672436156e-11] =#
#= [0.22094662872998225, 7.86070335746471e-13, 2.060418943388232e-12, 0.38976765717819484, 8.986242030520092e-12, 1.6627696437711969e-12, 0.38976765717819484, 5.190315840962435e-14] =#
#= [0.22220943616493682, 3.5174504807302716e-9, 1.1000095999956561e-11, 0.38889766560276917, 4.07201597216738e-9, 6.121657292133684e-12, 0.38889766560276917, 1.9472750321214626e-9] =#
#= [0.22222181866492313, 1.181674579547748e-16, 1.1118196016286348e-15, 0.3888891668212333, 3.907962717601832e-15, 8.887373975584962e-16, 0.3888891668212333, 2.3601231563403317e-18] =#

#= solpts = sharpener(point, num_iter, interval, direction_angle, reps, xa, ya, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
sharper_edge = ((solpts[3])[1:8]).^2
approx_sharper_edge = rationalize.(sharper_edge, tol=1e-5)
println("EDGE: $approx_sharper_edge") =#
#= xvector = solpts[1]
xvector = [xvector[1:end];xvector[1:end];-xvector[1:end];-xvector[1:end]]
yvector = solpts[2]
yvector = [yvector[1:end];-yvector[1:end];yvector[1:end];-yvector[1:end]]
scatter(xvector, yvector, legend=false, ms=1, color=:blue, ylimits=(0.62,0.625), xlimits=(0.62,0.625))
xlabel!("a_$xa")
ylabel!("a_$ya")
title!("Sharper edge at $point") =#

#= function test_goober()
    step = 1
    i = 1
    while (i <= 50)
        println(i)
        i += step
        step = step + 1
    end
end =#

#= solpts = sol_points(itstep, xa, ya, reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
xvector = solpts[1]
xvector = [xvector[1:end];xvector[1:end];-xvector[1:end];-xvector[1:end]]
yvector = solpts[2]
yvector = [yvector[1:end];-yvector[1:end];yvector[1:end];-yvector[1:end]]
scatter(xvector, yvector, legend=false, ms=3, color=:blue, ylimits=(-1,1), xlimits=(-1,1))
xlabel!("a_$xa")
ylabel!("a_$ya")
title!("Algebraic variety 2d projection. Fixed coefficients: $xa, $ya") =#

#= solpts = sol_points3D(itstep, xa, ya, za, reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
xvector = solpts[1]
xvector = [xvector[1:end];xvector[1:end];xvector[1:end];xvector[1:end];-xvector[1:end];-xvector[1:end];-xvector[1:end];-xvector[1:end]]
yvector = solpts[2]
yvector = [yvector[1:end];yvector[1:end];-yvector[1:end];-yvector[1:end];yvector[1:end];yvector[1:end];-yvector[1:end];-yvector[1:end]]
zvector = solpts[3]
zvector = [zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end]]
plt = plot(scatter(xvector, yvector, zvector, legend=false, ms=3, color=:blue, zlimits=(-1,1), ylimits=(-1,1), xlimits=(-1,1)))
xlabel!("a_$xa")
ylabel!("a_$ya")
zlabel!("a_$za")
title!("Algebraic variety 3d projection. Fixed coefficients: $xa, $ya, $za")
Plots.html(plt, "plot_$xa-$ya-$za") =#

#= for xa in 1:1
    for ya in (xa+1):2
        for za in (ya+1):3
            solpts = sol_points3D(itstep, xa, ya, za, reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
            xvector = solpts[1]
            xvector = [xvector[1:end];xvector[1:end];xvector[1:end];xvector[1:end];-xvector[1:end];-xvector[1:end];-xvector[1:end];-xvector[1:end]]
            yvector = solpts[2]
            yvector = [yvector[1:end];yvector[1:end];-yvector[1:end];-yvector[1:end];yvector[1:end];yvector[1:end];-yvector[1:end];-yvector[1:end]]
            zvector = solpts[3]
            zvector = [zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end];zvector[1:end];-zvector[1:end]]
            #= plt =  =#plot(scatter(xvector, yvector, zvector, legend=false, ms=3, color=:blue, zlimits=(-1,1), ylimits=(-1,1), xlimits=(-1,1)))
            xlabel!("a_$xa")
            ylabel!("a_$ya")
            zlabel!("a_$za")
            title!("Algebraic variety 3d projection. Fixed coefficients: $xa, $ya, $za")
            #Plots.html(plt, "plot_$xa-$ya-$za")
        end
    end
end =#

#= foo = [rand(10),rand(10),rand(10)]
xvector = foo[1]
yvector = foo[2]
zvector = foo[3]
scatter(xvector, yvector, zvector, legend=false, ms=3, color=:blue #= zlimits=(-1,1), ylimits=(-1,1), xlimits=(-1,1) =#)
xlabel!("x")
ylabel!("y")
zlabel!("z")
title!("Some test plot plotlyjs") =#

#@show solpts
#insert!([6, 5, 4, 3, 2, 1], 7, 0)
#= vectorrr = [0.5, 0.4]
@show (norm(vectorrr)^2 + norm(renormalize([6,1,1], vectorrr))^2) =#

#= xvector = solpts[1]
yvector = solpts[2]
scatter(xvector, yvector, legend=false, ylimits=(-1,1), xlimits=(-1,1))
xlabel!("a_$xa")
ylabel!("a_$ya")
title!("Algebraic variety 2d projection. Fixed coefficients: $xa, $ya") =#

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
nonneg_cached = nonneg_precompute(q, λ, μ, ν, vλ, vμ, vν);

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
 =#
#= for i in 2:18
    finding_sols(i)
end =#

#= @var a[1:(n+1)]
@var b[1:(n+1)]
a0 = a
b0 = b
#a0 = replace_element(a0, 0, 1)
#a0 = replace_element(a0, 0, 2)
#a0 = replace_element(a0, 0, 3)
a0 = replace_element(a0, 0, 9)
a0 = replace_element(a0, 0, 10)
a0 = replace_element(a0, 0, 11)
a0 = replace_element(a0, 0, 12)
c = [a0, b0]
c = collect(Iterators.flatten(c))
@show c
@var x 
@var y
y = conj(x)
@show [x,y] =#
#= c = [-0.4622107685291028 - 0.45597706603690147im, 0.6397718874547429 + 2.4390470319685527im, 2.0393150228597787 + 0.4806265114176439im, 
-2.53840868705064 + 0.7885844153027655im, -1.2109931296850276 + 1.751691326205311im, -1.4873195463578766 - 1.8189192052890415im,
-0.9856474787693799 - 0.7385035645159848im, 1.2818364886320763 - 1.6082571285947425im, 1.4025106773967477 + 1.8413126216224789im,
1.223391185485244 - 1.5837683410542889im, -2.1381407852762013 + 1.6920156980840477im, 2.261879675944938 + 1.9744823040126922im,
-1.989634203845159 - 1.6136287260568611im, -2.1347842297273725 + 1.6177054793948402im, 0.8499636507033455 - 1.7006669255487874im, -0.23798072459510927 - 0.8204878969725549im] =#
#= rulesvector = [rules1(n,q,d,c),rule4_5(n,q,d,t,c,λ,μ,ν,vλ,vμ,vν)]
rulesvector = collect(Iterators.flatten(rulesvector))
rulesvector = collect(Iterators.flatten(rulesvector))
@show rulesvector =#
#= F = System(rulesvector)
result = solve(F;start_system=:total_degree,stop_early_cb = is_success,threading = true)
println(c)
@show results(result; only_real=false, real_tol=1e-10) =#
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

#= 

Alternative to sol_points3D, with perturbation around a previous sol point, bad because of snowball-effect errors

function sol_points3D(step, x, y, z, reps, ε, n, q, d, t, λ, μ, ν, vλ, vμ, vν)
    outarr_x = []
    outarr_y = []
    outarr_z = []
    sols = []
    codeword_length = size(partitions_into_q_parts(n,q))[1]
    num_var_params = d * codeword_length
    cost(c0) = abs(rules1(n,q,d,c0)) + rule4_5(n,q,d,t,c0,λ,μ,ν,vλ,vμ,vν)
    if x < y < z
        for i in 0:step:0.7
            for j in 0:step:0.7
                for l in 0:step:0.7
                    if i^2 + j^2 + l^2 <= 1
                        res_loop_minimum = []; res_loop_minimizer = [];
                        fixed_locs = [x, y, z]
                        fixed_vals = [i, j, l]
                        costcl(codewords) = costf(n, q, d, t, codewords, λ, μ, ν, vλ, vμ, vν, fixed_locs, fixed_vals)
                        #costclosure = cc -> cost([cc[1:x-1];i;cc[x+1:y-1];j;cc[y+1:z-1];l;cc[z+1:end]])
                        #@show cost([0, 0, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32])
                        #vectorrr = [9, 2, 0.8366600265340756, -4.881276688943885e-33, -1.2026404222962906e-32, 3.313842311126637e-32, 1.308586079185809e-32, 0.5477225575051661, 0.5477225575051661, -2.2987713050905354e-33, 1.5098137440460015e-32, -8.858404570103842e-33, -4.336173109002335e-33, -0.8366600265340756, 2.2214455103673e-33, 2.3891432451743143e-32]
                        println("i is: $i, j is: $j, l is: $l")
                        #println([vectorrr[1:x-1];i;vectorrr[x+1:y-1];j;vectorrr[y+1:end]])
                        #@show costclosure(vectorrr)
                        lmin = 2*ε
                        @time begin
                            if sols != []
                                cs = sols[end]
                                #@time begin
                                    #println("start: $cs")
                                    res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                                Optim.Options(iterations=5000,
                                                            g_tol=1e-6,
                                                            f_tol=1e-6,
                                                            allow_f_increases=true,
                                                            show_trace=false,
                                                            callback=callback)
                                                    )
                                #end   
                                push!(res_loop_minimum, res.minimum)
                                push!(res_loop_minimizer, res.minimizer)
                                lmin = res.minimum
                            else
                                for k in 1:reps
                                    #@time begin
                                        cs =  renormalize(rand(num_var_params - t - 1), [i,j,l]) # this is for real
                                        cs = insert!(cs, x, 0)
                                        cs = insert!(cs, y, 0)
                                        cs = insert!(cs, z, 0)
                                    #end
                                    #@time begin
                                        #println("start: $cs")
                                        res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                                    Optim.Options(iterations=5000,
                                                                g_tol=1e-6,
                                                                f_tol=1e-6,
                                                                allow_f_increases=true,
                                                                show_trace=false,
                                                                callback=callback)
                                                        )
                                    #end
                                    #println("iteration $k: min val = $(res.minimum)")    
                                    push!(res_loop_minimum, res.minimum)
                                    push!(res_loop_minimizer, res.minimizer)
                                    if res.minimum <= ε
                                        break
                                    else 
                                        continue
                                    end
                                end
                                lmin = 0
                            end
                            if lmin > ε
                                #pow = trunc(Int, reps/3)
                                for k in 1:trunc(Int, reps/3)
                                    #@time begin
                                        #pow = pow - 1
                                        cs =  renormalize(rand(num_var_params - t - 1), [i,j,l]) * 0.3 # this is for real
                                        cs = insert!(cs, x, 0)
                                        cs = insert!(cs, y, 0)
                                        cs = insert!(cs, z, 0)
                                        cs = sols[end] + cs
                                    #end
                                    #@time begin
                                        #println("start: $cs")
                                        res = Optim.optimize(costcl, cs, LBFGS(linesearch=LineSearches.BackTracking()),#NelderMead()
                                                    Optim.Options(iterations=5000,
                                                                g_tol=1e-6,
                                                                f_tol=1e-6,
                                                                allow_f_increases=true,
                                                                show_trace=false,
                                                                callback=callback)
                                                        )
                                    #end
                                    #println("iteration $k: min val = $(res.minimum)")    
                                    push!(res_loop_minimum, res.minimum)
                                    push!(res_loop_minimizer, res.minimizer)
                                    if res.minimum <= ε
                                        break
                                    else 
                                        continue
                                    end
                                end
                            end
                            if (lmin <= ε) && (sols != [])
                                println("SKIPPED LOOP!")
                            end
                        end
                        minval,minloc = findmin(res_loop_minimum)
                        minx0 = res_loop_minimizer[minloc]
                        if minval <= ε
                            push!(outarr_x, i)
                            push!(outarr_y, j)
                            push!(outarr_z, l)
                            push!(sols, minx0)
                        end
                    end
                end
            end
        end
        return [outarr_x, outarr_y, outarr_z]
    else 
        return "ERROR: make it x < y < z, stupid"
    end
end
=#