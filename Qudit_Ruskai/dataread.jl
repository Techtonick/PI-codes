using JLD2 
using Optim
using PyPlot

dir = "/Users/liambond/Nextcloud/Qudit_PI/data/old/";
filelist = readdir(dir)[contains.(readdir(dir),"jld2")]

plt.close() 
for tval in 1:3 

    minvec = []; nvec = []; 
    for i in eachindex(filelist) 
        if contains(filelist[i],"t$(tval)")
            if tval == 1 || tval == 2 ||  (tval == 3 && contains(filelist[i],"_2.jld2"))
                push!(minvec,load(dir*filelist[i])["minimum"])
                push!(nvec,load(dir*filelist[i])["n"])
            end
        end
    end 

    ## 

    
    plt.scatter(nvec,minvec)
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("minimum")
    # plt.title("t = $(tval)")
    (tval == 3 || tval == 2) ? plt.xlim([5;57]) : nothing 
    plt.ylim([10^(-16);1^(-1)])
    plt.legend(["t=1","t=2","t=3"])

end 

plt.gcf()

# ## 

# plt.close()
# plt.plot(1:3,[7;25;46],".-")
# plt.gcf() 

# fscaling(t) = t^3 + 13*t + 10t + 1
# fscaling(1)
# fscaling(2)

# 3+3+1
# 3*2^2+3*2+1

# filelist[contains.(filelist,"t3")]

## 

filelist_secondrun = filelist[contains.(filelist,"_2.jld")]
itervec = []; nvec = []; 
plt.close() 
for i in eachindex(filelist_secondrun) 
        push!(itervec,load(dir*filelist_secondrun[i])["res"].iterations)
        push!(nvec,load(dir*filelist_secondrun[i])["n"])
end 
# plt.scatter(nvec,itervec)
# display(plt.gcf())


@show sum(itervec .== 10000) 
@show length(filelist_secondrun)

# Group itervec by nvec values
unique_n = sort(unique(nvec))
grouped_data = [itervec[nvec .== n] for n in unique_n]

fig, ax = subplots()
parts = ax.violinplot(grouped_data, positions=unique_n, showmeans=false, showmedians=false)
ax.scatter(nvec,itervec,s=20)

ax.set_xlabel("n")
ax.set_ylabel("iterations")
plt.gcf()


##

# plt.close()
# for i in 0:46
#     fileloaded = load(dir*filelist_secondrun[end-i]);
#     iterations = fileloaded["res"].iterations;
#     cache_iteration = fileloaded["cache_iteration"];
#     cache_value = fileloaded["cache_value"];
#     plt.plot(cache_iteration[1:iterations+1],cache_value[1:iterations+1])
# end
# plt.yscale("log")
# plt.xlabel("iterations")
# plt.ylabel("cost function value")
# plt.gcf()

# plot traces vs iterations 
labelflag = zeros(length(bigrange))
plt.close()
for i in eachindex(filelist_secondrun)
        fileloaded = load(dir*filelist_secondrun[i]);
        iterations = fileloaded["res"].iterations;
        cache_iteration = fileloaded["cache_iteration"];
        cache_value = fileloaded["cache_value"];
        nval = fileloaded["n"]
        n_iter = findfirst(nval .== bigrange)
        if labelflag[n_iter] == 0 
            plt.plot(cache_iteration[1:iterations+1],cache_value[1:iterations+1],color=colorvec[n_iter],label="n=$(nval)")
            labelflag[n_iter] = 1 
        else
            plt.plot(cache_iteration[1:iterations+1],cache_value[1:iterations+1],color=colorvec[n_iter])
        end            
end
plt.hlines(10^(-15),0,10000,color="black",ls="dashed")
plt.hlines(10^(-16),0,10000,color="black",ls="dashed")
plt.yscale("log")
plt.xlabel("iterations")
plt.ylabel("cost function value")
plt.legend()
display(plt.gcf())