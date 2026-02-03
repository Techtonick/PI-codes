using PyPlot 
using JLD2 
using Optim 
using LinearAlgebra

fs = 10.5; 

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



# ## 

xsols1 = load("/Users/liambond/Documents/PhD/Projects/Project - QuditQEC/PI-codes/data/ParamFixing/SymmetryFix/ParamFixed_xvec[1, 2, 3]_ivec[0, 0, 0]_reps200000.jld2")["xsols"];

# xsols2 = load("/Users/liambond/Documents/PhD/Projects/Project - QuditQEC/PI-codes/data/ParamFixing/SymmetryFix/ParamFixed_xvec[1, 2, 3]_ivec[0, 0, 0]_reps200000_iter2.jld2")["xsols"];

_, usolsM3 = unique_length(xsols1,0.1)

# ##


plt.close() 
usols = unique_length(xsols2,0.01)
[ plt.plot(usols[2][i],marker=".",linestyle="") for i in 1:usols[1] ] 
plt.gcf() 


## 

dirstring = "/Users/liambond/Documents/PhD/Projects/Project - QuditQEC/PI-codes/data/ParamFixing/SymmetryFix/";
filelist = readdir(dirstring)
filelist = filelist[contains.(filelist,"[1, 2, 3, 4]")]

vals = []; vecs = []; 
for i in eachindex(filelist) 
    fileloaded = load(dirstring*filelist[i]);
    sol_point_vec = fileloaded["sol_point_vec"];
    vals = vcat(vals,sol_point_vec[1])
    # push!(vals,sol_point_vec[1])
    # push!(vecs,sol_point_vec[2] )
    if i == 1 
        vecs = sol_point_vec[2]
    else 
        vecs = hcat(vecs,sol_point_vec[2])
    end 
end 
reps=1000*length(filelist);

ϵ_success = 1e-16;

success_indices = (1:reps)[vals .≤ ϵ_success];
sol_point_vec_success = [ vecs[:,i] for i in success_indices ];
successratio = length(sol_point_vec_success)/reps
nsols, xsols, distvec = unique_length(sol_point_vec_success,0.1);

# plt.close() 
# usols = unique_length(xsols,1e-6)
# [ plt.plot(usols[2][i],marker=".",linestyle="") for i in 1:usols[1] ] 
# plt.gcf() 

# plt.close() 
# plt.plot(distvec,marker=".",linestyle="")
# plt.yscale("log")
# plt.gcf()


## 

plt.close() 

latexwidth_inch = 6.5
fig, axs = plt.subplots(1,2,figsize=(latexwidth_inch,2.2),constrained_layout=true);
nparams = length(xsols1[1]);
xvec = 0:(nparams-1);
[ axs[1].plot(xvec,xsols1[i],linestyle="",marker=".",color="tab:blue") for i in eachindex(xsols1) ];

labels = string.(xvec[1:2:end]);
axs[1].set_xticks(xvec[1:2:end])
axs[1].set_xticklabels(labels)

nparams = length(xsols[1])
xvec = 0:(nparams-1)
[ axs[2].plot(xvec,xsols[i],linestyle="",marker=".",color="tab:blue") for i in eachindex(xsols) ];

labels = string.(xvec[1:2:end]);
axs[2].set_xticks(xvec[1:2:end])
axs[2].set_xticklabels(labels)


axs[1].tick_params(direction="in", top=true, right=true, left=true, bottom=true,labelsize=10)
axs[2].tick_params(direction="in", top=true, right=true, left=true, bottom=true,labelsize=10,labelleft=false)

ylims = axs[1].get_ylim()
[ axs[i].set_ylim(ylims) for i in 1:2 ] 

# axs[1].set_xlabel(L"M=2,t=3",fontsize=fs)
# axs[2].set_xlabel(L"M=3,t=4",fontsize=fs)
[ axs[i].set_xlabel(L"x_i",fontsize=fs) for i in 1:2 ] 

axs[1].set_ylabel("Coefficient",fontsize=fs)

axs[1].text(-0.1,0.62,"(a)",fontsize=fs)
axs[2].text(-0.1,0.62,"(b)",fontsize=fs)

display(plt.gcf())

plt.savefig("Fig_ParamFixingSymmetryFix.pdf")