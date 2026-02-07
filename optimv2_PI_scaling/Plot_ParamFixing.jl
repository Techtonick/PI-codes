using JLD2 
using PyPlot, PlotDefaults

fs = 10.5 

@load "/Users/liambond/Documents/PhD/Projects/Project - QuditQEC/PI-codes/data/ParamFixing/ParamFixed_x1_y2_i0_j0.jld2"

sol_point_vec_success = sol_point_vec[2][sol_point_vec[1] .≤ ϵ_success];

## 

plt.close() 

latexwidth_inch = 6.5
fig, axs = plt.subplots(1,2,figsize=(latexwidth_inch,2.2),constrained_layout=true);

xvec = collect(0:7); 
[ axs[1].plot(xvec,sol_point_vec_success[i][1:8],linestyle="",marker=".",color="tab:blue") for i in eachindex(sol_point_vec_success) ];
[ axs[2].plot(xvec,sol_point_vec_success[i][9:16],linestyle="",marker=".",color="tab:blue") for i in eachindex(sol_point_vec_success) ];

labels = string.(xvec);
[ axs[i].set_xticks(xvec) for i in 1:2 ] 
[ axs[i].set_xticklabels(labels) for i in 1:2 ] 

axs[1].tick_params(direction="in", top=true, right=true, left=true, bottom=true,labelsize=10)
axs[2].tick_params(direction="in", top=true, right=true, left=true, bottom=true,labelsize=10,labelleft=false)

ylims = axs[1].get_ylim()
[ axs[i].set_ylim(ylims) for i in 1:2 ] 

# axs[1].set_xlabel(L"|c_0\rangle"*" composition",fontsize=fs)
# axs[2].set_xlabel(L"|c_1\rangle"*" composition",fontsize=fs)
axs[1].set_xlabel(L"\alpha_i",fontsize=fs)
axs[2].set_xlabel(L"\beta_i",fontsize=fs)

axs[1].set_ylabel("Coefficient",fontsize=fs)

axs[1].text(0.1,0.7,"(a)",fontsize=fs)
axs[2].text(0.1,0.7,"(b)",fontsize=fs)

display(plt.gcf())

plt.savefig("Fig_ParamFixing.pdf")