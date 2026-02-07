using PyPlot, PlotDefaults

colorvec = ["tab:blue";"tab:orange";"tab:green";"tab:red";"tab:purple";"tab:brown";"tab:pink";"tab:gray";"tab:olive";"tab:cyan";"red"];
fs = 10.5; 

qvec_d2 = 2:9
nvec_d2 = [7;7;6;6;6;6;6;6]

qvec_d3 = 2:6
nvec_d3 = [14;9;8;7;7]

qvec_d4 = 2:4
nvec_d4 = [19;11;9]

plt.close() 

latexwidth_inch = 7
fig, axs = plt.subplots(1,3,figsize=(latexwidth_inch,2.0));

# KL solving 
axs[1].plot(qvec_d2,nvec_d2,".-",color=colorvec[1])
axs[2].plot(qvec_d3,nvec_d3,".-",color=colorvec[2])
axs[3].plot(qvec_d4,nvec_d4,".-",color=colorvec[3])

# ouyang scaling 
axs[1].plot(qvec_d2,9*ones(length(qvec_d2)),"--",color=colorvec[1])
axs[2].plot(qvec_d2,18*ones(length(qvec_d2)),"--",color=colorvec[2])
axs[3].plot(qvec_d2,27*ones(length(qvec_d2)),"--",color=colorvec[3])

# singleton bound 
[ axs[i].hlines(5,2,9,color="black",ls="dotted") for i in 1:3 ] 

[ axs[i].set_xlabel("Physical dimension "*L"{\rm d}_{\rm P}",fontsize=fs) for i in 1:3 ] 

axs[1].set_ylim([4,27])
axs[2].set_ylim([4,27])
axs[2].set_ylim([4,27])

axs[1].set_ylabel("Block length "*L"n",fontsize=fs)

[ axs[i].tick_params(direction="in", top=true, right=true, left=true, bottom=true, labelsize=fs-1) for i in 1:3 ] 

[ axs[i].set_yticks(5:4:30) for i in 1:3 ] 

[ axs[i].tick_params(labelleft=false) for i in 2:3 ] 
subplots_adjust(wspace=0.05, hspace=0)

axs[1].text(2,26,"(a)",fontsize=fs)
axs[2].text(2,26,"(b)",fontsize=fs)
axs[3].text(2,24,"(c)",fontsize=fs)

# plt.legend([L"d=2",L"d=3"])

display(plt.gcf())

plt.savefig("Fig_QuditScaling.pdf",bbox_inches="tight")