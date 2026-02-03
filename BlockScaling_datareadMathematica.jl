using JLD2 
using PyPlot, PlotDefaults

colorvec = ["tab:blue";"tab:orange";"tab:green";"tab:red";"tab:purple";"tab:brown";"tab:pink";"tab:gray";"tab:olive";"tab:cyan";"red"];
fs = 10;

plt.close()
latexwidth_inch = 6.5
sval = 8
fig, axs = plt.subplots(1,2,figsize=(latexwidth_inch,2.2),constrained_layout=true);

### panel (a) 

filedir = "/Users/liambond/Nextcloud/Qudit_PI/ComplexScaling/"
filelist = readdir(filedir)
filelist = sort(filelist; by = s -> parse(Int, match(r"n(\d+)", s).captures[1]))

tcounter = zeros(5);
for i in eachindex(filelist)

    fileloaded = load(filedir*filelist[i])

    nval = fileloaded["Dataset1"][:n]
    tval = fileloaded["Dataset1"][:t]
    val = fileloaded["Dataset1"][:val]

    tcounter[t] += 1 

    if tcounter[t] == 1
        axs[1].scatter(repeat([nval],length(val)),val,s=sval,color=colorvec[tval],label=L"t = "*"$t")
    else 
        axs[1].scatter(repeat([nval],length(val)),val,s=sval,color=colorvec[tval],label="_nolegend_")
    end

end 

### panel (b) 
filedir = "/Users/liambond/Nextcloud/Qudit_PI/Ruskai_scaling/"
filelist = readdir(filedir)
filelist = sort(filelist; by = s -> parse(Int, match(r"n(\d+)", s).captures[1]))

tcounter = zeros(5);
for i in eachindex(filelist)

    fileloaded = load(filedir*filelist[i])

    nval = fileloaded["Dataset1"][:n]
    tval = fileloaded["Dataset1"][:t]
    val = fileloaded["Dataset1"][:val]

    tcounter[t] += 1 

    if tcounter[t] == 1
        axs[2].scatter(repeat([nval],length(val)),val,s=sval,color=colorvec[tval],label=L"t = "*"$t")
    else 
        axs[2].scatter(repeat([nval],length(val)),val,s=sval,color=colorvec[tval],label="_nolegend_")
    end

end 



tvec = [7;19;37;61;91]


[ axs[i].tick_params(direction="in", top=true, right=true, left=true, bottom=true,labelsize=fs-1) for i in 1:2 ] 

# plt.legend(fontsize=fs)
# ytextval = 10^(-33)
ytextval = 1.6*10^(-2)
axs[2].text(8.3,ytextval,L"t=1",fontsize=fs,color=colorvec[1])
axs[2].text(20.8,ytextval,L"t=2",fontsize=fs,color=colorvec[2])
axs[2].text(38,ytextval,L"t=3",fontsize=fs,color=colorvec[3])
axs[2].text(62,ytextval,L"t=4",fontsize=fs,color=colorvec[4])
axs[2].text(80.5,ytextval,L"t=5",fontsize=fs,color=colorvec[5])

axs[1].set_xlabel("Block length "*L"n",fontsize=fs)
axs[2].set_xlabel("Block length "*L"n",fontsize=fs)
axs[1].set_ylabel(L"f_{\rm cost}",fontsize=fs)
[ axs[i].set_yscale("log") for i in 1:2 ] 

tvec = [[7;19;37],[7;19;37;61;91]];
for i in 1:2
    ylims = axs[i].get_ylim()
    [ axs[i].vlines(tvec[i][j],ylims[1],ylims[2],color=colorvec[j],linestyle="dashed") for j in eachindex(tvec[i]) ]
    axs[i].set_ylim(ylims)
end 

display(plt.gcf())
# 
# plt.savefig("Fig_NumericsOptimizationMathematica.pdf")

##
 
