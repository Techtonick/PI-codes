using JLD2 
using PyPlot, PlotDefaults
using Optim 

PyPlot.matplotlib["rcParams"]["text.usetex"] = false

colorvec = ["tab:blue";"tab:orange";"tab:green";"tab:red";"tab:purple";"tab:brown";"tab:pink";"tab:gray";"tab:olive";"tab:cyan";"red"];
fs = 10;

plt.close()
latexwidth_inch = 6.5
sval = 6
fig, axs = plt.subplots(1,2,figsize=(latexwidth_inch,2.2),constrained_layout=true);

### panel (a) 

filedir = "/Users/liambond/Nextcloud/Qudit_PI/KL_general/data/KLcomplex/";
# filedir = "/Users/liambond/Nextcloud/Qudit_PI/KL_general/data/BFGS_complex/";

dselector = "d2";

folderlist = readdir(filedir)[contains.(readdir(filedir),dselector)]
folderlist = sort(folderlist; by = s -> parse(Int, match(r"n(\d+)", s).captures[1]))
tlist = [ parse(Int, match(r"t(\d+)", folderlist[i]).captures[1]) for i in eachindex(folderlist) ] 

for outer_i in eachindex(folderlist)

    @show outer_i 

    folderstring = filedir*folderlist[outer_i]*"/";
    filelist = readdir(folderstring)

    tcounter = zeros(5);

    minvallist = []; nlist = []; qlist = []; 
    for j in eachindex(filelist)
        
        try 
            fileloaded = load(folderstring*filelist[j])
            if fileloaded["stoppedby"][:iterations] == false
                push!(minvallist,fileloaded["minval"]) 
                push!(nlist,fileloaded["n"])
                push!(qlist,fileloaded["q"])
            else 
                stop_counter += 1 
            end 
        catch 
            @warn "errr loading $(folderstring*filelist[j])"
        end
    end

    # FIXME: 
    if nlist[1] < 40 
        # axs[1].scatter(nlist,minvallist,color=colorvec[tlist[outer_i]],label="_nolegend_",s=sval)
        axs[1].plot(nlist, minvallist, linestyle="", marker=".", markersize=sval, color=colorvec[tlist[outer_i]])
    end

end 

### panel (b) 

filedir = "/Users/liambond/Nextcloud/Qudit_PI/KL_general/data/Ruskai_FloatAutoDiff/";
dselector = "d2";

folderlist = readdir(filedir)[contains.(readdir(filedir),dselector)];
folderlist = sort(folderlist; by = s -> parse(Int, match(r"n(\d+)", s).captures[1]));
tlist = [ parse(Int, match(r"t(\d+)", folderlist[i]).captures[1]) for i in eachindex(folderlist) ] ;

ncounterlist = []; ncounter = []; 
for outer_i in eachindex(folderlist)

    mod(outer_i,100) == 0 ? println(outer_i) : nothing 

    folderstring = filedir*folderlist[outer_i]*"/";
    filelist = readdir(folderstring)

    minvallist = []; nlist = []; qlist = []; 
    for j in eachindex(filelist)
        
        # try 
            fileloaded = load(folderstring*filelist[j])
            if fileloaded["stoppedby"][:iterations] == false
                push!(minvallist,fileloaded["minval"]) 
                push!(nlist,fileloaded["n"])
                push!(qlist,fileloaded["q"])
            else 
                stop_counter += 1 
            end 
            # (fileloaded["t"] == 5 && minimum(minvallist) < 1e-15) ? println(folderstring*filelist[j]) : nothing
        # catch 
            # @warn "errr loading $(folderstring*filelist[j])"
        # end
    end

    if isempty(nlist) 
    else 
        if isempty(findall(x -> in(nlist[1],x), ncounterlist))
            push!(ncounterlist,nlist[1])
            push!(ncounter,length(nlist))
        else
            loc = findall(x -> in(nlist[1],x), ncounterlist)[1]
            @show loc 
            @show ncounter
            ncounter[loc] += length(nlist)
        end 
    end 


    # @show nlist[1] 
    # @show length(nlist)


    # axs[2].scatter(nlist,minvallist,color=colorvec[tlist[outer_i]],label="_nolegend_",s=sval)
    axs[2].plot(nlist, minvallist, linestyle="", marker=".", markersize=sval, color=colorvec[tlist[outer_i]])

end 

# ### ADD MATHEMATICA POINTS

# ### panel (b) 
# filedir = "/Users/liambond/Nextcloud/Qudit_PI/Ruskai_scaling/"
# filelist = readdir(filedir)
# filelist = sort(filelist; by = s -> parse(Int, match(r"n(\d+)", s).captures[1]))

# tcounter = zeros(5);
# for i in eachindex(filelist)

#     fileloaded = load(filedir*filelist[i])

#     nval = fileloaded["Dataset1"][:n]
#     tval = fileloaded["Dataset1"][:t]
#     val = fileloaded["Dataset1"][:val]

#     if (nval[1] == 7) || (nval[1] == 19) || (nval[1] == 37) || (nval[1] == 61) || (nval[1] == 91)
#         minval = minimum(val) 
#         axs[2].plot(nval[1],minval,linestyle="",marker="x",color=colorvec[tval[1]],markersize=sval-1)
#     end 

# end 




[ axs[i].tick_params(direction="in", top=true, right=true, left=true, bottom=true,labelsize=fs-1) for i in 1:2 ] 

# ytextval = 10^(-33)

## 
xlims = axs[1].get_xlim()
axs[1].set_xlim(1,xlims[2])
axs[1].text(2.3,4*10^(-3),"(a)",fontsize=fs,color="black")

xlims = axs[1].get_xlim()
axs[1].hlines(1e-22,xlims[1],xlims[2],color="black",linestyle="dashed",linewidth=0.7)

xlims = axs[2].get_xlim()
axs[2].set_xlim(-4,xlims[2])
axs[2].text(-2.1,8*10^(-4),"(b)",fontsize=fs,color="black")

xlims = axs[2].get_xlim()
axs[2].hlines(1e-22,xlims[1],xlims[2],color="black",linestyle="dashed",linewidth=0.7)

ytextval = 1.2*10^(-3)



axs[1].text(7.5,ytextval,L"t=1",fontsize=fs,color=colorvec[1])
axs[1].text(19.8,ytextval,L"t=2",fontsize=fs,color=colorvec[2])
axs[1].text(32.4,ytextval,L"t=3",fontsize=fs,color=colorvec[3])

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
plt.savefig("Fig_OptimizationNumericsJulia.pdf")