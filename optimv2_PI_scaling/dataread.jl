
using JLD2 
using PyPlot, PlotDefaults
using Optim 

colorvec = ["tab:blue";"tab:orange";"tab:green";"tab:red";"tab:purple";"tab:brown";"tab:pink";"tab:gray";"tab:olive";"tab:cyan";"red"];


filedir = "/Users/liambond/Nextcloud/Qudit_PI/KL_general/data/KLcomplex/";

dselector = "d2";
tselector = "t5"; 

folderlist = readdir(filedir)[contains.(readdir(filedir),dselector)]
# folderlist = folderlist[contains.(folderlist,tselector)]

alpha = 1; 

plt.close()
qlist_global = [0]; 
ndone = [0]
stop_counter = 0; 
ntotal = 0; 
for i in eachindex(folderlist)

    folderstring = filedir*folderlist[i]*"/";
    filelist = readdir(folderstring)

    minvallist = []; nlist = []; qlist = []; 
    for j in eachindex(filelist)
        ntotal += 1 
        try 
            fileloaded = load(folderstring*filelist[j])
            # if fileloaded["stoppedby"][:iterations] == false
                push!(minvallist,fileloaded["minval"]) 
                push!(nlist,fileloaded["n"])
                push!(qlist,fileloaded["q"])
            # else 
                stop_counter += 1 
            # end 
        catch 
            @warn "errr loading $(folderstring*filelist[j])"
        end
    end

    # if sum(ndone .== nlist[1]) == 0
        # if sum(qlist_global .== qlist[1]) == 1 
            plt.scatter(nlist,minvallist,color=colorvec[qlist[1]-1],label="_nolegend_", alpha=alpha)
        # else
            # push!(qlist_global,qlist[1])
            # plt.scatter(nlist,minvallist,color=colorvec[qlist[1]-1],label="q=$(qlist[1])", alpha=alpha)
        # end
    # end

    if minimum(minvallist) < 1e-14 
        # @show minimum(minvallist)
        push!(ndone,nlist[1])
    end 


end 

@show stop_counter;
@show ntotal;

# axes = plt.gca() 
# plt.hlines(1e-15,axes.get_xlim()[1],axes.get_xlim()[2],color="black",linestyle="dashed")

if dselector == "d3"
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,1,5,3,4]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
else 
    plt.legend(loc="upper right")
end

plt.xlabel("n")
plt.ylabel("cost")
plt.yscale("log")
display(plt.gcf())