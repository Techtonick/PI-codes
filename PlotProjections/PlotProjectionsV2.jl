using PyPlot, PlotDefaults
using JLD2 
using JSON 

jsondata = JSON.parsefile("PlotProjections/plots_table_data.json");
[ jsondata[i]["name"] for i in 29:56 ] 
aydinpoints = [ (data[i]["x"][1], data[i]["y"][1]) for i in 29:56 ]  
aydincoords = [ parse.(Int, match(r"\[(\d+),\s*(\d+)\]", jsondata[i]["name"]).captures) for i in 29:56 ] 



function pop_zeros(xvals,yvals) 
    x = deepcopy(xvals)
    y = deepcopy(yvals)
    xlength = length(x) 

    i = 1 

    while i ≤ length(x) 

        if x[i] == 0 || y[i] == 0 
            popat!(x,i)
            popat!(y,i)
        else 
            i += 1 
        end 

    end 

    return x,y 

end;

# we did two independent runs. first we check that the points where solutions were found are identical between the 
# two runs. 

plottol = 1e-15

dir_A = "/Users/liambond/Documents/PhD/Projects/Project - QuditQEC/PI-codes/data/ParamFixing/n7_q2_d2_t1_iter1_itstep0.02/";
files_A = readdir(dir_A)[contains.(readdir(dir_A),"jld2")]

dir_B = "/Users/liambond/Documents/PhD/Projects/Project - QuditQEC/PI-codes/data/ParamFixing/n7_q2_d2_t1_iter11_itstep0.02/";
files_B = readdir(dir_B)[contains.(readdir(dir_B),"jld2")]


for i in eachindex(files_A)

    solnvals_A = load(dir_A*files_A[i])["solpts"][4];
    xvals_A = load(dir_A*files_A[i])["solpts"][1][solnvals_A .≤ plottol]; 
    yvals_A = load(dir_A*files_A[i])["solpts"][2][solnvals_A .≤ plottol]; 

    solnvals_B = load(dir_B*files_B[i])["solpts"][4];
    xvals_B = load(dir_B*files_B[i])["solpts"][1][solnvals_B .≤ plottol]; 
    yvals_B = load(dir_B*files_B[i])["solpts"][2][solnvals_B .≤ plottol]; 

    @show xvals_A == xvals_B 

end 

# they are identical, so we can just proceed with using the "A" (or "B") files.



## 

dir = "/Users/liambond/Documents/PhD/Projects/Project - QuditQEC/PI-codes/data/ParamFixing/n7_q2_d2_t1_iter11_itstep0.02/";
files = readdir(dir)[contains.(readdir(dir),"jld2")];

plottol = 1e-15; 
latexwidth_inch = 7
fs = 10.5; 

plt.close() 
fig, axes = subplots(7, 7, figsize=(latexwidth_inch, latexwidth_inch), constrained_layout=true)

# global axis limits
xlims = (-0.95, 0.95)
ylims = (-0.95, 0.95)

# projcolor
projcolor = "tab:blue";
projalpha = 1.0; 
projs = 1.0;

for i in eachindex(files) 

    xa = load(dir*files[i])["xa"]; 
    ya = load(dir*files[i])["ya"]; 
    solnvals = load(dir*files[i])["solpts"][4];
    xvals = load(dir*files[i])["solpts"][1][solnvals .≤ plottol]; 
    yvals = load(dir*files[i])["solpts"][2][solnvals .≤ plottol]; 

    # ax = axes[xa, ya-1]
    i = ya-1 
    j = xa 
    ax = axes[ya-1,xa]

    ax.plot(xvals,yvals,color=projcolor,alpha=projalpha,marker=".",linestyle="",markersize=projs,markeredgecolor="none")

    xnonzero, ynonzero = pop_zeros(xvals,yvals) 

    ax.plot(xnonzero,-ynonzero,color=projcolor,alpha=projalpha,marker=".",linestyle="",markersize=projs,markeredgecolor="none")
    ax.plot(-xnonzero,ynonzero,color=projcolor,alpha=projalpha,marker=".",linestyle="",markersize=projs,markeredgecolor="none")
    ax.plot(-xvals,-yvals,color=projcolor,alpha=projalpha,marker=".",linestyle="",markersize=projs,markeredgecolor="none")

    aydiniter = findfirst(x->x==[xa,ya],aydincoords)
    ax.plot(aydinpoints[aydiniter][1],aydinpoints[aydiniter][2],marker="*",linestyle="",markersize=3projs,color="red")


    # set axis limits 
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    # Ticks on all sides facing inward
    ax.tick_params(direction="in", top=true, right=true, left=true, bottom=true, 
                    labeltop=false, labelbottom=false, labelleft=false, labelright=false,
                    labelsize=fs-1)


    # Show x labels only on bottom row
    if i == 7
        ax.tick_params(labelbottom=true)
        ax.set_xlabel("$(xa-1)",fontsize=fs)
        ax.xaxis.set_label_position("bottom") 
    end

    # Show y labels only on leftmost column
    if j == 1
        ax.tick_params(labelleft=true)
        ax.set_ylabel("$(ya-1)",fontsize=fs)
        ax.yaxis.set_label_position("left") 
    end


end 

[ j>i ? axes[i,j].axis("off") : nothing for i in 1:7, j in 1:7]

subplots_adjust(wspace=0, hspace=0)

fig.supxlabel("Coefficient fixed",fontsize=11)
fig.supylabel("Coefficient fixed",fontsize=11)

display(plt.gcf())

plt.savefig("Fig_ProjectionsV2.pdf")