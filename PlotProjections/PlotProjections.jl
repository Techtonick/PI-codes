using PyPlot 
using JSON

data = JSON.parsefile("PlotProjections/plots_table_data.json")
data[1]["x"];
data[1]["y"];

plt.close()

fig, axes = subplots(7, 7, figsize=(14, 14))

# Find global axis limits
xlims = (-0.9, 0.9)
ylims = (-0.9, 0.9)

data_idx = 1

for i in 1:7
    for j in 1:7
        ax = axes[i, j]
        
        if j >= i
            ax.scatter(data[data_idx]["x"], data[data_idx]["y"], s=10, alpha=0.6)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            
            # Ticks on all sides facing inward
            ax.tick_params(direction="in", top=true, right=true, left=true, bottom=true, 
                          labeltop=false, labelbottom=false, labelleft=false, labelright=false)
            
            # Show x labels only on top row
            if i == 1
                ax.tick_params(labeltop=true)
                ax.set_xlabel("$j")
                ax.xaxis.set_label_position("top") 
            end
            
            # Show y labels only on rightmost column
            if j == 7
                ax.tick_params(labelright=true)
                ax.set_ylabel("$i")
                ax.yaxis.set_label_position("right") 
            end
            
            
            data_idx += 1
        else
            ax.axis("off")
        end
    end
end


subplots_adjust(wspace=0, hspace=0)


display(plt.gcf())

plt.savefig("Fig_Projections.pdf",bbox_inches="tight")