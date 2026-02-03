using PyPlot, PlotDefaults
using JLD2 

fs = 10.5; 

# plt.close()

filestring = "/Users/liambond/Nextcloud/Qudit_PI/KL_general/data/n7_q2_d2_t1/iter5.jld2";
load_minx0 = load(filestring)["minx0"];
@show load(filestring)["minval"]

# # Data
x = 0:7
y1 = load_minx0[1:8]
y2 = load_minx0[9:16]

# # Per-bar colours
colors1 = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
colors2 = reverse(colors1)


plt.close() 

latexwidth_inch = 7/2
fig, axs = plt.subplots(1,1,figsize=(latexwidth_inch,2.0));

# First bar set
axs.bar(x, y1, color=colors1, alpha=0.5)
axs.bar(x, y2, edgecolor=colors2,fill=0, linewidth=2)

axs.axhline(0, color="black", linewidth=0.8)

# labels = ["[7,0]", "[1,6]", "[2,5]", "[4,3]", "[3,4]", "[2,5]", "[1,6]", "[0,7]"];
labels = string.(0:7);
xticks(x,labels)
axs.tick_params(direction="in", top=true, right=true, left=true, bottom=true,labelsize=10)

# xlabel("Codeword composition", fontsize=fs)
xlabel(L"\alpha_i,\beta_i",fontsize=fs)
ylabel("Coefficient",fontsize=fs)


display(plt.gcf())

plt.savefig("Fig_PhaseFlipSymmetry.pdf",bbox_inches="tight")








## 

# fig, axs = subplots(2, 1, sharex=true, figsize=(6, 4))

# # Top bar chart
# axs[1].bar(x, y1, color=colors1, width=0.6)
# axs[1].axhline(0, color="black", linewidth=0.8)
# axs[1].set_ylabel(L"|0_L\rangle")

# # Bottom bar chart
# axs[2].bar(x, y2, color=colors2, width=0.6)
# axs[2].axhline(0, color="black", linewidth=0.8)
# axs[2].set_ylabel(L"|1_L\rangle")
# axs[2].set_xlabel("Partition label")

# # Turn off boxes (spines) for both subplots
# for ax in axs
#     # ax.spines["top"].set_visible(false)
#     # ax.spines["right"].set_visible(false)
#     # ax.spines["left"].set_visible(false)
#     # ax.spines["bottom"].set_visible(false)
#     ax.tick_params(left=true, bottom=true, direction="in")
# end

# axs[1].set_ylim([-0.63;0.63])
# axs[2].set_ylim([-0.63;0.63])

# labels = ["[7,0]", "[1,6]", "[2,5]", "[4,3]", "[3,4]", "[2,5]", "[1,6]", "[0,7]"];

# axs[2].set_xticks(x)
# axs[2].set_xticklabels(labels)

# fig.subplots_adjust(hspace=0.05)

# display(plt.gcf())
