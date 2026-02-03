using PyPlot, PlotDefaults
fs = 10.5; 


tvec_complex = 1:3
nvec_complex = [7;19;37]

tvec_ruskai = 1:5 
nvec_ruskai = [7;19;37;61;91]

tvec_smooth = 1:0.001:5 
nfunc(t) = 3t^2 + 3t + 1
nfunc_AAB(t) = 4t^2 + 2t + 1
nfunc_Ouyang(t) = 4t^2 + 4t + 1

plt.close() 

latexwidth_inch = 7/2
fig, axs = plt.subplots(1,1,figsize=(latexwidth_inch,2.0));

plt.plot(tvec_complex,nvec_complex,color="tab:blue",label="_nolegend_",marker=".",linestyle="",markersize=8)
plt.plot(tvec_ruskai,nvec_ruskai,color="tab:blue",label="_nolegend_",marker="x",linestyle="",markersize=5)
plt.plot(tvec_smooth,nfunc.(tvec_smooth),color="tab:blue")
plt.plot(tvec_smooth,nfunc_AAB.(tvec_smooth),color="tab:orange")
plt.plot(tvec_smooth,nfunc_Ouyang.(tvec_smooth),color="tab:green")

plt.xlabel("Error weight "*L"t", fontsize = fs)
plt.ylabel("Block length "*L"n(t)", fontsize=fs)

plt.legend([L"3t^2 + 3t+1";"AAB";"Ouyang"],fontsize=fs)

plt.tick_params(direction="in", top=true, right=true, left=true, bottom=true)

plt.gcf()

plt.savefig("Fig_QubitScaling.pdf",bbox_inches="tight")