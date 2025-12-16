using WaterLily,StaticArrays,Plots,OrdinaryDiffEq

# data from https://link.springer.com/book/10.1007/978-3-030-85884-1
Astar = hcat([[2.396560724439479, 0.004837941076358709],
              [2.7598158701945277, 0.008500056730078254],
              [3.1230710159495763, 0.012268320663615828],
              [3.486326161704625, 0.012303703423555246],
              [3.878641719120078, 0.034771755985023245],
              [4.227366659044924, 0.0666162399304111],
              [4.605152010630174, 0.1366741046102644],
              [4.953876950555022, 0.26086759199727705],
              [5.331662302140272, 0.6853545629892971],
              [5.7094476537255225, 0.6780303316818579],
              [6.101763211140975, 0.677764960982313],
              [6.43595794523562, 0.6350402783555843],
              [6.813743296820871, 0.643638289020839],
              [7.191528648406122, 0.6433198441813851],
              [7.540253588330968, 0.640772285465754],
              [7.918038939916219, 0.6353587231950382],
              [8.266763879841065, 0.6353587231950382],
              [8.644549231426316, 0.6350402783555843],
              [9.022334583011567, 0.6280344918875989],
              [9.400119934596818, 0.619754926061798],
              [9.748844874521664, 0.591413335350403],
              [10.126630226106915, 0.5738988691804396],
              [10.489885371861963, 0.5578174047880187],
              [10.882200929277415, 0.5337748194092509],
              [11.230925869202263, 0.48696342800953074],
              [11.608711220787512, 0.4328278053023714],
              [11.95743616071236, 0.40098332135698356],
              [12.335221512297611, 0.3086343179153588],
              [12.713006863882862, 0.19080972731742374],
              [13.090792215468111, 0.07330358155894268],
              [13.439517155392958, 0.03795620437956204],
              [13.846362918638613, 0.011153763725527233],
              [14.180557652733258, 0.00929616882871298],
              [14.543812798488306, 0.013754396581067252],
              [14.921598150073558, 0.01184372754434404]]...)
fstar = hcat([[3.882641754839218, 0.8192837443973193],
              [4.231342382316907, 0.8144172028911374],
              [4.609114902868386, 0.8118488576315332],
              [4.987049512827683, 0.8417254088369965],
              [5.33623640852883, 0.9341935567260276],
              [5.743150289043168, 0.9478283633408822],
              [6.091920383409921, 0.9568667774625919],
              [6.469878148998909, 0.99137831387735],
              [6.818682976810194, 1.007369205813001],
              [7.211039684793754, 1.0156061778627166],
              [7.5597731160801676, 1.0173058654030367],
              [7.93749932537227, 1.0054675497248393],
              [8.300840099720451, 1.0226075397837793],
              [8.678543153382869, 1.0061342388962853],
              [9.027313247749621, 1.0151726530179932],
              [9.405120501745634, 1.0195567855723304],
              [9.782869866667426, 1.0123534551034297],
              [10.146131525947505, 1.0136572456972743],
              [10.5095282597341, 1.0419984500120076],
              [10.887312358100425, 1.04174759735705],
              [11.236036141207801, 1.0415160410601665],
              [11.613820239574126, 1.0412651884052089],
              [11.991604337940451, 1.0410143357502513],
              [12.340339698862671, 1.0431002720580143],
              [12.747160956858258, 1.0381951378356895],
              [13.095988940299227, 1.0588210149806336]]...)



function main(D;Vᵣ=4.f0,Mᵣ=4.6,ξ=0.01,U=1,Re=500,T=Float32,mem=Array)
    # fsi parameters
    mₐ = π*(D/2.f0)^2           # added-mass coefficient circle
    m = Mᵣ*mₐ                   # mass as a fraction of the added-mass
    k = (U/(2π*Vᵣ*D))^2*(m+mₐ)  # spring stiffness from reduced velocity
    c = 2ξ*m*√(k/(m+mₐ))        # structural damping
    Tₙ = inv(2π*c/(2ξ*m))
    println("Period of oscillation TₙU/D: ", Tₙ/D, ", Vᵣ: ", Vᵣ)

    # the FSI ODE
    function fsi!(du, u, p, t)
        # unpack parameters and functions, here we can
        # pass a0 and Fy to the ODE solver
        y₁,v₁,a₁,Fy = u
        # unpack parameters, never change
        m,mₐ,k,c = p
        # rates, second-order ODE as first-order system, we need to fill the acceleration
        # in the solution vector to use it in the solver, the rate is d[3:4] = 0.
        du[1] = v₁
        u[3] = du[2] = (Fy - k*y₁ - c*v₁ + mₐ*a₁)/(m + mₐ)
        du[3:4] .= 0
    end

    # initial condition FSI, pull on the spring a bit
    y0,v0,a0,Fy = u₀ = [0.0f0,0.f0,0.f0,0.f0]
    # parameters and time span
    params = (m,mₐ,k,c)
    tspan = (0,100*Tₙ/D)

    # fsi problem and solver
    fsi = init(ODEProblem(fsi!, u₀, tspan, params), Tsit5(),
               reltol=1e-6, abstol=1e-6, save_everystep=false)

    # make a body
    body = AutoBody((x,t)->√sum(abs2,x)-D/2.f0, RigidMap(SA[2.f0D,2.f0D+y0],0.f0))

    # generate sim
    sim = Simulation((6D,4D), (U,0), D; ν=U*D/Re, body, T, mem)
    data = []

    # run for 50 periods
    @time for tᵢ in range(0,50*Tₙ/D;step=0.5)
        # update until time tᵢ in the background
        while sim_time(sim) < tᵢ
            # pressure force
            Fy = -WaterLily.pressure_force(sim)[2]
            # compute motion and acceleration 1DOF
            SciMLBase.set_u!(fsi, [y0,v0,a0,Fy])
            OrdinaryDiffEq.step!(fsi, sim.flow.Δt[end], true)
            y0,v0,a0 = fsi.u[1:3]
            # update the body, pass new position and velocity
            sim.body = setmap(sim.body;x₀=SA[2.f0sim.L,2.f0sim.L+y0],V=SA[0.f0,v0])
            # measure body, update flow
            sim_step!(sim,remeasure=true)
            # store state
            push!(data,[sim_time(sim), y0])
        end
        println("tU/L=",round(tᵢ,digits=4),", y/D=",round(y0/sim.L, digits=2),", Fy=",
                round(2Fy/sim.L, digits=3), ", a=", round(a0, digits=3))
    end
    return Tₙ,data
end

using CUDA,JLD2,FFTW
frequencies = []; amplitudes = []
D = 64 # diameter
# run for the different Vr
for Vᵣ in round.(Astar[1,1:4:end];digits=3)
    period,data = main(D;Vᵣ=Vᵣ,T=Float32,mem=CuArray)
    # extract the displacement time series
    t,yD = getindex.(data,1),getindex.(data,2)./D
    yD,t = yD[t.>25*period/D],t[t.>25*period/D] # keep the last 50%
    # find the VIV amplitude using the last 50% of the data
    push!(amplitudes, sum(sort(abs.(yD))[end-99:end])/100)
    # find the peak oscillationfrequency
    F = rfft(yD); F = F./sum(abs.(F))
    freqs = 2rfftfreq(length(t))
    push!(frequencies, freqs[argmax(abs.(F))]*sim.L/sim.U*period)
end

#     # @save "TwoD_CircleVIV_sim.jld2" data
# @load "TwoD_CircleVIV_sim.jld2"

# # postprocess and plot
# using FFTW
# t,yD = getindex.(data,1),getindex.(data,2)./sim.L
# # p1=plot(t, yD, xlabel="tU/D", ylabel="y/D", label=:none, legend=:topright)
# F = rfft(yD)
# F = F ./ sum(abs.(F))
# freqs = 2rfftfreq(length(t))
p2=plot(freqs*sim.L/sim.U, abs.(F), xlims=(0,1), xlabel="fD/U", ylabel="PSD(y/D)",label=:none)
# # plot(p1,p2, layout=(2,1))
# # savefig("VIV_response.png")

# # find the VIVI frequency and amplitude
# A_max = sum(sort(abs.(yD[t.>25*period]))[end-99:end])/100
# fviv = freqs[argmax(abs.(F))]*sim.L/sim.U/0.25

# period= 4.0
# t,yD = getindex.(data,1),getindex.(data,2)./sim.L
# yD,t = yD[t.>25*period],t[t.>25*period] # keep the last 50%
# # find the VIV amplitude using the last 50% of the data
# A_max = sum(sort(abs.(yD))[end-99:end])/100
# # find the frequency
# F = rfft(yD); F = F./sum(abs.(F))
# freqs = 2rfftfreq(length(t))
# fviv = freqs[argmax(abs.(F))]*sim.L/sim.U*period
# p1=scatter(Astar[1,:], Astar[2,:],xlims=(0,16),ylims=(0,0.8),
#             ylabel="A*",xformatter=_->"",label="Modarres-Sadeghi (2021)",margin=2*Plots.mm)
# scatter!(p1,[4.0],[A],marker=:s,label="WaterLily.jl",title="VIV response of a circular cylinder")
# p2=hline([1.0],ls=:dash,color=:black,lw=0.6,label=:none)
# plot!(p2,[0,16],[0,π],ls=:dot,color=:black,lw=0.6,label="St=0.2")
# scatter!(p2,fstar[1,:], fstar[2,:],color=1,xlims=(0,16),ylims=(0,π),xlabel="Vᵣ",ylabel="f*",label=:none)
# scatter!(p2,[4.0],[fviv],marker=:s,color=2,label="WaterLily.jl",title="VIV frequency of a circular cylinder")
# plot(p1,p2,layout=grid(2,1,heights=(6/8,2/8)))