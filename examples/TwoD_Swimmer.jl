using WaterLily,StaticArrays,Plots

function make_sim(;L=128, U=1, Re=10_000, St=0.5, T=Float32, mem=Array)

    # Motion parameters
    A = T(0.4*L)
    f = T(2π*St*U/L)
    Wt = Wh = T(19/800*L)
    Sb = T(18/800*L)
    St = T(580/800*L)
    @inline E(x) = 0.05f0-0.13f0*x + 0.28f0*x^2.f0
    function sdf(x,t)
        xc = clamp(x[1],0.f0,L)
        # t₁ = √((x[1]-xc)^2.f0 + (x[2]+A*E(xc/L)*sin(f*xc-f*t))^2.f0) # mapping does not induce velocity
        t₁ = √((x[1]-xc)^2.f0 + (x[2])^2.f0)
        return (xc < Sb) ? t₁ - √(2.f0*Wh*xc-xc^2.f0) :
                           ((Sb <= xc < St) ? t₁ - Wh - (Wh-Wt)*((xc-Sb)/(St-Sb))^2.f0 :
                                              t₁ - Wt*(L-xc)/(L-St))
    end
    function map(x,t)
        x = x .- L
        xc = clamp(x[1],0.f0,L)
        return SA[x[1], x[2]+A*E(xc/L)*sin(f*xc-f*t)]
    end

    # add a body
    body = AutoBody(sdf,map)

    # make the sim
    Simulation((4L,2L), (U,0), L; U, ν=U*L/Re, body, T=T, mem=mem, exitBC=true)
end
using CUDA
# generate
sim = make_sim(;L=128) #,mem=CuArray)

# run the sim
@time @gif for tᵢ in range(0.,10.0;step=0.1)

    # update the flow
    sim_step!(sim,tᵢ;remeasure=true)

    # plot the flow and geometry
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I])<0.001,0.0,sim.flow.σ[I])
    flood(sim.flow.σ,shift=(-2,-1.5),clims=(-16,16), axis=([], false),
          cfill=:seismic,legend=false,border=:none,size=(6*sim.L,6*sim.L))
    body_plot!(sim)

    # compute the forces
    F = WaterLily.pressure_force(sim)/sim.L

    println("tU/L=",round(tᵢ,digits=4), " Forces=",round.(F,digits=3))
end
