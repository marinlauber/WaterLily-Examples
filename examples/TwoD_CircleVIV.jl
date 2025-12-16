using WaterLily,StaticArrays,Plots

# required to keep things global
let
    # parameters
    Re = 250; U = 1
    p = 5; L = 2^p
    radius, center = L/2, 2L

<<<<<<< Updated upstream
=======
function main(D;Vᵣ=4.f0,Mᵣ=4.6,ξ=0.01,U=1,Re=250,T=Float32,mem=Array)
>>>>>>> Stashed changes
    # fsi parameters
    T = 4*radius    # VIV period
    mₐ = π*radius^2 # added-mass coefficent circle
    m = 0.1*mₐ      # mass as a fraction of the added-mass, can be zero
    k = (2*pi/T)^2*(m+mₐ)

    # initial condition FSI
    p0=radius/3; v0=0; a0=0; t0=0

    mutable struct Mapping <: Function
        x :: Float64
        t :: Float64
        vx :: Float64
        function Mapping(x)
            new(x, 0, 0)
        end
    end
    function (l::Mapping)(x,t)# this allows the structure to be called like: Mapping(x,t)
        return x - SA[l.x + (t-l.t)*l.vx, 0]
    end

<<<<<<< Updated upstream
    # motion function uses global var to adjust
    # posx(t) = p0 + (t-t0)*v0
=======
    # initial condition FSI, pull on the spring a bit
    y0,v0,a0,Fy = u₀ = [0.f0D,0.f0,0.f0,0.f0]
    # parameters and time span
    params = (m,mₐ,k,c)
    tspan = (0,1000)
>>>>>>> Stashed changes

    # motion definition
    # map(x,t) = x - SA[posx(t), 0]

    # make a body
<<<<<<< Updated upstream
    circle = AutoBody((x,t)->√sum(abs2, x .- center) - radius, Mapping(p0))

    # generate sim
    sim = Simulation((6L,4L), (U,0), radius; ν=U*radius/Re, body=circle)

    # get start time
    duration=10; step=0.1; t₀=round(sim_time(sim))

    @time @gif for tᵢ in range(t₀,t₀+duration;step)
=======
    body = AutoBody((x,t)->√sum(abs2,x)-D/2.f0, RigidMap(SA[5.f0D,5.f0D+y0],0.f0))

    # generate sim
    sim = Simulation((20D,10D), (U,0), D; ν=U*D/Re, body, T, mem)

    # get start time
    duration=100.0; step=0.1
    data = []
>>>>>>> Stashed changes

        # update until time tᵢ in the background
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U

            # measure body
            measure!(sim,t)

            # update flow
            mom_step!(sim.flow,sim.pois)

            # pressure force
            force = -WaterLily.pressure_force(sim)

            # compute motion and acceleration 1DOF
<<<<<<< Updated upstream
            Δt = sim.flow.Δt[end]
            accel = (force[1] - k*p0 + mₐ*a0)/(m + mₐ)
            p0 += Δt*(v0+Δt*accel/2.)
            v0 += Δt*accel
            a0 = accel

            # update time, sets the pos/v0 correctly
            t0 = t; t += Δt
=======
            SciMLBase.set_u!(fsi, [y0,v0,a0,Fy])
            OrdinaryDiffEq.step!(fsi, sim.flow.Δt[end], true)
            y0,v0,a0 = fsi.u[1:3]
            # update the body, pass new position and velocity
            sim.body = setmap(sim.body;x₀=SA[5.f0sim.L,5.f0sim.L+y0],V=SA[0.f0,v0])
            # measure body, update flow
            sim_step!(sim,remeasure=true)
            # store state
            push!(data,[sim_time(sim), y0, v0, a0, Fy])
>>>>>>> Stashed changes
        end

        # plot vorticity
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ; shift=(-0.5,-0.5),clims=(-5,5))
        body_plot!(sim); plot!(title="tU/L $tᵢ")

        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    end
<<<<<<< Updated upstream
end
=======
    return sim,data
end

# run
using CUDA
sim,data = main(64;Vᵣ=2.0,mem=CuArray);

# postprocess and plot
using FFTW
p1=plot(getindex.(data,1), getindex.(data,2)./sim.L,
        xlabel="tU/D", ylabel="y/D", label=:none, legend=:topright)
F = rfft(getindex.(data,2))
F = F ./ sum(abs.(F))
freqs = 2rfftfreq(length(getindex.(data,1)), sim.L)
p2=plot(freqs, abs.(F), xlims=(0,1), xlabel="fD/U", ylabel="PSD(y/D)",label=:none)
plot(p1,p2, layout=(2,1))
savefig("VIV_response.png")
>>>>>>> Stashed changes
