using WaterLily,BiotSavartBCs,Plots,StaticArrays
using OrdinaryDiffEq

# dummy overwrite
import WaterLily: @loop,scale_u!,conv_diff!,udf!,accelerate!,BDIM!,CFL
# Biot-Savart momentum step with U and acceleration prescribed
import BiotSavartBCs: biot_mom_step!,biot_project!
function biot_mom_step!(a::Flow{N},b,ω...;λ=quick,udf=nothing,fmm=true,U,kwargs...) where N
    a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    @log "p"
    conv_diff!(a.f,a.u⁰,a.σ,λ,ν=a.ν)
    udf!(a,udf,t₀; kwargs...)
    BDIM!(a);
    biot_project!(a,b,ω...,U;fmm) # new
    # corrector u → u¹
    @log "c"
    conv_diff!(a.f,a.u,a.σ,λ,ν=a.ν)
    udf!(a,udf,t₁; kwargs...)
    BDIM!(a); scale_u!(a,0.5)
    biot_project!(a,b,ω...,U;fmm,w=0.5) # new
    push!(a.Δt,CFL(a))
end

# falling body acceleration term
fall!(flow,t;acceleration) = for i ∈ 1:ndims(flow.p)
    @loop flow.f[I,i] += acceleration[i] over I ∈ CartesianIndices(flow.p)
end

# equilibrium in body frame of reference
function equilibrium!(du,u,p,t)
    # states, we pass the force through it
    _,vₓ,aₓ,fₓ = u
    # parameters
    m,mₐ,g = p
    # rates
    du[1] = vₓ
    # here we need to fill the solution for the acceleration, and the rate for the velocity
    u[3] = du[2] = (fₓ + m*g - mₐ*aₓ)/(m + mₐ)
end

# set new initial condition and solve a step dt of the ODE problem
function solve_ODE!(problem,u¹,dt)
    SciMLBase.set_u!(problem,u¹)
    OrdinaryDiffEq.step!(problem,dt,true)
    return problem.u
end

# needed for stability of the ODE solver
function run(L=128;U=1,radius=L/2.f0,Re=500,mem=Array)

    # fsi parameters
    ρ = 10.f0                # buoyancy corrected density
    mₐ = π*radius^2          # added-mass coefficient ellipse
    m = ρ*π*radius^2        # mass
    pos = 0.f0; vel = 0.f0;
    acc = 0.f0; forces = 0.f0
    g = -U^2/L

    # make the sim
    body = AutoBody((x,t)->√(sum(abs2,x-SA[L,1.5f0L]))-L/2.f0)
    sim = BiotSimulation((5L,3L), (0,0), L/2.f0; U, ν=U*L/2Re, body, mem)

    # initial conditions for ODE solver
    u₀ = [pos,vel,acc,forces]
    params = (m,mₐ,g)
    equilibrium = init(ODEProblem(equilibrium!,u₀,(0,1000),params),Tsit5(),
                       abstol=1e-6,reltol=1e-6,save_everystep=false)

    @gif for tᵢ in range(0,40.0;step=0.1)
        # update
        while sim_time(sim) < tᵢ
            # compute pressure forces and moment
            forces = -WaterLily.pressure_force(sim)[1] # in lab frame
            # solve the ODE
            u = solve_ODE!(equilibrium,[pos,vel,acc,forces],sim.flow.Δt[end])
            pos,vel,acc = u[1:3]
            # remeasure the sim
            measure!(sim)
            biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                           fmm=sim.fmm,udf=fall!,acceleration=SA[-acc,0.f0],U=SA[-vel,0.0]) # change of frame
        end
        # plot vorticity
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ,shift=(-1.5,-1.5),clims=(-10,10),axis=([],false),
              cfill=:seismic,legend=false,border=:none)
        body_plot!(sim)
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3),
                " F₁=", round(forces,digits=3), " a₁=", round(acc,digits=3))
    end
    return sim
end
using CUDA
sim = run(;mem=CuArray); # run the sim
