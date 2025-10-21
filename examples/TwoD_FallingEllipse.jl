using WaterLily,BiotSavartBCs,Plots,CUDA,StaticArrays
using ForwardDiff

# new body type
struct RigidBody{D,T,F<:Function,A<:AbstractVector} <: AbstractBody
    sdf :: F
    center :: A
    velocity :: A
    rot :: T
    ω :: T
    function RigidBody(sdf,center,velocity,rot,ω)
        T,D = eltype(center),length(center)
        new{D,T,typeof(sdf),typeof(center)}(sdf,center,velocity,T(rot),T(ω))
    end
end
function WaterLily.sdf(body::RigidBody{2,T},x,t=0;kwargs...) where T
    R = SA{T}[cos(body.rot) sin(body.rot); -sin(body.rot) cos(body.rot)]
    return body.sdf(R*(x.-body.center),t)
end
function WaterLily.measure(body::RigidBody{2,T},x,t;fastd²=Inf) where T
    # eval d=f(x,t), and n̂ = ∇f
    d = WaterLily.sdf(body,x,t)
    d^2>fastd² && return (d,zero(x),zero(x)) # skip n,V
    n = ForwardDiff.gradient(x->WaterLily.sdf(body,x,t), x)
    any(isnan.(n)) && return (d,zero(x),zero(x))

    # correct general implicit fnc f(x₀)=0 to be a pseudo-sdf
    #   f(x) = f(x₀)+d|∇f|+O(d²) ∴  d ≈ f(x)/|∇f|
    m = √sum(abs2,n); d /= m; n /= m

    # The rigid body velocity is given by the rigid body motion
    # v = v + ω×(x-c)
    v = body.velocity + body.ω*SA[-(x[2]-body.center[2]),(x[1]-body.center[1])]
    return (d,n,v)
end
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

# define an ellipse
@inline @fastmath ellipse(x,t;radius=1,Λ=1) = √sum(abs2, SA[x[1]/Λ,x[2]])-radius/Λ

# make the simulation
function circle(L;center,α0=0.f0,Re=500,U=1,Λ=4.f0,mem=Array)
    # body = AutoBody((x,t)->√sum(abs2, x .- (2L,L)) - L/2)
    vel,ω = SA[0.f0,0.f0],0.f0
    body = RigidBody((x,t)->ellipse(x,t;radius=L/2,Λ=Λ),center,vel,α0,ω)
    BiotSimulation((4L,4L), (0,0), L/2.f0; U, ν=U*L/2Re, body, mem)
end

import WaterLily: @loop
# falling body acceleration term
fall!(flow,t;acceleration) = for i ∈ 1:ndims(flow.p)
    @loop flow.f[I,i] += acceleration[i] over I ∈ CartesianIndices(flow.p)
end

#helper to rotate a vector
@inline @fastmath rotate(v,θ) = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]*v
# needed for stability of the ODE solver
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=0.1)

function run(L=128,U=1,Λ=4.f0,radius=L/2.f0)

    # fsi parameters
    ρ = 10.f0 # buoyancy corrected density
    mₐ = SA[π*radius^2,π*radius^2/Λ^2] # added-mass coefficient ellipse
    m = ρ*π*radius^2/Λ # mass
    vel = SA[0.f0,0.f0]
    a0 = SA[0.f0,0.f0]
    pos = SA[0.f0,0.f0]
    g = SA[0.f0,-U^2/L]

    # rotation variables
    Im = 0.25f0*m*(radius^2+radius^2/Λ^2)
    Iₐ = 0.125f0*π*(radius^2-radius^2/Λ^2)^2 # added mass ellipse m₆₆
    dω = 0.0f0; dω₀=0.0f0; ω = 0.0f0; rot =0.0f0
    α = 0.05f0

    # make the sim
    center,force = SA[2.f0L,L],SA[0.f0,0.f0]
    sim = circle(L;center=center,α0=α,mem=Array)
    store=[]

    @gif for tᵢ in range(0,40.0;step=0.1)
        # update
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U
            Δt = sim.flow.Δt[end]
            # remeasure the sim
            measure!(sim)
            biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                           fmm=sim.fmm,udf=fall!,acceleration=-a0,U=-vel) # change of frame

            # compute pressure force and moment
            force = -WaterLily.pressure_force(sim) # in lab frame
            moment = WaterLily.pressure_moment(center,sim)[1]
            force = rotate(force,α+rot) # transform to body frame
            a0 = rotate(a0,α+rot)

            # update force in body frame
            accel = (force + m.*g - mₐ.*a0)./(m .+ mₐ)
            accel = rotate(accel,-(α+rot)) # back to lab frame
            pos = pos + Δt.*(vel+Δt.*accel./2.)
            vel = vel + Δt.*accel
            a0 = copy(accel)
            dω = (moment - dω₀*Iₐ)/(Im+Iₐ)
            rot = rot + Δt.*(ω+Δt*dω/2.)
            ω = ω + Δt*dω
            dω₀ = copy(dω)

            # update the body
            sim.sim.body = RigidBody((x,t)->ellipse(x,t;radius=L/2,Λ=Λ),center,SA[0.f0,0.f0],α+rot,ω)

            # save position, velocity, etc
            push!(store,[Δt,force...,accel...,pos...,vel...,α+rot])

            # update time, must be done globally to set the pos/vel correctly
            t_init = t; t += Δt
        end
        # plot vorticity
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        # @inside sim.flow.σ[I] = ifelse(sdf(sim.body,loc(0,I),WaterLily.time(sim))<0,NaN,sim.flow.σ[I])
        flood(sim.flow.σ|>Array,shift=(-1.5,-1.5),clims=(-5,5), axis=([], false),
                cfill=:seismic,legend=false,border=:none,size=(1080,1080))
        body_plot!(sim)
        plot!([center[1]],[center[2]],marker=:o,color=:red,legend=:none)
        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3),
                " F₂=", round(force[2],digits=3), " a₂=", round(a0[2],digits=3))
    end
    return sim,store
end

sim,data = run(); # run the sim

let # post-process
    store_matrix = reduce(vcat,data')
    p1 = plot(store_matrix[:,6],store_matrix[:,7],linez=cumsum(store_matrix[:,1]),
                colorbar=:true, aspect_ratio=:equal,label=:none, lw=3, xlabel="X", ylabel="Y",)
    p2 = plot(cumsum(store_matrix[:,1]),[store_matrix[:,8],store_matrix[:,9]],
                colorbar=:false, label=["Vx" "Vy"])
    p3 = plot(cumsum(store_matrix[:,1]),store_matrix[:,10],colorbar=:false, label="rotation")
    plot(p1, p2, p3, layout= @layout[a{0.5w} [grid(2,1)]])
end