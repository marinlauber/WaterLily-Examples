using WaterLily,Plots,StaticArrays,BiotSavartBCs,ForcePartition

"""
    Solves the potential in an unbounded domain using the Biot-Savart and FMℓM methods.
"""
function potential_unbounded!(FPM::ForcePartitionMethod{A,T},body;x₀=0,axis=nothing,tᵢ=0) where {A,T}
    @inside FPM.σ[I] = FPM.pois.x[I]
    # generate source term
    isnothing(axis) && (axis=A) # if we provide an axis, we use it
    @inside FPM.pois.z[I] = source(body,loc(0,I,T),x₀,axis,tᵢ)
    # solver for potential
    solver!(FPM.pois); pop!(FPM.pois.n) # keep the tol the same as the pressure and don't write the iterations
    # copy to the FPM
    @inside FPM.ϕ[I] = FPM.pois.x[I]
    @inside FPM.pois.x[I] = FPM.σ[I] # put back pressure field
end


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

# #  momentum step with U and acceleration prescribed
import WaterLily: mom_step!,scale_u!,conv_diff!,udf!,accelerate!,BDIM!,CFL,project!
# @fastmath function mom_step!(a::Flow{N},b::AbstractPoisson;λ=quick,udf=nothing,U,kwargs...) where N
#     a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
#     # predictor u → u'
#     @log "p"
#     conv_diff!(a.f,a.u⁰,a.σ,λ;ν=a.ν,perdir=a.perdir)
#     udf!(a,udf,t₀; kwargs...)
#     BDIM!(a); BC!(a.u,U,a.exitBC,a.perdir,t₁) # BC MUST be at t₁
#     project!(a,b); BC!(a.u,U,a.exitBC,a.perdir,t₁)
#     # corrector u → u¹
#     @log "c"
#     conv_diff!(a.f,a.u,a.σ,λ;ν=a.ν,perdir=a.perdir)
#     udf!(a,udf,t₁; kwargs...)
#     BDIM!(a); scale_u!(a,0.5); BC!(a.u,U,a.exitBC,a.perdir,t₁)
#     project!(a,b,0.5); BC!(a.u,a.uBC,a.exitBC,a.perdir,t₁)
#     push!(a.Δt,CFL(a))
# end

# define a line
@inline @fastmath line(x,t,L=1,thk=1) = (y = x .- SA[clamp(x[1],-L/2,L/2),0]; √sum(abs2,y)-thk/2)
@inline @fastmath ellipse(x,t;radius=1,Λ=1) = √sum(abs2, SA[x[1]/Λ,x[2]])-radius/Λ

import WaterLily: @loop
# falling body acceleration term
fall!(flow,t;acceleration) = for i ∈ 1:ndims(flow.p)
    @loop flow.f[I,i] += acceleration[i] over I ∈ CartesianIndices(flow.p)
end

#helper to rotate a vector
@inline @fastmath rotate(v,θ) = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]*v

# needed for stability of the ODE solver
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=0.1)
function plate(L=2^6;ν,A=1/2,U=1,ϵ=0.5,Λ=4,f=0.5f0,thk=2ϵ+√2,mem=Array)
    # pure heave motion
    function map(x,t)
        return x - SA[3L,3L-0.5f0*A*L*sin(2π*f*t*U/L)]
    end
    BiotSimulation((6L,6L),(0,0),L;U,ν,body=AutoBody((x,t)->ellipse(x,t;radius=L/2,Λ=Λ),map),ϵ,mem)
end


function run(L=64,U=1,Λ=4.f0,radius=L/2.f0)

    # some parameters
    # Re = L√(v²+u²)/ν   instantaneous Reynolds number
    # M = (m+m₁₁)/ρL²   -> m = ML² - m₁₁ (ρ=1)
    # A= a/L            -> a = AL
    # Ref = Laf/ν       -> ν = Laf/Ref = AL²f/Ref
    M = 1.f0
    A = 1.f0
    Ref = 10^4.f0
    f = 1.f0

    # derived parameters
    mₐ = SA[π*radius^2/Λ^2,π*radius^2] # [m₁₁,m₂₂] added-mass coefficient ellipse
    m = M*L^2 .- mₐ[1]                 # body mass
    vel = SA[0.01f0,0.f0]
    a0 = SA[0.f0,0.f0]
    pos = SA[0.f0,0.f0]

    # make the sim
    force = SA[0.f0,0.f0]
    sim = plate(L;A,ν=A*L^2*f/Ref,mem=Array)
    store=[]

    @gif for tᵢ in range(0,10.;step=0.05)
        # update
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U
            Δt = sim.flow.Δt[end]
            # remeasure the sim
            measure!(sim)
            biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                           fmm=sim.fmm,udf=fall!,acceleration=-a0,U=-vel) # change of frame
            # mom_step!(sim.flow,sim.pois;udf=fall!,acceleration=-a0,U=-vel)

            # compute pressure force and moment
            force = -WaterLily.total_force(sim) # in lab frame, fix y

            # update force in body frame
            accel = (force.*SA[1,0] + mₐ.*a0)./(m .+ mₐ)
            pos = pos + Δt.*(vel+Δt.*accel./2.)
            vel = vel + Δt.*accel
            a0 = copy(accel)

            # save position, velocity, etc
            push!(store,[Δt,force...,accel...,pos...,vel...])

            # update time, must be done globally to set the pos/vel correctly
            t_init = t; t += Δt
        end
        # plot vorticity
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(sdf(sim.body,loc(0,I),WaterLily.time(sim))<0,NaN,sim.flow.σ[I])
        flood(sim.flow.σ|>Array,shift=(-1.5,-1.5),clims=(-5,5), axis=([], false),
              cfill=:seismic,legend=false,border=:none,size=(1080,1080))
        body_plot!(sim);
        xs = SA[3sim.L,3sim.L] - sim.body.map(SA[3sim.L,3sim.L],WaterLily.time(sim))
        quiver!([xs[1]],[xs[2]],quiver=([force[1]],[force[2]]),color=:black,lw=2,arrow=:closed,legend=false)
        quiver!([xs[1]],[xs[2]],quiver=([sim.L*vel[1]],[sim.L*vel[2]]),color=:red,lw=2,arrow=:closed,legend=false)
        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3),
                " F₁=", round(force[1],digits=3), " a₁=", round(a0[1],digits=3))
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
    p3 = plot(cumsum(store_matrix[:,1]),2store_matrix[:,2]/sim.L,colorbar=:false, label="force")
    plot(p1, p2, p3, layout= @layout[a{0.5w} [grid(2,1)]])
end