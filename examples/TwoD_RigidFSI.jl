using WaterLily,StaticArrays#,CUDA,BiotSavartBCs
using JLD2,Plots
using FastGaussQuadrature
using LinearAlgebra: dot
const xᵢ,wᵢ = gausslegendre(1000)

WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=0.4)

# force on _each_ body
each_force(flow,body) = WaterLily.pressure_force(flow,body)
each_force(flow,body::WaterLily.SetBody{typeof(min)}) = mapreduce(bod->each_force(flow,bod),vcat,(body.a,body.b))

function ellipse(D,Λ=6.0;A₀=1.4,St=0.5,U=1,Re=100,T=Float32,mem=Array,use_biotsavart=false)
    # h₀=T(A₀*D/2); ω=T(2π*St*U/D); amp=T(π/10)
    # parameters for the motion
    A₁ =T(A₀*D); σᵣ=T(0.628); σₜ=T(0.628); β=π/10; ϕ=0; f=T(St*U/2D)
    # start-up conditioner
    @inline C(t) = (tanh(8t-2)+tanh(2))/(1+tanh(2))
    # rotation shape function
    @inline Gᵣ(t) = tanh(σᵣ*cos(2π*t+ϕ))
    # fast quadrature integral on the interval [a,b]
    @inline ∫(f,a,b) = (b-a)/2*dot(f.(0.5*(b+a).+xᵢ.*(b-a)/2), wᵢ)
    # translation shape function
    @inline Gₜ(t) = ∫(u->tanh(σₜ*cos(2π*u+ϕ)),t,100)
    # maximum of the shape function
    maxGᵣ = maximum(Gᵣ.(0:0.01:10)); maxGₜ = maximum(Gₜ.(0:0.01:10))
    # rotation angle
    @inline α₁(t) = -β*Gᵣ(f*t)/maxGᵣ
    # linear motion
    @inline X₁(t) = A₁/2*Gₜ(f*t)/maxGₜ*C(t/D)
    # map of the first foil
    function mapα(x,t)
        α = α₁(t); R = SA[cos(α) sin(α); -sin(α) cos(α)]
        R * (x .- SA[3D-X₁(t),4D])
    end
    function mapθ(x,t)
        α = α₁(t); R = SA[cos(α) sin(α); -sin(α) cos(α)]
        # position of hinge/torsion spring
        x₀ = R*(x.-SA[3D-X₁(t),4D])
        # rotation diff θ
        θ = -α # will be solved for after
        R = SA[cos(θ) sin(θ); -sin(θ) cos(θ)]
        return R*(x₀ .+ SA[0,0.55D]) .+ SA[0,0.55D]
    end
    # sdf of the ellipse
    sdf(x,t) = √sum(abs2,SA[x[1],x[2]/Λ])-D÷2/Λ
    # make a first body
    body = AutoBody(sdf,mapα)
    # add another one
    body += AutoBody(sdf,mapθ)
    # use_biotsavart && BiotSimulation((6D,6D), (0,0), D; U, body, ν=U*D/Re, T, mem)
    Simulation((6D,6D), (0,0), D; U, body, ν=U*A₁/Re, T, mem)
end

# run
sim = ellipse(32;St=0.5,T=Float64,mem=Array,use_biotsavart=false)
R = inside(sim.flow.p)
forces = []
anim = @animate for tᵢ in range(0.,10,step=0.1)
    while sim_time(sim) < tᵢ
        sim_step!(sim;remeasure=true)
        # pres,visc = WaterLily.pressure_force(sim),WaterLily.viscous_force(sim)
        pres = each_force(sim.flow,sim.body)
        push!(forces,[sim_time(sim),pres...])
    end
    # solve the equation of motion
    # θ =
    println("tU/L=",round(sim_time(sim),digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ[R]|>Array,clims=(-10,10))
end
gif(anim,"airfoil.gif")

using WaterLily,StaticArrays#,CUDA,BiotSavartBCs
using JLD2,Plots
using FastGaussQuadrature
using LinearAlgebra: dot
const xᵢ,wᵢ = gausslegendre(1000)
using OrdinaryDiffEq

T=Float64
c=32
U=1
A₀=1.4
St=0.1

# parameters for the motion
A₁ =T(A₀*c); σᵣ=T(0.628); σₜ=T(0.628); β=T(π/4); ϕ=0; f=T(St*U/2c)
# start-up conditioner
@inline C(t) = (tanh(8t-2)+tanh(2))/(1+tanh(2))
# rotation shape function
@inline Gᵣ(t) = tanh(σᵣ*cos(2π*t+ϕ))
# fast quadrature integral on the interval [a,b]
@inline ∫(f,a,b) = (b-a)/2*dot(f.(0.5*(b+a).+xᵢ.*(b-a)/2), wᵢ)
# translation shape function
@inline Gₜ(t) = ∫(u->tanh(σₜ*cos(2π*u)),t,100)
# maximum of the shape function
maxGᵣ = maximum(Gᵣ.(0:0.01:50.0)); maxGₜ = maximum(Gₜ.(0:0.01:50.0))
# rotation angle
@inline α₁(t) = -β*Gᵣ(f*t)/maxGᵣ
using ForwardDiff
@inline dα₁(t) = ForwardDiff.derivative(τ->α₁(τ),t)
@inline ddα₁(t) = ForwardDiff.derivative(τ->dα₁(τ),t)
# linear motion
@inline X₁(t) = A₁/2*Gₜ(f*t)/maxGₜ*C(t/c)
@inline dX₁(t) = ForwardDiff.derivative(τ->X₁(τ),t)
@inline ddX₁(t) = ForwardDiff.derivative(τ->dX₁(τ),t)

# t = 0.0:0.1:4.0c

# plot(t*f,C.(t/c),xlabel="t/T",ylabel="C(t)",label="start-up function")

# plot(t*f,X₁.(t)./c,xlabel="t/T",ylabel="X₁(t)/c",label="position")
# plot!(t*f,dX₁.(t)./c,xlabel="t/T",ylabel="dX₁/dt/c",label="linear velocity")
# plot!(t*f,ddX₁.(t)./c,xlabel="t/T",ylabel="dX₁/dt/c",label="linear acceleration")

# plot(t*f,rad2deg.(α₁.(t)),xlabel="t/T",ylabel="α₁(t)",label="angle")
# plot!(t*f,rad2deg.(dα₁.(t)),xlabel="t/T",ylabel="dX₁/dt",label="angular velocity")
# plot!(t*f,rad2deg.(ddα₁.(t)),xlabel="t/T",ylabel="dX₁/dt",label="angular acceleration")

# function
function fsi!(du, u, p, t)
    # unpack state
    θ,dθ,Mh = u
    # unpack parameters and functions
    m,I,L₁,L₂,K,R,α,ddX,dα,ddα = p
    # rates
    du[1] = dθ
    du[2] = (-R*dθ - K*θ - (m*L₂*cos(α(t) + θ))*ddX(t)
            -(I + m*(L₂^2+L₁*L₂*cos(θ)))*ddα(t)
            -(m*L₁*L₂*sin(θ))*dα(t)^2 + Mh) / (I+m*L₂^2)
    du[3] = 0.0f0 # dummy
end

# fsi parameters
radius = c/2.f0
Λ = 5.f0
ρ = 5.f0 # density ratios
m = ρ*π*radius^2/Λ # mass
K = 456.f0*f^2*c^4
R = 3.95f0*f*c^4
L₁ = L₂ = (1.f0+0.25f0/5.1f0)c/2.f0
# rotation variables, moment of inertia with parallel axis theorem
I = 0.25f0*m*(radius^2+radius^2/Λ^2) + m*(L₂^2)
Iₐ = 0.125f0*π*(radius^2-radius^2/Λ^2)^2 + m*(L₂^2) # added mass ellipse m₆₆
ω = 0.0f0; α = 0.0f0

u₀ = [α, ω, 0.f0] # initial θ, dθ
params = (m,I,L₁,L₂,K,R,α₁,ddX₁,dα₁,ddα₁)
tspan = (0.0,4.0c)

# pass to solvers, dummy case
# prob = ODEProblem(fsi!, u₀, tspan, params)
# sol = solve(prob, Tsit5(), dtmax=0.01c)
# plot(sol.t*f, X₁.(sol.t)./c, label="X₁(t)/c", xlabel="t/T")
# plot!(sol.t*f, [rad2deg.(getindex.(sol.u,1)), rad2deg.(getindex.(sol.u,2))],
#      label=["θ(t)" "dθ/dt"], xlabel="t/T")

# fsi problem and solver
fsi = init(ODEProblem(fsi!, u₀, tspan, params), Tsit5(),
            reltol=1e-6, abstol=1e-6, save_everystep=false)


# force on _each_ body
each_force(flow,body) = WaterLily.pressure_force(flow,body)
each_force(flow,body::WaterLily.SetBody{typeof(min)}) = mapreduce(bod->each_force(flow,bod),vcat,(body.a,body.b))

function ellipse(D,Λ=5.0;A₀=1.4,St=0.5,U=1,Re=100,T=Float32,mem=Array,use_biotsavart=false)

    # map of the first foil
    function map(x,t)
        α = α₁(t); R = SA[cos(α) sin(α); -sin(α) cos(α)]
        R * (x .- SA[3D-X₁(t),4D])
    end
    # function mapθ(x,t)
    #     α = α₁(t); R = SA[cos(α) sin(α); -sin(α) cos(α)]
    #     # position of hinge/torsion spring
    #     x₀ = R*(x.-SA[3D-X₁(t),4D])
    #     # rotation diff θ
    #     θ = -α # will be solved for after
    #     R = SA[cos(θ) sin(θ); -sin(θ) cos(θ)]
    #     return R*(x₀ .+ SA[0,0.55D]) .+ SA[0,0.55D]
    # end
    # sdf of the ellipse
    sdf(x,t) = √sum(abs2,SA[x[1],x[2]/Λ])-D÷2/Λ
    # make a first body
    body = AutoBody(sdf,map)
    # position of hinge/torsion spring
    α₀ = -α₁(0); R = SA[cos(α₀) sin(α₀); -sin(α₀) cos(α₀)]
    x₀ = SA_F32[3D,4D]-R*SA[0,L₁+L₂]
    # add another one
    body += RigidBody(sdf,x₀,SA[0,0],-α₀,0.f0)
    Simulation((6D,6D), (0,0), D; U, body, ν=U*A₁/Re, T, mem)
end

# run
sim = ellipse(c,Λ;T=Float64,mem=Array,use_biotsavart=false)
R = inside(sim.flow.p)
# @gif for tᵢ in 0:0.1:5
let tᵢ=rand()
    # tᵢ = tᵢ*sim.L
    @show tᵢ
    # update the bodys
    α₀ = -α₁(tᵢ); Rt = SA[cos(α₀) sin(α₀); -sin(α₀) cos(α₀)]
    xₕ = SA_F32[3sim.L-X₁(tᵢ),4sim.L]-Rt*SA[0,L₁]
    pivot = SA[0,L₂]; x₀ = xₕ-pivot
    # make new body
    body = deepcopy(sim.body.a)
    body += RigidBody((x,t)->√sum(abs2,SA[x[1],x[2]/Λ])-c÷2/Λ,x₀,SA[0,0],pivot,-α₀-0.5,ω)
    sim.body = body
    measure!(sim,tᵢ)
    flood(sim.flow.σ[R],shift=(-0.5,-0.5), clims=(-1,1), color=:oslo)
    scatter!([xₕ[1]],[xₕ[2]],markersize=5,color=:pink,label="xₕ")
    scatter!([body.b.center[1]],[body.b.center[2]],markersize=5,color=:red,label="center")
end

# forces = []
# let
#     α₀ = -α₁(0); Rt = SA[cos(α₀) sin(α₀); -sin(α₀) cos(α₀)]
#     xₕ = SA_F32[3sim.L-X₁(0),4sim.L]-Rt*SA[0,L₁]
#     θ,ω = 0.f0,0.f0
#     anim = @animate for tᵢ in range(0,1/f,step=0.1/f)
#         while sim_time(sim) < tᵢ
#             t = sim_time(sim)
#             # get moment
#             # moment = -WaterLily.pressure_moment(xₕ,sim.flow,sim.body.b)[1]
#             # # update ODE solver
#             # SciMLBase.set_u!(fsi, [θ,ω,0.f0])
#             # OrdinaryDiffEq.step!(fsi, sim.flow.Δt[end], true)
#             # # extract results
#             # θ,ω = fsi.u[1:2]
#             # Vb = SA[dX₁(t),0.f0]
#             # α₀ = -α₁(t); Rt = SA[cos(α₀) sin(α₀); -sin(α₀) cos(α₀)]
#             # xₕ = SA_F32[3sim.L-X₁(t),4sim.L]-Rt*SA[0,L₁]
#             # pivot = SA[0,L₂]; x₀ = xₕ-pivot
#             # # update the body
#             # body = deepcopy(sim.body.a)
#             # body += RigidBody((x,t)->√sum(abs2,SA[x[1],x[2]/Λ])-c÷2/Λ,x₀,Vb,pivot,-α₀+θ,ω)
#             # sim.body = body

#             # measure and update flow
#             sim_step!(sim)
#         end
#         # solve the equation of motion
#         # θ =
#         println("tU/L=",round(sim_time(sim),digits=2),", Δt=",round(sim.flow.Δt[end],digits=3))
#         @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
#         flood(sim.flow.σ[R]|>Array,clims=(-10,10))
#         body_plot!(sim)
#     end
#     gif(anim,"airfoil.gif")
# end
