using WaterLily,StaticArrays,ParametricBodies,Plots

function make_sim(positions;L=128,Re=1e3,U=1,T=Float32,mem=Array)

    # Map from simulation coordinate x to surface coordinate ξ
    function mapit(x,t,nose,αₘ)
        R = SA[cos(αₘ) -sin(αₘ); sin(αₘ) cos(αₘ)]
        ξ = R*(x-nose-SA[0.25f0L,0])+SA[0.25f0L,0] # move to origin and align with x-axis
        return SA[ξ[1],abs(ξ[2])]    # reflect to positive y
    end

    # Define foil using NACA0012 profile equation: https://tinyurl.com/NACA00xx
    NACA(s) = 0.6f0*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
    foil(s,t) = L*SA[(1-s)^2,NACA(1-s)]

    # empty first body
    body = WaterLily.NoBody() # HashedBody(foil,(0,1);map=(x,t)->mapit(x,t,SA[ L,1.5f0L],π/18.f0),T,mem)

    # add other ones
    for (α,c) in positions
        body += HashedBody(foil,(0,1);map=(x,t)->mapit(x,t,L.*c,α),T,mem)
    end

    Simulation((8L,4L),(U,0),L;ν=U*L/Re,body,T,mem)
end
using CUDA

function to_mem(a::AbstractSimulation;mem=CuArray)
    # make sure we don;t move the body if we copy
    @warn !isa(sim.body,WaterLily.NoBody) "Simulation cannot have a time dependent mapping"
    dims = size(sim.flow.p).-2
    # make the new sim, make it with an empty body
    b = Simulation(dims,a.flow.uBC,a.L;body=WaterLily.NoBody(),ν=a.flow.ν,T=eltype(a.flow.p),mem)
    # copy the flow and body
    copyto!(b.flow.σ, a.flow.σ)
    copyto!(b.flow.V, a.flow.V)
    copyto!(b.flow.μ₀, a.flow.μ₀)
    copyto!(b.flow.μ₁, a.flow.μ₁)
    # re-initialise the pressure solver
    WaterLily.update!(b.pois)
    return b;
end
# angle and position of the airfoils
positions = ((0.f0,SA[1,1.25f0]), (-π/18.f0,SA[1.9f0,1.5f0]), (-π/6.f0,SA[2.8f0,1.95f0]))

sim = make_sim(positions;mem=Array);
sim = to_mem(sim;mem=CuArray)
flood(sim.flow.σ,clims=(-1,1))

sim_gif!(sim,duration=10,step=0.1,clims=(-16,16),remeasure=false,
         plotbody=true,shift=(-2,-1.5),axis=([], false),cfill=:seismic,
         legend=false,border=:none)

# drag(flow,body,t) = sum(inside(flow.p)) do I
#     d,n,_ = measure(body,WaterLily.loc(0,I),t)
#     flow.p[I]*n[1]*WaterLily.kern(clamp(d,-1,1))
# end

# function Δimpulse!(sim)
#     Δt = sim.flow.Δt[end]*sim.U/sim.L
#     sim_step!(sim)
#     Δt*drag(sim.flow,sim.body,WaterLily.time(sim))
# end

# function mean_drag(φ,period=2)
#     sim = make_sim(φ)
#     sim_step!(sim,period) # warm-in transient period
#     impulse = 0           # integrate impulse
#     while sim_time(sim)<2period
#         impulse += Δimpulse!(sim)
#     end
#     impulse/period        # return mean drag
# end

# using Optim
# θ = Optim.minimizer(optimize(x->-mean_drag(first(x)), [0f0], Newton(),
#                     Optim.Options(show_trace=true,f_tol=1e-2); autodiff = :forward))