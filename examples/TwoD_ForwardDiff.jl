using WaterLily,StaticArrays,ParametricBodies,Plots
using ForwardDiff

# NOTE: T defaults to typeof(α) so that, under ForwardDiff, every field array
# (p, u, μ₀, μ₁, the Poisson hierarchy, and the body hash) holds Dual numbers
# and the α-partials propagate through the whole solve.
function make_sim(α;L=32,Re=1e3,U=1,T=typeof(α),mem=Array)

    # Map from simulation coordinate x to surface coordinate ξ
    function mapit(x,t,nose,αₘ)
        R = SA[cos(αₘ) -sin(αₘ); sin(αₘ) cos(αₘ)]
        ξ = R*(x-nose-SA[0.25f0L,0])+SA[0.25f0L,0] # move to origin and align with x-axis
        return SA[ξ[1],abs(ξ[2])]    # reflect to positive y
    end

    # Define foil using NACA0012 profile equation: https://tinyurl.com/NACA00xx
    NACA(s) = 0.6f0*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
    foil(s,t) = L*SA[(1-s)^2,NACA(1-s)]

    # make a body
    # scale=one(T) skips ParametricBodies' get_scale, which would otherwise run an
    # inner ForwardDiff.jacobian over `map` at construction. Since `map` closes over
    # the outer Dual α, that nested AD collapses the value type and throws
    # `Float64(::Dual)`. For a rigid rotation |dx/dξ|=1, so scale=one(T) is exact.
    body = HashedBody(foil,(0,1);map=(x,t)->mapit(x,t,L.*SA[2.f0,2.f0],α),scale=one(T),T,mem)

    Simulation((8L,4L),(U,0),L;ν=U*L/Re,body,T,mem)
end
using CUDA

# compute lift from a sim
function mean_lift(α;period=1)
    println("Testing α=$(α)")
    sim = make_sim(α)
    sim_step!(sim,10) # warm-in transient period
    impulse = 0           # integrate impulse
    t₀ = sim_time(sim)
    while sim_time(sim)<t₀+period
        Δt = sim.flow.Δt[end]*sim.U/sim.L
        sim_step!(sim)
        impulse -= Δt*2WaterLily.pressure_force(sim)[2]
    end
    impulse/period/sim.L # return mean lift coefficient
end

# test a single run
cl = map(α->mean_lift(α), 0:0.1:0.4)

# lift curve slope
@time dCldα = ForwardDiff.derivative(α -> mean_lift(α), 0.1)
