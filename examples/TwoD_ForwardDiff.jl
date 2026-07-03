using WaterLily,StaticArrays,ParametricBodies,Plots
using ForwardDiff

# NOTE: T defaults to typeof(α) so that, under ForwardDiff, every field array
# (p, u, μ₀, μ₁, the Poisson hierarchy, and the body hash) holds Dual numbers
# and the α-partials propagate through the whole solve.
function make_sim(α;L=32,Re=1e3,U=1,T=typeof(α),mem=Array,initial_condition=false)

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
    α_eff, U₀ = initial_condition ? (zero(T), (cos(α)*U,sin(α)*U)) : (α, (U,0))
    body = HashedBody(foil,(0,1);map=(x,t)->mapit(x,t,L.*SA[2.f0,2.f0],α_eff),scale=one(T),T,mem)

    Simulation((8L,4L),U₀,L;ν=U*L/Re,body,T,mem)
end
using CUDA

# compute lift from a sim
function mean_lift(α;warmup=20,period=1,initial_condition=false)
    println("Testing α=$(α)")
    sim = make_sim(α;initial_condition)
    sim_step!(sim,warmup) # warm-in transient period
    impulse = 0           # integrate impulse
    t₀ = sim_time(sim)
    while sim_time(sim)<t₀+period
        Δt = sim.flow.Δt[end]*sim.U/sim.L
        sim_step!(sim)
        impulse -= Δt*2WaterLily.pressure_force(sim)[2]
    end
    impulse/period/sim.L # return mean lift coefficient
end

# test with finite difference and moving geom or initial conditions
@time dcldα_map = map(α -> map(i -> mean_lift(α+0.0001i),                         [-1,1]), -0.1:0.01:0.1)
@time dcldα_uBC = map(α -> map(i -> mean_lift(α+0.0001i; initial_condition=true), [-1,1]), -0.1:0.01:0.1)

# lift curve slope with auto diff
@time dCldα = ForwardDiff.derivative(γ -> mean_lift(γ; initial_condition=true), 0.0)
# this is messed-up, even for non-zero values of the AoA, I get unbounded values

using Plots
p1=plot(-0.1:0.01:0.1,  sum.(dcldα_map)./2, label="Cl (map)", lw=2) # average is the lift
p2=plot(-0.1:0.01:0.1, sum.(dcldα_uBC)./2, label="Cl (uBC)", lw=2) # average is the lift
for (c1,c2,α) in zip(dcldα_map,dcldα_uBC,-0.1:0.01:0.1)
    # plot gradient line
    dcldα1 = first(diff(c1)/0.0002) # average is the lift and diff/2h is gradient
    dcldα2 = first(diff(c2)/0.0002) # average is the lift and diff/2h is gradient
    plot!(p1, α .+ [-0.01,0.01] , sum(c1)/2 .+ dcldα1.*[-0.01,0.01], c=:blue,
         label=α≈0.1 ? "dCldα (map)" : :none, lw=1.5)
    plot!(p2, α .+ [-0.01,0.01] , sum(c2)/2 .+ dcldα2.*[-0.01,0.01], c=:red,
         label=α≈0.1 ? "dCldα (uBC)" : :none, lw=1.5)
end
for p in [p1,p2]
    plot!(p,[-0.01,0.01] , dCldα.*[-0.01,0.01], c=:orange, label="dCldα (uBC-AD)",lw=1.5)
    plot!(p,xlabel="Angle of Attack α (∘)", ylabel="2Fy/ρUL",ylims=(-0.25,0.25))
end
plot(p1, p2)
savefig("lift_curve_slop.png")