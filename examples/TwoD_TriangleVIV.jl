using WaterLily,ParametricBodies,StaticArrays,Plots

function main(L=64;Re=550,U=1,mr=1.2f0,ξ=0.021f0,T=Float32,pivot=T(25L/12),h=T(L*√3/2),mem=Array)
    # Define the triangle as a order 1 nurbs, pivot is a 1.75L from the geometric center
    x₀,xₚ,θ,ω = SA{T}[3L, 4L],SA{T}[pivot,0],T(0.1),T(0)
    body = ParametricBody(BSplineCurve(SA{T}[0 0 h 0;L/2 -L/2 0 L/2]);map=RigidMap(x₀,θ;xₚ,ω))

    # make sim
    sim = Simulation((12L,8L),(U,0),L;ν=U*L/Re,body,T,mem)

    # Set dynamical properties
    m,ma = √3L^2/4,mr*√3L^2/4 # mass (density = 1) and added mass
    Is = L*h^3/12 + m*pivot^2   # solid moment of inertia
    Ia = ma*h^2/8 + ma*pivot^2  # added moment of inertia
    c = 2ξ*m                    # rotational damping coefficient
    α = zero(T)                 # initial angular acceleration

    # Integrate and plot
    @time @gif for tᵢ in range(0,50;step=0.1)
        # update until time tᵢ in the background
        while sim_time(sim) < tᵢ
            # the step we are doing
            Δt = sim.flow.Δt[end]
            # get moment
            moment = -WaterLily.pressure_moment(x₀+xₚ, sim.flow, sim.body)[1]
            # update rotation ODE
            α = (moment + c*ω + Ia*α) / (Is + Ia)
            ω += Δt*α; θ += Δt*ω # Verlet
            # update the body
            sim.body = setmap(sim.body; θ=T(θ), ω=T(ω))
            # measure and update flow
            sim_step!(sim;remeasure=true)
        end
        @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I])<0.001,0.0,sim.flow.σ[I])
        flood(sim.flow.σ,shift=(-1.5,-1.5),clims=(-5,5),axis=([],false),
              cfill=:seismic,legend=false,border=:none,size=(1080,720)); body_plot!(sim)
        plot!(title="tU/L $tᵢ, θ=$(round(rad2deg(θ), digits=1))°")
        scatter!([x₀[1]+xₚ[1]],[x₀[2]+xₚ[2]],marker=:o,color=:red,legend=:none)
        println("tU/L=", round(tᵢ, digits=4), ", θ=", round(rad2deg(θ), digits=1), "°, ω=", round(ω*sim.L/sim.U, digits=3))
    end
    return sim
end

# run the main function
using CUDA
sim = main(mem=CuArray);