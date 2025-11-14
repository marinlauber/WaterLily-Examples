using WaterLily,StaticArrays,Plots,Clustering
import LinearAlgebra: tr
@inline @fastmath function Qcriterion(I,u)
    J = ∇u(I,u)
    S,Ω = (J+J')/2,(J-J')/2
    0.5*(√(tr(Ω*Ω'))^2-√(tr(S*S'))^2)
end
@inline @fastmath ∇u(I::CartesianIndex{2},u) = @SMatrix [WaterLily.∂(i,j,I,u) for i ∈ 1:2, j ∈ 1:2]
@inline @fastmath ∇u(I::CartesianIndex{3},u) = @SMatrix [WaterLily.∂(i,j,I,u) for i ∈ 1:3, j ∈ 1:3]

function hover(L=2^6;Re=250,U=1,amp=π/4,ϵ=0.5,thk=2ϵ+√2,mem=Array)
    # Line segment SDF
    function sdf(x,t)
        y = x .- SA[0,clamp(x[2],-L/2,L/2)]
        √sum(abs2,y)-thk/2
    end
    # Oscillating motion and rotation
    function map(x,t)
        α = amp*cos(t*U/L); R = SA[cos(α) sin(α); -sin(α) cos(α)]
        R * (x - SA[3L-L*sin(t*U/L),4L])
    end
    Simulation((6L,6L),(0,0),L;U,ν=U*L/Re,body=AutoBody(sdf,map),ϵ,mem)
end
"""
    segment!(σ;δ=0,ϵ=3)
"""
function segment!(σ::AbstractArray;δ=0,ϵ=3,min_neighbors=1,min_cluster_size=1)
    # extract region of nonzero criterion
    Is = findall(σ .> δ); pnts = hcat(Vector.(loc.(0,Is))...)
    # dbscan clustering
    clusters = dbscan(pnts,ϵ;min_neighbors,min_cluster_size)
    # # compute centers and strengths
    # center = []; Γ =[]
    # for c in clusters.clusters
    #     # centroid
    #     push!(center,sum(pnts[:,c.core_indices],dims=2)/length(c.core_indices))
    #     # circulation Γᵢ = ∑_Vᵢωdxdy
    #     push!(Γ,sum(@views(σ[Is[c.core_indices]])))
    # end
    # # centroid area-averaged velocity
    # # fill back the masked array
    # σ .= 0; σ[Is] = clusters.assignments;
    # return center,Γ
    return Is,clusters.assignments
end

# using CUDA
sim = hover()#mem=CuArray)
sim_step!(sim,10)

# Q-criterion field
@inside sim.flow.σ[I] = WaterLily.μ₀(sdf(sim.body,loc(0,I),WaterLily.time(sim)),1)*Qcriterion(I,sim.flow.u)*sim.L^2/sim.U
flood(sim.flow.σ|>Array,shift=(-2,-1.5),clims=(-20,20), axis=([], false),
        cfill=:seismic,legend=false,border=:none,size=(6*sim.L,6*sim.L))

flood(sim.flow.σ|>Array,shift=(-2,-1.5),clims=(3,20), axis=([], false),
        cfill=:seismic,legend=false,border=:none,size=(6*sim.L,6*sim.L))

# segment the vortical structures
cs = segment!(sim.flow.σ;δ=3,ϵ=3,min_cluster_size=16)
p1=flood(sim.flow.σ,axis=([], false),cfill=:tab10)
scatter!(getindex.(cs[1],1),getindex.(cs[1],2),label=:none,color=:black,alpha=0.5)

# step one an segment again
sim_step!(sim,sim_time(sim)+0.1)

@inside sim.flow.σ[I] = WaterLily.μ₀(sdf(sim.body,loc(0,I),WaterLily.time(sim)),1)*Qcriterion(I,sim.flow.u)*sim.L^2/sim.U
cs = segment!(sim.flow.σ;δ=3,ϵ=3,min_cluster_size=16)
p2=flood(sim.flow.σ,axis=([], false),cfill=:tab10)
scatter!(getindex.(cs[1],1),getindex.(cs[1],2),label=:none,color=:black,alpha=0.5)

# together
plot(p1,p2,layout=(1,2))