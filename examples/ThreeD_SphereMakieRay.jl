using WaterLily,StaticArrays,BiotSavartBCs,CUDA

import BiotSavartBCs: interaction,symmetry,image
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₂,sgn₂ = image(T,size(ω),-2)  # image target and sign in y
    T₃,sgn₃ = image(T,size(ω),-3)  # image target and sign in z
    T₂₃,_   = image(T₃,size(ω),-2) # image of image!
    # Add up the four contributions
    return interaction(ω,T,args...)+sgn₃*interaction(ω,T₃,args...)+
     sgn₂*(interaction(ω,T₂,args...)+sgn₃*interaction(ω,T₂₃,args...))
end

# sphere with Biot-Savart boundary conditions
function make_sphere(N=2^6; R=N÷2, U=1, Re=3700, T=Float32, mem=CuArray)
    body = AutoBody((x,t)->√sum(abs2,x .- SA_F32[N,0,0])-R)
    BiotSimulation((3N,N,N), (U,0,0), R; ν=U*R/Re, body, T, mem, nonbiotfaces=(-2,-3))
end

# make the sim
sim = make_sphere(2^7; mem=CuArray)
sim_step!(sim, 10; remeasure=false, verbose=true)

## Visualization
using GLMakie
viz!(sim; duration=10, remeasure=false)