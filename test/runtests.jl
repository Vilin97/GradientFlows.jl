using GradientFlows
using SafeTestsets

@time begin
    @time @safetestset "Diffusion Tests" begin include("diffusion.jl") end
end
