using GradientFlows, SafeTestsets

@time begin
    @time @safetestset "Unit Tests" begin
        include("unit/Lp.jl")
        include("unit/blob.jl")
        include("unit/problem.jl")
    end
    @time @safetestset "Diffusion Tests" begin
        include("diffusion.jl")
    end
end
