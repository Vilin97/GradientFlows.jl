using GradientFlows, SafeTestsets

include("testutils.jl")
@time begin
    @time @safetestset "Unit Tests" begin
        include("unit/Lp.jl")
        include("unit/blob.jl")
        include("unit/problem.jl")
        include("unit/landau.jl")
    end
    include("diffusion.jl")
end
