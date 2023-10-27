using GradientFlows, Test

@time begin
    @testset "Unit Tests" begin
        include("unit/Lp.jl")
        include("unit/blob.jl")
        include("unit/problem.jl")
        include("unit/landau.jl")
        include("unit/experiment.jl")
        include("unit/io.jl")
        include("unit/marginal.jl")
        include("unit/kde.jl")
    end
    @testset "Diffusion Tests" begin
        include("diffusion.jl")
    end
    @testset "Landau Tests" begin
        include("landau.jl")
    end
end
