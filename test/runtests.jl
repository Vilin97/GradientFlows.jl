using GradientFlows

include("testutils.jl")

@time begin
    @testset "Unit Tests" begin
        include("unit/Lp.jl")
        include("unit/blob.jl")
        include("unit/problem.jl")
        include("unit/landau.jl")
        include("unit/experiment.jl")
        include("unit/io.jl")
    end
    @testset "Experiment Tests" begin
        include("experiment_set.jl")
    end
    @testset "Diffusion Tests" begin
        include("diffusion.jl")
    end
    @testset "Landau Tests" begin
        include("landau.jl")
    end
end
