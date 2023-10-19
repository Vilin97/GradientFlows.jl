using GradientFlows, SafeTestsets

@time begin
    @time @safetestset "Unit Tests" begin
        include("unit/Lp.jl")
        include("unit/blob.jl")
        include("unit/problem.jl")
        include("unit/landau.jl")
        include("unit/experiment.jl")
        include("unit/io.jl")
    end
    @time @safetestset "Diffusion Tests" begin
        include("diffusion.jl")
    end
    @time @safetestset "Landau Tests" begin
        include("landau.jl")
    end
    @time @safetestset "Experiment Tests" begin
        include("experiment_set.jl")
    end
end
