using GradientFlows, Test

@time begin
    @testset "Unit Tests" begin
        files = readdir("unit")
        for file in files
            include("unit/$file")
        end
    end
    @testset "Diffusion Tests" begin
        include("diffusion.jl")
    end
    @testset "Fokker-Planck Tests" begin
        include("fpe.jl")
    end
    @testset "Landau Tests" begin
        include("landau_isotropic.jl")
        include("landau_anisotropic.jl")
        include("landau_coulomb.jl")
    end
    @testset "Run Experiments" begin
        include("run_experiments.jl")
    end
end
