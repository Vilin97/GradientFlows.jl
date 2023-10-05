using Test, StableRNGs, GradientFlows, LinearAlgebra
using .GradientFlows: normsq, fastdot

rng = StableRNG(123)

function test_function_equality(f, g, args)
    @test f(args) == g(args)
end

# Test normsq(z)
@testset "normsq(z) tests" begin
    for z in [[1, 2, 3], [0, 0, 0], [-1, -2, -3]]
        test_function_equality(normsq, x -> sum(abs2.(x)), z)
    end
    z = rand(rng, 3)
    (@elapsed normsq(z)) < (@elapsed sum(abs2.(z)))
end

# Test normsq(z1, z2)
@testset "normsq(z1, z2) tests" begin
    for z1 in [[1, 2, 3], [0, 0, 0], [-1, -2, -3]]
        for z2 in [[1, 2, 3], [0, 0, 0], [-1, -2, -3]]
            test_function_equality(z -> normsq(z...), z -> sum(abs2.(z[1] .- z[2])), (z1, z2))
        end
    end
    z1 = rand(rng, 3)
    z2 = rand(rng, 3)
    (@elapsed normsq(z1, z2)) < (@elapsed sum(abs2.(z1 .- z2)))
end

# Test fastdot(z, v)
@testset "fastdot(z, v) tests" begin
    for z in [[1, 2, 3], [0, 0, 0], [-1, -2, -3]]
        for v in [[1, 2, 3], [0, 0, 0], [-1, -2, -3]]
            test_function_equality(p -> fastdot(p...), p -> dot(p...), (z, v))
        end
    end
    z = rand(rng, 3)
    v = rand(rng, 3)
    (@elapsed fastdot(z, v)) < (@elapsed dot(z, v))
end