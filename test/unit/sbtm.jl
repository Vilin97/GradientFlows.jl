using GradientFlows, Distributions, LinearAlgebra, StableRNGs, Flux, Test
using GradientFlows: score, initialize, update!, l2_error_normalized, score_matching_loss, mlp
using Zygote: gradient, jacobian
using Flux.OneHotArrays: onehot
using Random: seed!

divergence(s, u) = tr(jacobian(s, u)[1])
rng = StableRNG(123)

@testset "SBTM tests" begin
    d, n = 2, 1000
    dist = MvNormal(1.0 * I(d))
    u = rand(rng, dist, n)
    score_values = score(dist, u)

    # test score_matching_loss
    α = 0.4e0
    s = mlp(d; depth=1, rng=rng)
    true_loss = (sum(abs2, s(u)) + 2 * divergence(s, u)) / size(u, 2)
    approx_loss = mean(score_matching_loss(s, u, ζ, α) for ζ in [randn(rng, d, n) for _ in 1:1000])
    @test approx_loss ≈ true_loss rtol = 0.1

    # test initialize
    solver = initialize(SBTM(s), u, copy(score_values), "dummy_problem_name")
    @test solver.score_values == score_values
    x = rand(rng, d, n)
    @test solver.s(x) == s(x)
    @test solver.optimiser_state == Flux.setup(solver.optimiser, s)

    # test train_s!
    train_s!(solver, u, score_values)
    @test l2_error_normalized(solver.s(u), score_values) ≈ 0 atol = solver.init_loss_tolerance

    # test update!
    prob = (solver=solver, f! = (args...) -> nothing, diffusion_coefficient=(u, params) -> 1, params=nothing)
    integrator = (u=u, p=prob, t=0.,)
    u .-= 0.01 * solver.score_values
    ζ = randn(rng, d, n)
    old_loss = score_matching_loss(solver.s, u, ζ, solver.denoising_alpha)
    update!(solver, integrator)
    new_loss = score_matching_loss(solver.s, u, ζ, solver.denoising_alpha)
    @test new_loss < old_loss

    # for coverage
    @test "$(SBTM())" == "SBTM"
end