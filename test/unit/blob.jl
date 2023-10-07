using GradientFlows, Distributions, LinearAlgebra, StableRNGs
using GradientFlows: score, initialize, update!

rng = StableRNG(123)

struct DummyIntegrator
    u::Matrix{Float64}
end

d, n = 2, 10
dist = MvNormal(1.0 * I(d))
u = rand(rng, dist, n)
score_values = score(dist, u)

# test initialize
blob = initialize(Blob(), score_values)
@test blob.score_values == score_values

# test that update! is idempotent
integrator = DummyIntegrator(u)
update!(blob, integrator)
score_values = copy(blob.score_values)
update!(blob, integrator)
@test blob.score_values == score_values

# Now we change u
old_u = copy(u)
u .-= 0.01 * blob.score_values
update!(blob, integrator)
@test blob.score_values != score_values
