"Make a homogeneous landau problem with Maxwell kernel with the given dimension, number of particles, and solver."
function landau_problem(d, n, solver_; t0::F=6.0, t_end::F=6.5, dt::F=0.01, rng=DEFAULT_RNG) where {F}
    if solver_ isa SBTM && F == Float64
        return landau_problem(d, n, solver_; t0=Float32(t0), t_end=Float32(t_end), dt=Float32(dt), rng=rng)
    end
    f! = choose_f!(d)
    tspan = (t0, t_end)
    ρ(t, params) = LandauDistribution(d, params.K(t))
    params = LandauParams(d, F(1 / 24))
    ρ0 = ρ(t0, params)
    u0 = F.(rand(rng, ρ0, n))
    solver = initialize(solver_, u0, score(ρ0, u0))
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver)
end

struct LandauParams{T,F}
    d::Int # dimension
    B::T
    C::T
    K::F   # variance of MvNormal
end
LandauParams(d, B::T, C=one(T)) where {T} = LandauParams(d, B, C, t -> 1 - exp(-(d - 1) * 2 * B * t))

# Distribution (2π * K)^(-d / 2) * exp(-x² / (2K)) * (P + Q * x²)
struct LandauDistribution{F} <: ContinuousMultivariateDistribution
    d::Int
    K::F
end

function pdf(dist::LandauDistribution, x::AbstractVector)
    K = dist.K
    d = dist.d
    P = ((d + 2) * K - d) / (2K)
    Q = (1 - K) / (2K^2)
    x² = sum(abs2, x)
    return (2π * K)^(-d / 2) * exp(-x² / (2K)) * (P + Q * x²)
end

function Random.rand(dist::LandauDistribution, n::Int)
    return rand(DEFAULT_RNG, dist, n)
end

function Random.rand(rng::Random.AbstractRNG, dist::LandauDistribution, n::Int)
    d = dist.d
    K = dist.K
    β = 1.5
    proposal = MvNormal(K * I(d) * β)
    xs = [[x, zeros(typeof(K), d - 1)...] for x in 0:0.01:5]
    M = maximum(x -> pdf(dist, x) / pdf(proposal, x), xs)+1
    u = zeros(typeof(K), d, n)
    for i in 1:n
        u[:, i] = rejection_sample(dist, proposal, M, rng)
    end
    return u
end

function rejection_sample(target_dist, proposal_dist, M, rng=DEFAULT_RNG)
    f(x) = pdf(target_dist, x)
    g(x) = pdf(proposal_dist, x)
    while true
        x = rand(rng, proposal_dist)
        if M * g(x) < f(x)
            error("M = $M is too low: $(M*g(x)) = Mg(x) < f(x) = $(f(x)) for x = $x.")
        end
        if rand(rng) * M * g(x) < f(x) # accept with probability f(x)/Mg(x)
            return x
        end
    end
end

function gradlogpdf(dist::LandauDistribution, x)
    K = dist.K
    d = dist.d
    P = ((d + 2) * K - d) / (2K)
    Q = (1 - K) / (2K^2)
    return x .* (-1 / K + 2Q / (P + Q * sum(abs2, x)))
end

mean(dist::LandauDistribution) = zeros(dist.d)
cov(dist::LandauDistribution) = I(dist.d)

# f! for different dimensions
function choose_f!(d)
    if d == 3
        return landau_3d_f!
    elseif d == 5
        return landau_5d_f!
    else
        error("Landau problem with dimension d = $d is not implemented.")
    end
end

function landau_3d_f!(du, u, prob, t)
    s = prob.solver.score_values
    du .= 0
    n = size(u, 2)
    @tturbo for p = 1:n
        Base.Cartesian.@nexprs 3 i -> dx_i = zero(eltype(du))
        for q = 1:n
            dotzv = zero(eltype(du))
            normsqz = zero(eltype(du))
            Base.Cartesian.@nexprs 3 i -> begin
                z_i = u[i, p] - u[i, q]
                v_i = s[i, q] - s[i, p]
                dotzv += z_i * v_i
                normsqz += z_i * z_i
            end
            Base.Cartesian.@nexprs 3 i -> begin
                dx_i += v_i * normsqz - dotzv * z_i
            end
        end
        Base.Cartesian.@nexprs 3 i -> begin
            du[i, p] += dx_i
        end
    end
    du .*= prob.params.B/n
    nothing
end
function landau_5d_f!(du, u, prob, t)
    s = prob.solver.score_values
    du .= 0
    n = size(u, 2)
    @tturbo for p = 1:n
        Base.Cartesian.@nexprs 5 i -> dx_i = zero(eltype(du))
        for q = 1:n
            dotzv = zero(eltype(du))
            normsqz = zero(eltype(du))
            Base.Cartesian.@nexprs 5 i -> begin
                z_i = u[i, p] - u[i, q]
                v_i = s[i, q] - s[i, p]
                dotzv += z_i * v_i
                normsqz += z_i * z_i
            end
            Base.Cartesian.@nexprs 5 i -> begin
                dx_i += v_i * normsqz - dotzv * z_i
            end
        end
        Base.Cartesian.@nexprs 5 i -> begin
            du[i, p] += dx_i
        end
    end
    du .*= prob.params.B/n
    nothing
end
