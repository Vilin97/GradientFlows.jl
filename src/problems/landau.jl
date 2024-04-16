############ Isotropic Landau with Maxwell kernel ############
struct LandauParams{T,F}
    d::Int # dimension
    B::T   # collision kernel constant
    C::T
    K::F   # variance of MvNormal
end
LandauParams(d, B::T, C=one(T)) where {T} = LandauParams(d, B, C, t -> 1 - C * exp(-(d - 1) * 2 * B * t))

"Make an isotropic landau problem with Maxwell kernel with the given dimension, number of particles, and solver."
function landau_problem(d, n, solver_; dt::F=0.01, rng=DEFAULT_RNG, kwargs...) where {F}
    params = LandauParams(d, F(1 / 24))
    # Choose the starting time `t0` so that P ≈ 0 and P ≥ 0.
    t0 = round(log((params.d + 2) * params.C / 2) / (2params.B * (params.d - 1)), RoundUp, digits=1)
    ρ(t, params) = PolyNormal(d, params.K(t))
    ρ0 = ρ(t0, params)
    f! = landau_f!(d)
    tspan = (t0, t0 + 1)
    u0 = rand(rng, ρ0, n)
    name = "landau"
    solver = initialize(solver_, u0, score(ρ0, u0), name; kwargs...)
    covariance(t, params) = cov(ρ(t, params))
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver, name, landau_diffusion_coefficient, covariance)
end

############ Anisotropic Landau with Maxwell kernel ############
"Make an anisotropic landau problem with Maxwell kernel with the given dimension, number of particles, and solver."
function anisotropic_landau_problem(d, n, solver_; dt::F=0.01, rng=DEFAULT_RNG, kwargs...) where {F}
    params = (B=F(1 / 24),) # B = constant in the collision kernel
    t0 = F(0)
    ρ0 = MvNormal(diagm([F(1.8), F(0.2), ones(F, d - 2)...]))
    ρ(t, params) = t ≈ 0 ? ρ0 : MvNormal(mean(ρ0), covariance(F(Inf), params)) # if t > 0, steady-state, only accurate for large t

    f! = landau_f!(d)
    tspan = (t0, t0 + 10)
    u0 = rand(rng, ρ0, n)
    name = "anisotropic_landau"
    solver = initialize(solver_, u0, score(ρ0, u0), name; kwargs...)
    function covariance(t, params)
        Σ₀ = cov(ρ0)
        Σ∞ = I(d) .* tr(Σ₀) ./ d
        return Σ∞ - (Σ∞ - Σ₀)exp(-4d * params.B * t)
    end
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver, name, landau_diffusion_coefficient, covariance)
end

############ Landau with Coulomb kernel ############
"Make an anisotropic landau problem with Coulomb kernel with Gaussian initial condition with the given dimension, number of particles, and solver."
function coulomb_landau_normal_problem(d, args...; kwargs...)
    ρ0 = MvNormal(diagm([1.8, 0.2, ones(d - 2)...]))
    name = "coulomb_landau_normal"
    return coulomb_landau_problem_aux(args...; ρ0=ρ0, name=name, kwargs...)
end

"Make an anisotropic landau problem with Coulomb kernel with mixture of Gaussians initial condition with the given dimension, number of particles, and solver."
function coulomb_landau_mixture_problem(d, args...; kwargs...)
    if d != 2
        error("Coulomb Landau mixture problem with dimension d != 2 is not implemented.")
    end
    δ = 0.2
    μ = [sqrt(8*(1-δ))/2, 0]
    ρ0 = MixtureModel(MvNormal[MvNormal(μ, δ*I(2)), MvNormal(-μ, δ*I(2))], [1/2, 1/2])
    name = "coulomb_landau_mixture"
    return coulomb_landau_problem_aux(args...; ρ0=ρ0, name=name, kwargs...)
end

"Auxillary function to make a landau problem with Coulomb kernel."
function coulomb_landau_problem_aux(n, solver_; ρ0, name, dt::F=1.0, rng=DEFAULT_RNG, kwargs...) where {F}
    d = length(mean(ρ0))
    @assert(eltype(mean(ρ0)) == typeof(dt))
    t0 = F(0)
    params = (B=F(1 / 24),)
    ρ(t, params) = t ≈ 0 ? ρ0 : MvNormal(mean(ρ0), covariance(t, params)) # if t > 0, steady-state, only accurate for large t
    γ = -3
    f! = landau_f!(d, γ)
    tspan = (t0, t0 + 400) # t_end = 40 is used in https://www.sciencedirect.com/science/article/pii/S2590055220300184
    u0 = rand(rng, ρ0, n)
    solver = initialize(solver_, u0, score(ρ0, u0), name; kwargs...)
    function covariance(t, params) # steady-state, only accurate for large t
        Σ₀ = cov(ρ0)
        Σ∞ = I(d) .* tr(Σ₀) ./ d
        return Σ∞
    end
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver, name, landau_diffusion_coefficient, covariance)
end

############ Helper functions ############

"D = A∗u"
function landau_diffusion_coefficient(u, params)
    d, n = size(u)
    return (I(d) .* sum(abs2, u) .- u * u') .* (params.B / n)
end

"f! for different dimensions"
function landau_f!(d, γ=0)
    if d == 2
        return (args...) -> landau_2d_f!(args...; γ=γ)
    elseif d == 3
        return (args...) -> landau_3d_f!(args...; γ=γ)
    elseif d == 5
        return (args...) -> landau_5d_f!(args...; γ=γ)
    elseif d == 10
        return (args...) -> landau_10d_f!(args...; γ=γ)
    else
        error("Landau problem with dimension d = $d is not implemented.")
    end
end

landau_2d_f!(du, u, prob, t; γ) = landau_f_aux!(du, u, prob, Val(2); γ=γ)
landau_3d_f!(du, u, prob, t; γ) = landau_f_aux!(du, u, prob, Val(3); γ=γ)
landau_5d_f!(du, u, prob, t; γ) = landau_f_aux!(du, u, prob, Val(5); γ=γ)
landau_10d_f!(du, u, prob, t; γ) = landau_f_aux!(du, u, prob, Val(10); γ=γ)

@generated function landau_f_aux!(du, u, prob, ::Val{d}; γ) where {d}
    quote
        ε = γ < 0 ? eps(eltype(du)) : 0 # TODO: keep epsilon positive
        s = prob.solver.score_values
        du .= 0
        n = size(u, 2)
        @tturbo for p = 1:n
            Base.Cartesian.@nexprs $d i -> dx_i = zero(eltype(du))
            for q = 1:n
                dotzv = zero(eltype(du))
                normsqz = zero(eltype(du))
                Base.Cartesian.@nexprs $d i -> begin
                    z_i = u[i, p] - u[i, q]
                    v_i = s[i, q] - s[i, p]
                    dotzv += z_i * v_i
                    normsqz += z_i * z_i
                end
                normz_pow_γ = 1/(sqrt(normsqz) + ε)^(-γ)
                Base.Cartesian.@nexprs $d i -> begin
                    dx_i += (v_i * normsqz - dotzv * z_i) * normz_pow_γ
                end
            end
            Base.Cartesian.@nexprs $d i -> begin
                du[i, p] += dx_i
            end
        end
        du .*= prob.params.B / n
        nothing
    end
end
