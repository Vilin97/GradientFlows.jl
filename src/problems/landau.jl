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
    t0 = round(log((params.d + 2) * params.C / 2) / (2params.B * (params.d - 1)), RoundUp, digits=1)
    ρ(t, params) = PolyNormal(d, params.K(t))
    ρ0 = ρ(t0, params)
    f! = landau_f!(d)
    tspan = (t0, t0 + 4)
    u0 = rand(rng, ρ0, n)
    name = "landau"
    solver = initialize(solver_, u0, score(ρ0, u0), name; kwargs...)
    covariance(t, params) = cov(ρ(t, params))
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver, name, landau_diffusion_coefficient, covariance)
end

############ Anisotropic Landau ############
"""
    landau_problem_factory(d, n, solver_; IC::String, γ::Number, covariance_scale::Int, kwargs...)

Return a function `problem(d, n, solver)` that creates a landau problem with the given dimension, number of particles, and solver.

IC ∈ {"normal", "mixture"} : initial condition
γ  ∈ {0, -3}               : power in collision kernel
covariance_scale = 1       : scale of the covariance matrix
"""
function landau_problem_factory(d; IC::String, γ::Number, covariance_scale=1::Int, kwargs...)
    σ = covariance_scale
    δ = 0.2
    if IC == "normal"
        ρ0 = MvNormal(diagm(σ .* [2-δ, δ, ones(d - 2)...]))
    elseif IC == "mixture"
        μ = [sqrt(σ), zeros(d-1)...]
        Σ = diagm(σ .* [1-δ, δ, ones(d-2)...])
        ρ0 = MixtureModel(MvNormal[MvNormal(μ, Σ), MvNormal(-μ, Σ)], [1/2, 1/2])
    end
    function covariance(t, params)
        Σ₀ = cov(ρ0)
        Σ∞ = I(d) .* tr(Σ₀) ./ d
        return γ==0 ? Σ∞ - (Σ∞ - Σ₀)exp(-4d * params.B * t) : Σ∞
    end
    if γ == 0
        dt = 0.01
        t_end = 4.
    elseif γ==-3
        dt = 1.0
        t_end = 300. # t_end = 40 is used in https://www.sciencedirect.com/science/article/pii/S2590055220300184
    end
    kernel = γ == 0 ? "maxwell" : "coulomb"
    name = σ==1 ? "$(kernel)_landau_$(IC)" : "$(kernel)_landau_$(IC)_cov_$σ"
    
    return (d, n, solver_; kwargs...) -> landau_problem_aux(n, solver_; covariance=covariance, γ=γ, ρ0=ρ0, name=name, dt=dt, t_end=t_end, kwargs...)
end

"Auxillary function to make a landau problem."
function landau_problem_aux(n, solver_; covariance, γ, ρ0, name, dt::F, t_end::F, rng=DEFAULT_RNG, kwargs...) where {F}
    d = length(mean(ρ0))
    @assert(eltype(mean(ρ0)) == typeof(dt))
    t0 = F(0)
    params = (B=F(1 / 24),)
    ρ(t, params) = t ≈ 0 ? ρ0 : MvNormal(mean(ρ0), covariance(999*t_end, params)) # if t > 0, steady-state, only accurate for large t
    f! = landau_f!(d, γ)
    tspan = (t0, t0 + t_end)
    u0 = rand(rng, ρ0, n)
    solver = initialize(solver_, u0, score(ρ0, u0), name; kwargs...)
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver, name, landau_diffusion_coefficient, covariance)
end

############ Helper functions ############

# TODO: this should depend on γ
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
    elseif d == 6
        return (args...) -> landau_6d_f!(args...; γ=γ)
    elseif d == 10
        return (args...) -> landau_10d_f!(args...; γ=γ)
    else
        error("Landau problem with dimension d = $d is not implemented.")
    end
end

landau_2d_f!(du, u, prob, t; γ) = landau_f_aux!(du, u, prob, Val(2); γ=γ)
landau_3d_f!(du, u, prob, t; γ) = landau_f_aux!(du, u, prob, Val(3); γ=γ)
landau_5d_f!(du, u, prob, t; γ) = landau_f_aux!(du, u, prob, Val(5); γ=γ)
landau_6d_f!(du, u, prob, t; γ) = landau_f_aux!(du, u, prob, Val(6); γ=γ)
landau_10d_f!(du, u, prob, t; γ) = landau_f_aux!(du, u, prob, Val(10); γ=γ)

@generated function landau_f_aux!(du, u, prob, ::Val{d}; γ) where {d}
    quote
        ε = eps(eltype(du))
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
