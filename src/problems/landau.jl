"Make a homogeneous landau problem with Maxwell kernel with the given dimension, number of particles, and solver."
function landau_problem(d, n, solver_; dt::F=0.01, rng=DEFAULT_RNG, kwargs...) where {F}
    if solver_ isa SBTM && F == Float64
        return landau_problem(d, n, solver_; dt=Float32(dt), rng=rng, kwargs...)
    end
    params = LandauParams(d, F(1 / 24))
    t0_ = t0(params)
    f! = choose_f!(d)
    tspan = (t0_, t0_ + F(0.5))
    ρ(t, params) = PolyNormal(d, params.K(t))
    ρ0 = ρ(t0_, params)
    u0 = rand(rng, ρ0, n)
    name = "landau"
    solver = initialize(solver_, u0, score(ρ0, u0), name; kwargs...)
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver, name)
end

struct LandauParams{T,F}
    d::Int # dimension
    B::T
    C::T
    K::F   # variance of MvNormal
end
LandauParams(d, B::T, C=one(T)) where {T} = LandauParams(d, B, C, t -> 1 - C * exp(-(d - 1) * 2 * B * t))

"Choose the starting time `t0` so that P ≈ 0 and P ≥ 0."
t0(params::LandauParams) = round(log((params.d + 2) * params.C / 2) / (2params.B * (params.d - 1)), RoundUp, digits=1)

# f! for different dimensions
function choose_f!(d)
    if d == 3
        return landau_3d_f!
    elseif d == 5
        return landau_5d_f!
    elseif d == 10
        return landau_10d_f!
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
    du .*= prob.params.B / n
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
    du .*= prob.params.B / n
    nothing
end
function landau_10d_f!(du, u, prob, t)
    s = prob.solver.score_values
    du .= 0
    n = size(u, 2)
    @tturbo for p = 1:n
        Base.Cartesian.@nexprs 10 i -> dx_i = zero(eltype(du))
        for q = 1:n
            dotzv = zero(eltype(du))
            normsqz = zero(eltype(du))
            Base.Cartesian.@nexprs 10 i -> begin
                z_i = u[i, p] - u[i, q]
                v_i = s[i, q] - s[i, p]
                dotzv += z_i * v_i
                normsqz += z_i * z_i
            end
            Base.Cartesian.@nexprs 10 i -> begin
                dx_i += v_i * normsqz - dotzv * z_i
            end
        end
        Base.Cartesian.@nexprs 10 i -> begin
            du[i, p] += dx_i
        end
    end
    du .*= prob.params.B / n
    nothing
end
