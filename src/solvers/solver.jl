abstract type Solver end

# initialize(solver, u0, score_values, problem_name) initialize the solver with the score values
# update!(solver, integrator) fill in solver.score_values

function Base.show(io::IO, solver::S) where {S<:Solver}
    Base.print(io, "$(typeof(solver))")
end

reset!(solver::Solver, u0, score_values) = (solver.score_values .= score_values; nothing)

"Log values of interest throughout the simulation."
struct Logger{S}
    log_level::Int
    score_values::Vector{S}
end

Logger(log_level, score_values::S) where {S} = Logger(log_level, S[])
Logger(log_level) = Logger(log_level, nothing)
Logger() = Logger(0, nothing)

function log!(logger::Logger, solver::Solver)
    if logger.log_level > 0
        push!(logger.score_values, solver.score_values)
    end
    nothing
end