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
        push!(logger.score_values, copy(solver.score_values))
    end
    nothing
end