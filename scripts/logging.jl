using Logging, Dates, LoggingExtras

"Usage: @log f(args...)"
macro log(expr)
    quote
        io = open(joinpath("data", "logs", "global.log"), "a")
        logger = TeeLogger(global_logger(), FileLogger(io))
        with_logger(logger) do
            @info now()
            local value = $(esc(expr))
            value
        end
        Base.close(io)
    end
end