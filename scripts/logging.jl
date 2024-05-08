using Logging, Dates, LoggingExtras, GradientFlows

"Usage: @log f(args...)"
macro log(expr)
    quote
        io = open(joinpath("data", "logs", "global.log"), "a")
        filter_fun(args) = args.level > Logging.Debug
        console_logger = EarlyFilteredLogger(filter_fun, ConsoleLogger())
        file_logger = EarlyFilteredLogger(filter_fun, FileLogger(io))
        logger = TeeLogger(console_logger, file_logger)
        with_logger(logger) do
            @info now()
            local value = $(esc(expr))
            value
        end
        Base.close(io)
    end
end