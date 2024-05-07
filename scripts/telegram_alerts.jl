using Telegram, Telegram.API, ConfigEnv

function sendTelegramMessage(message::String = "Calculation finished")
    dotenv() # populate ENV with the data from .env
    TelegramClient()
    sendMessage(text = message)
    nothing
end

"Usage: @trySendTelegramMessage f(args...)"
macro trySendTelegramMessage(expr)
    quote
        function_str = string($(Expr(:quote, expr)))
        try
            elapsed = @elapsed begin
                local value = $(esc(expr))
                value
            end
            sendTelegramMessage("$function_str finished in $(convert_time_to_string(elapsed)).")
        catch e
            sendTelegramMessage("Error in $function_str.")
            rethrow(e)
        end
    end
end

function convert_time_to_string(seconds)
    int_seconds = round(Int, seconds)
    
    result = ""
    (int_seconds ÷ 86400) > 0 && (result *= "$(int_seconds ÷ 86400) day(s) ")
    (int_seconds %= 86400; int_seconds ÷ 3600) > 0 && (result *= "$(int_seconds ÷ 3600) hour(s) ")
    (int_seconds %= 3600; int_seconds ÷ 60) > 0 && (result *= "$(int_seconds ÷ 60) minute(s) ")
    (int_seconds %= 60) > 0 && (result *= "$int_seconds second(s)")
    
    return result
end
