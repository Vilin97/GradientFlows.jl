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
            sendTelegramMessage("$function_str finished in $elapsed seconds.")
        catch e
            sendTelegramMessage("Error in $function_str.")
            rethrow(e)
        end
    end
end