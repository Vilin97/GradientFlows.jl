using Telegram, Telegram.API, ConfigEnv

function sendTelegramMessage(message::String = "Calculation finished")
    dotenv() # populate ENV with the data from .env
    TelegramClient()
    sendMessage(text = message)
end

# TODO: make this a macro
function trySendTelegramMessage(f_name, f, f_args)
    try
        elapsed = @elapsed f(f_args...)
        sendTelegramMessage("$f_name finished in $elapsed seconds.")
    catch e
        sendTelegramMessage("Error in $f_name.")
        rethrow(e)
    end
end