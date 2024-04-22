using Telegram, ConfigEnv

function sendTelegramMessage(message::String = "Calculation finished")
    dotenv() # populate ENV with the data from .env
    TelegramClient()
    sendMessage(text = message)
end