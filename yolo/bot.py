from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext, CallbackQueryHandler, filters
from os import environ, remove
import uuid
from queue import Queue
from yolo_net import SeaCrapNet
import asyncio
import logging

# BOT_API = '7976452257:AAHkwBawzsKpGUWYXXnrzcNO14TGtE7n2lk'

class Telegram_bot:
    def __init__(self):
        self.api = '7976452257:AAHkwBawzsKpGUWYXXnrzcNO14TGtE7n2lk'
        self.model = SeaCrapNet()
        self.queue = Queue()
        
    def setup_logging(self):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def start(self, update: Update, context: CallbackContext):
        user = update.effective_user
        message = f"Привет, {user.first_name}, отправь фотографию!"
        
        await update.message.reply_text(message)
        
    async def handle_message(self, update: Update, context: CallbackContext):
        await update.message.reply_text("Отправь фотографию")
        
    async def save_photo(self, user, file):
        unique_id = str(uuid.uuid4())
        file_name = f"{user.id}_{unique_id}.jpg"
        await file.download_to_drive(custom_path=file_name)
        return file_name
    
        
    async def handle_photo(self, update: Update, context: CallbackContext):
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        user = update.effective_user
        file_name = await self.save_photo(user=user, file=file)
        self.queue.put(file_name)
        await update.message.reply_text("Подождите, идёт обработка...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.model.pipeline, file_name)
        await context.bot.send_photo(chat_id=update.message.chat_id, photo=file_name)
        # context.bot.send_text()
        msg = 'Найденный мусор по классам:\n\n'
        for class_, amount in self.model.detected.items():
            if amount > 0:
                msg = msg + f'{class_}: {amount}\n'
        if msg == 'Найденный мусор по классам:\n\n':
            msg = 'Мусор не найден'
        await context.bot.send_message(chat_id=update.message.chat_id, text=msg)
        remove(file_name)
        
        
        
        
    def run(self):
        self.setup_logging()
        application = Application.builder().token(self.api).read_timeout(30).write_timeout(600).build()
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(MessageHandler(filters=filters.PHOTO, callback=self.handle_photo))
        application.add_handler(MessageHandler(filters=None, callback=self.handle_message))
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
        
if __name__ == "__main__":
    bot = Telegram_bot()
    bot.run()
