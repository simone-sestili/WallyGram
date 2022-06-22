import os
import json
from turtle import up

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import *


from train import train_pipeline
from sbert import load_data, load_model, load_embeddings, text2image


PROJECT_CONFIG = 'config_unsplash.json'


# ========== INITIALIZATION ==========

print('Initialization...')
config = json.load(open(PROJECT_CONFIG))
config['embeddings']['path'] = os.path.join(config['data']['folder'], config['embeddings']['filename'])
processes = {
    'ready_for_input': False,
    'device_selected': '',
    'result_images': []
}

# if all files generated with train.py are not present then it has to be run
if config['embeddings']['filename'] not in os.listdir(config['data']['folder']) or config['top_categories']['filename'] not in os.listdir(config['data']['folder']):
    print(train_pipeline())

folder_file_path = load_data(
    data_folder=config['data']['folder'],
    filename=config['data']['filename'],
    download_url=config['data']['download_url']
)

model = load_model(model_folder=config['model']['folder'], model_name=config['model']['name'])

image_names, image_embeddings = load_embeddings(
    data=folder_file_path,
    model=model,
    embeddings_path=config['embeddings']['path'],
    use_precomputed=config['embeddings']['use_precomputed_embeddings'],
    download_url=config['embeddings']['download_url'],
    batch_size=config['embeddings']['batch_size'],
)


# ========== TELEGRAM COMMAND FUNCTIONS ==========

def start_command(update, context):
    keyboard = [[
        InlineKeyboardButton("\U0001F4F1 Mobile", callback_data='mobile'),
        InlineKeyboardButton("\U0001F4BB Desktop", callback_data='desktop'),
    ]]
    update.message.reply_text(
        f"Hi, I'm {config['bot_name']}, I will help you find a beautiful wallpaper!\nPlease select the device you are searching a wallpaper for.", 
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


def hint_command(update, context):
    update.message.reply_text('''This bot is designed for searching wallpapers according to some keywords, and using semantic similarity.\n
First off, please select your device using /mobile or /desktop.\n
Then, simply text me what you are searching for, if you like my suggestion you can download it in high-resolution, otherwise just ask me for another similar image, or try with another search.\n
If you need inspiration, just use /top_categories to see the most searched images.\n
Use /help if you need further assistance.
''')


def help_command(update, context):
    update.message.reply_text('If you need help, contact @Simon_6')


def mobile_command(update, context):
    processes['ready_for_input'] = True
    processes['device_selected'] = 'mobile'
    update.message.reply_text('Text me something to search a wallpaper for mobile.\nIf you want to switch to desktop you can simply hit /desktop at any time.')

    
def desktop_command(update, context):
    processes['ready_for_input'] = True
    processes['device_selected'] = 'desktop'    
    update.message.reply_text('Text me something to search a wallpaper for desktop.\nIf you want to switch to mobile you can simply hit /mobile at any time.')


def top_categories_command(update, context):
    top_categories = json.load(open(os.path.join(config['data']['folder'], config['top_categories']['filename'])))
    update.message.reply_text('The top-{} most searched categories are:\n\n{}'.format(len(top_categories), '\n'.join(top_categories)))


def show_image(update, context):
    if len(processes['result_images']) == 0:
        update.message.reply_text('No more images to show. Please try with another search.')
    else:
        # get image path
        image_corpus_id = processes['result_images'].pop(0)['corpus_id']
        image_path = os.path.join(folder_file_path, image_names[image_corpus_id])
        print(image_path)
        # show low-res image
        chat_id = update.message.chat_id
        context.bot.send_photo(chat_id=chat_id, photo=open(image_path,'rb'))
        # show buttons for new download and another image
        keyboard = [[
            InlineKeyboardButton("\U0001F4F1 Mobile", callback_data='mobile'),
            InlineKeyboardButton("\U0001F4BB Desktop", callback_data='desktop'),
        ]]
        update.message.reply_text(
            f"Hi, I'm {config['bot_name']}, I will help you find a beautiful wallpaper!\nPlease select the device you are searching a wallpaper for.", 
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        # ready to accept new searches
        processes['ready_for_input'] = True


def handle_message(update, context):
    user_text = str(update.message.text).lower()
    print('user_text', user_text)
    if processes['ready_for_input']:
        processes['ready_for_input'] = False
        processes['result_images'] = text2image(
            query=user_text,
            image_names=image_names,
            image_embeddings=image_embeddings,
            folder_file_path=folder_file_path,
            img_model=model,
            top_k=config['text2image']['top_k'],
            DEBUG=False
        )
        show_image(update, context)
        

def button(update, context):
    # Parses the CallbackQuery when the feedback button is clicked
    query = update.callback_query
    query.answer()  # CallbackQueries need to be answered, even if no notification to the user is needed
    # mobile btn
    if query.data == 'mobile':
        processes['ready_for_input'] = True
        processes['device_selected'] = 'mobile'
        # edit message without buttons
        query.edit_message_text(f"Hi, I'm {config['bot_name']}, I will help you find a beautiful wallpaper!\nPlease select the device you are searching a wallpaper for.")
        # append new message
        chat_id = query.message.chat_id
        context.bot.send_message(chat_id=chat_id, text='Text me something to search a wallpaper for mobile.\nIf you want to switch to desktop you can simply hit /desktop at any time.')
    # desktop btn
    elif query.data == 'desktop':
        processes['ready_for_input'] = True
        processes['device_selected'] = 'desktop'
        # edit message without buttons
        query.edit_message_text(f"Hi, I'm {config['bot_name']}, I will help you find a beautiful wallpaper!\nPlease select the device you are searching a wallpaper for.")
        # append new message
        chat_id = query.message.chat_id
        context.bot.send_message(chat_id=chat_id, text='Text me something to search a wallpaper for desktop.\nIf you want to switch to mobile you can simply hit /mobile at any time.')
        

def error(update, context):
    print(f'Update {update} caused error {context}')


def main():
    updater = Updater(config['API_KEY'], use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start_command))
    dp.add_handler(CommandHandler('hint', hint_command))
    dp.add_handler(CommandHandler('help', help_command))
    dp.add_handler(CommandHandler('mobile', mobile_command))
    dp.add_handler(CommandHandler('desktop', desktop_command))
    dp.add_handler(CommandHandler('top_categories', top_categories_command))

    dp.add_handler(MessageHandler(Filters.text, handle_message))

    updater.dispatcher.add_handler(CallbackQueryHandler(button))

    dp.add_error_handler(error)

    updater.start_polling()
    updater.idle()


print('Bot started...')
main()
