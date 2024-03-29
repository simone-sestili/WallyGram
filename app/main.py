import os
import json

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import *

from train import train_pipeline
from dataloader import get_data_zipweb, mobile_desktop_split
from sbert import load_model, load_embeddings, text2image


PROJECT_CONFIG = 'config_unsplash.json'


# ========== INITIALIZATION ==========

print('Initialization...')

with open('api.key') as f:
    API_KEY = f.read()
config = json.load(open(PROJECT_CONFIG))

# load device-independent model
model = load_model(model_folder=config['model']['folder'], model_name=config['model']['name'])

embeddings = {}
for device in ['mobile', 'desktop']:
    
    # os-independent creation of paths
    for filetype in ['images_folder', 'embeddings', 'top_categories', 'classification_labels']:
        config[device][filetype]['path'] = os.path.join(config['data_folder'], config[device][filetype]['filename'])
        
    # if explicitly specified in the the configuration file, refresh the dataset of images
    if config['refresh_images']:
        # download zip and extract all images
        images_all_folder_path = get_data_zipweb(config, 'unsplash-25k-photos.zip')
        # split by device
        mobile_desktop_split(config, images_all_folder_path)
        # new data requires to train again the embeddings
        print(f'> {device} pipeline execution...')
        print(train_pipeline(model, config[device]))
    
    # if not all files generated with train.py are present then the embeddings creation pipeline has to be run
    if config[device]['embeddings']['filename'] not in os.listdir(config['data_folder']) or config[device]['top_categories']['filename'] not in os.listdir(config['data_folder']):
        print(f'> {device} pipeline execution...')
        print(train_pipeline(model, config[device]))

    # load local variables for model and embeddings
    embeddings[device] = load_embeddings(
        data=config[device]['images_folder']['path'],
        model=model,
        embeddings_path=config[device]['embeddings']['path'],
        use_precomputed=config[device]['embeddings']['use_precomputed_embeddings'],
        batch_size=config[device]['embeddings']['batch_size'],
        data_type='image'
    )


# initialize processes status with default values
processes = {
    'ready_for_input': True,
    'device_selected': 'mobile',
    'result_images': [],
    'current_image_path': ''
}



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
    update.message.reply_text('If you need help, contact @{}'.format(config['developer_telegram_username']))


def mobile_command(update, context):
    processes['ready_for_input'] = True
    processes['device_selected'] = 'mobile'
    update.message.reply_text('Text me something to search a wallpaper for mobile.\nIf you want to switch to desktop you can simply hit /desktop at any time.')

    
def desktop_command(update, context):
    processes['ready_for_input'] = True
    processes['device_selected'] = 'desktop'    
    update.message.reply_text('Text me something to search a wallpaper for desktop.\nIf you want to switch to mobile you can simply hit /mobile at any time.')


def top_categories_command(update, context):
    device_selected = processes['device_selected']
    top_categories = json.load(open(config[device_selected]['top_categories']['path'], encoding='utf-8'))
    update.message.reply_text('The top-{} most searched categories for {} are:\n\n{}'.format(len(top_categories), device_selected, '\n'.join(top_categories)))


def show_image(update, context, chat_id: str):
    if len(processes['result_images']) == 0:
        update.message.reply_text('No more images to show. Please try with another search.')
    else:
        # get image path
        device_selected = processes['device_selected']
        image_corpus_id = processes['result_images'].pop(0)['corpus_id']
        processes['current_image_path'] = embeddings[device_selected][0][image_corpus_id]
        # show low-res image and buttons for download and another image
        button_matrix = [[
            InlineKeyboardButton("\U0001F5BC Hi-Res image", callback_data='download'),
            InlineKeyboardButton("\U0001F504 Another image", callback_data='another'),
        ]]
        context.bot.send_photo(chat_id=chat_id, photo=open(processes['current_image_path'],'rb'), reply_markup=InlineKeyboardMarkup(button_matrix))
        # ready to accept new searches
        processes['ready_for_input'] = True


def handle_message(update, context):
    user_text = str(update.message.text).lower()
    if processes['ready_for_input']:
        processes['ready_for_input'] = False
        device_selected = processes['device_selected']
        processes['result_images'] = text2image(
            query=user_text,
            image_names=embeddings[device_selected][0],
            image_embeddings=embeddings[device_selected][1],
            folder_file_path=config[device_selected]['images_folder']['path'],
            img_model=model,
            top_k=config['text2image']['top_k'],
            DEBUG=False
        )
        show_image(update, context, chat_id=update.message.chat_id)
        

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
    # download hi-res image
    elif query.data == 'download':
        chat_id = query.message.chat_id
        context.bot.send_document(chat_id=chat_id, document=open(processes['current_image_path'], 'rb'))
    # show another image
    elif query.data == 'another':
        chat_id = query.message.chat_id
        show_image(update, context, chat_id=chat_id)


def error(update, context):
    print(f'Update {update} caused error {context}')


def main():
    updater = Updater(API_KEY, use_context=True)
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
