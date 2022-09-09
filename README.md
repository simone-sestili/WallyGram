# WallyGram

The purpose of this application is search for wallpapers according to user requests written in natural language. In order to do that the system has a finite dataset of images to choose from, and it integrates a semantic search engine (based on BERT models) in order to find the images that most closely match the meaning of the user request. The user can specify what kind of device they would like the wallpaper for, so that the system will filter the recommended images according to their aspect ratio. The entire application is accessible through a Telegram bot for a user-friendly experience.

# Installation
## Using Python
The application is entirely built in Python 3.9, all the necessary dependencies can be installed by running:
```
pip install -r requirements.txt
```
When doing so it is suggested to create an ad-hoc Python environment with all the necessary dependencies. Then it is required to use the following commands, in order, to download the dataset, create the model embeddings, and finally run the application:
```
python dataloader.py

python train.py

python main.py
```
## Using Docker
The application can be entirely deployed as a ready-for Docker container. The Docker image can be created by simply running the script `helper_make.sh` and then the Docker container can be run by simply running the script `run.sh`.

# Back-end
The entire dataset of images is converted into a list of sentence embeddings, through a CLIP model (namely [clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32)); these embeddings are generated only when the database is updated and then stored in a pickle file.

Once a textual search request is performed, the user query is converted into a sentence embedding using the same model; then the embedding of the query is compared with the embeddings of all the images, and the images are sorted according to decreasing values of cosine similarity between the couples of embeddings. The images whose embeddings have the highest cosine similarity with the query embeddings, are ones whose semantic meaning is the closest to the meaning of the user's request.

The semantic search is powered by a BERT model, using the [sentence-transformers](https://www.sbert.net/) public library; in particular, the model and the dataset used are defined in the application's configuration file at `/app/config.json`.

The whole application will be executed by running `python main.py`, since this will handle the documents preprocessing, the models download, the embeddings pre-computing, the actual search process, and the front-end implementation.

# Front-end
This application has been designed to be used through a custom [Telegram](https://telegram.org/) bot. Please consider that a Telegram bot is a personal property of the owner, therefore whoever wants to run this application by themselves, should be aware of the need to use a personal bot. 

## How to create a bot
Any Telegram user can create a personal bot for free by simply searching *BotFather* on the Telegram search page, executing the aforementioned bot and by following its instructions to create a personal bot. 

When creating a bot it is very important to take note of the chosen `username` and of the `API_KEY` that will be automatically assigned. 

From BotFather it is mandatory to define a list of commands as follows (descriptions are mandatory but can be any text, since their only purpose is to guide the final user):
```
start - description
hint - description
help - description
mobile - description
desktop - description
top_categories - description
```
It is important for the commands to have the same exact syntax defined in the `main()` function of the `/app/main.py` script.

Using *BotFather* it is also possible to change some other features of the bot, such as the name and the profile picture.

*(Please consider that I do not have any affiliations with the owners or creators of BotFather).*

## How to link front-end with back-end
Once the bot has been created, it is necessary to create a new file at `/app/api.key'`, whose only text has to be the personal `API_KEY`. It is not recommended to share this information with anyone else since this will give them access to your bot. The purpose of the `API_KEY` is to connect the back-end with the proper bot on the Telegram platform, this connection is done automatically by the [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) library when `/app/main.py` is run.

## How to use the bot
Once the application is running, the bot can be accessed by searching for its `username` in the Telegram search page, then starting the bot and using the previously defined commands, whose purpose is described in the following:
- `start`: allows to select target device (mobile or desktop);
- `hint`: provides a quick guide on the other commands;
- `help`: provides the developer's contact;
- `mobile`: switches search target to mobile devices;
- `desktop`: switches search target to desktop devices;
- `top_categories`: suggests some possible searches by returning the top-$N$ categories for the images of the target device, where $N$ can be defined by the developer in `/app/config.json`.

Then, once prompted by the bot, the user can search images by simply sending a message to the bot with the sole content of its query, without any additional command. Please consider that, in order to reduce data consumption, all images are given in a compressed format, to download the images in full-resolution the user can simply use the *Hi-Res image* button that will appear below each image.
# Dataset
The dataset is made by 25'000 images sampled from [unsplash.com](https://unsplash.com/), it is commonly used for academic purposes as it includes images from several topics and scenarios.

In particular, for this application it was used and downloaded the *Lite Dataset* specified in the official unsplash repository at [github.com/unsplash/datasets](https://github.com/unsplash/datasets).
