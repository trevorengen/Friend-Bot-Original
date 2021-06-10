import os
import discord
from dotenv import load_dotenv
import brain
import re
from brain import build_model, generate_one_step, send_message


load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

client = discord.Client()

@client.event
async def on_ready():
	print(f'{client.user} has connected to Discord.')

@client.event
async def on_message(message):
	if message.author == client.user:
		return

	# If message content starts with required string will attempt to
	# create a response using the keras OneStep custom model.
	if message.content.startswith('!AMichael'):
		clean_string = re.split('!AMichael', message.content)
		message_to_send = send_message(clean_string, 30)
		await message.channel.send(message_to_send)
model = brain.model
build_model(model)
client.run(TOKEN)