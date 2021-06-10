import os
import re

def cleaner(file_name, person_name, enc='UTF-8'):
	with open(file_name, 'r', encoding=enc) as file:
		data = file.read()
	data = data.lower()
	data = trim_tool(r'<.(\d*?)', data)
	data = trim_tool(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', data)
	data = re.findall('{.*?m":"(.*?)"', data)
	data = '\n'.join(data)
	data = data.replace('\\', '')
	data = data.replace('<', '')
	data = data.split('\n')
	max = 422
	for i in data:
		if len(i) < max:
			i = i + (' ' * (max - len(i)))
		if len(i) < 10:
			if i in data:
				data.remove(i)
	data = '\n'.join(data)
	print(max)
	# New file creation
	with open(person_name + '.txt', 'w', encoding=enc) as file:
		file.write(data)

# Can be fed a regular expression to trim the file.
def trim_tool(regex_string, data):
	data = re.split(regex_string, data)
	data = '\n'.join(data)
	return data


if __name__ == '__main__':
	cleaner('dht-filtered.txt', 'michael')