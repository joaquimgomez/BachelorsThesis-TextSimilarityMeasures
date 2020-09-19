import argparse
from tika import parser as tike_parser
from os import listdir, mkdir
from os.path import isfile, join

def PDFtoTXT(path):
	rawFile = tike_parser.from_file(path)
	output = str(rawFile['content'])

	#output = output.encode('utf-8', errors='ignore')

	#output = output.replace('\n', '')

	return output

def main(org, dest):
	# Obtain all the folders
	folders = [(folder, join(org, folder)) for folder in listdir(org) if not isfile(join(org, folder)) and not folder.startswith('.')]

	mkdir(dest)

	numFolders = 0
	numFiles = 0
	for (folderName, path) in folders:
		print("Converting files from folder " + folderName + ".")
		mkdir(dest + "/" + folderName + "/")

		files = [(file, join(path, file)) for file in listdir(path) if  not file.startswith('.')] # isfile(join(path, file)) and

		for (file, filePath) in files:
			txtFile = PDFtoTXT(filePath)

			with open(dest + "/" + folderName + "/" + file + ".txt", 'w') as newTXTFile:
				newTXTFile.write(str(txtFile))

			numFiles = numFiles + 1

		numFolders = numFolders + 1

	print("\n")
	print("Number of processed folders: " + str(numFolders))
	print("Number of processed files: " + str(numFiles))
	print("\n")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Creates txt files from pdf files in the input directory.")
	parser.add_argument("--origen", "-o",
						help = "Directory of original pdf files",
						default = "./")
	parser.add_argument("--destination", "-d",
						help = "Directory where txt files goes. The destination folder must not exist.",
						default = "./")

	args = parser.parse_args()
	main(args.origen, args.destination)
