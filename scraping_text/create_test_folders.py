#Colin Crovella 2020

import os
import re
import random

english_folder = "english_text_samples/"
french_folder = "french_text_samples/"
scots_folder = "scots_text_samples/"

mode = "scots" #Can be English, French, or Scots

folder_path = ""

if (mode == "english"):
    folder_path = english_folder
elif (mode == "french"):
    folder_path = french_folder
elif (mode == "scots"):
    folder_path = scots_folder

#Gets the filenames of all the text files in the given directory
text_filenames = os.listdir(folder_path)


regex = re.compile('[^a-z\s]')
sentences = []

for filename in text_filenames:
    file = open(folder_path+filename,"r")
    lines = file.readlines()
    #Remove whitespace or placeholder lines
    for line in lines:
        if (line[0] == '\n'):
            lines.remove(line)
        if (line[0] == '#'):
            lines.remove(line)
            
    #turn lowercase
    for i in range(len(lines)):
        lines[i] = lines[i].lower()

    #Create list of sentences
    for line in lines:
        group = re.split("[.!?:;]",line)
        for item in group:
            if (len(item) > 5):
                sentences.append(item)

    for i in range(len(sentences)):
        sentences[i] = regex.sub('',sentences[i])
        if (sentences[i][0] == ' '):
            sentences[i] = sentences[i][1:]

def space_out_string(string):
    outstring = ""
    for character in string:
        if (character == " "):
            outstring += "_"
        else:
            outstring += character
        outstring += " "
    return outstring

i=0
j=1
while (i < len(sentences)-3):
    outfile = open(mode+"/"+str(j)+".txt", "w")
    outstring = ""
    outstring += space_out_string(sentences[i])
    i+=1
    outstring += space_out_string(sentences[i])
    i+=1
    outstring += space_out_string(sentences[i])
    i+=1
    outstring += space_out_string(sentences[i])
    outfile.write(outstring)
    outfile.close()
    i+=1
    j+=1

