import os.path
import string
import xml.etree.ElementTree as ET # xml reading
import xmltodict
import pickle

''' 
using the dataset at:
https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database
specifically 
https://fki.tic.heia-fr.ch/DBs/iamOnDB/data/original-xml-all.tar.gz

goal: grab strokes
should normalize positional and time data:

x-list of strokes
- each index contains a list of x^(i) stroke points
y-list of strings
- each index is the correct text
'''

# x[i] is a list of strokes, which are lists of points (themselves a size 3 list)
x = []
y = []
char_to_index = {
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 
    'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 
    'x': 24, 'y': 25, 'z': 26,  # Lowercase letters

    'A': 27, 'B': 28, 'C': 29, 'D': 30, 'E': 31, 'F': 32, 'G': 33, 'H': 34, 'I': 35, 'J': 36, 'K': 37, 
    'L': 38, 'M': 39, 'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 'U': 47, 'V': 48, 
    'W': 49, 'X': 50, 'Y': 51, 'Z': 52,  # Uppercase letters

    '0': 53, '1': 54, '2': 55, '3': 56, '4': 57, '5': 58, '6': 59, '7': 60, '8': 61, '9': 62,  # Digits

    ',': 63, '.': 64, "\'": 65, '-': 66,  # Special characters: comma, period, apostrophe, hyphen
    '\"': 67,  ':': 68, '#': 69, '?':70, ';':71, '(': 72, ')': 73, '!': 74, # Space 

    'blank': 0  # Blank token for CTC
}

def parse_xml_dict(xml_dict):
     # strokes is a list containing all strokes, divided into metadata and points
    strokes = xml_dict['WhiteboardCaptureSession']['StrokeSet']['Stroke']
    all_stroke_data = []
    for i in range(len(strokes)):
        stroke = strokes[i]
        if isinstance(stroke['Point'], dict): # handles strokes with a single point
            stroke['Point'] = [stroke['Point']]
        
        stroke_data = []
        for point in stroke['Point']:
            stroke_data.append([point['@x'], point['@y'], point['@time']])
        
        all_stroke_data.append(stroke_data)

    x.append(all_stroke_data)
    y.append(xml_dict['WhiteboardCaptureSession']['Transcription']['Text'])
    if '\#' in y[-1]:
        print(y[-1])
    while '&quot;' in y[-1]:
        y[-1] = y[-1].replace('&quot;', '\'')
    while '&apos;' in y[-1]:
        y[-1] = y[-1].replace('&apos;', '\'')
    while ' ' in y[-1]:
        y[-1] = y[-1].replace(' ', '')
    while '\n' in y[-1]:
        y[-1] = y[-1].replace('\n', '')
    y[-1] = [char_to_index[char] if char in char_to_index.keys() else None for char in y[-1]]

if __name__ == "__main__":
    flag = False
    # using IAM database
    # writer_id ranges from a01 to z01

    for writer_id_char in list(string.ascii_lowercase):
        if writer_id_char == 'z':
            break
        print(writer_id_char)
        for writer_id_num in range(0, 11):
            print(writer_id_num)
            padded_writer_id_num = "{:02}".format(writer_id_num)
            for i in range(1000):
                padded_num = "{:03}".format(i)
                orig_path = "IAM/original/" + f"{writer_id_char}{padded_writer_id_num}/" + f"{writer_id_char}{padded_writer_id_num}-" + padded_num

                if os.path.isfile(orig_path + '/strokesz.xml'):
                    xmlTree = ET.parse(orig_path + '/strokesz.xml')
                    root = xmlTree.getroot()
                    xml_string = ET.tostring(root, encoding='unicode')
                    xml_dict = xmltodict.parse(xml_string)
                    
                    parse_xml_dict(xml_dict)
    print(len(x))
    print(len(y))
    
    xmod=[]
    ymod = []
    for i in range(1560):
        if len(x[i]) >= len(y[i]):
            xmod.append(x[i])
            ymod.append(y[i])
    
    print(len(xmod))
    print(len(ymod))
    with open('x_data.pkl', 'wb') as file:
        pickle.dump(xmod, file)
        file.close()
    with open('y_data.pkl', 'wb') as file:
        pickle.dump(ymod, file)
        file.close()
