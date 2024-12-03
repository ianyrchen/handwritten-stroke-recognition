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
    while '&quot;' in y[-1]:
        y[-1] = y[-1].replace('&quot;', '\'')
    while '&apos;' in y[-1]:
        y[-1] = y[-1].replace('&apos;', '\'')
    while ' ' in y[-1]:
        y[-1] = y[-1].replace(' ', '')
    while '\n' in y[-1]:
        y[-1] = y[-1].replace('\n', '')

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
    

    with open('x_data.pkl', 'wb') as file:
        pickle.dump(x, file)
        file.close()
    with open('y_data.pkl', 'wb') as file:
        pickle.dump(y, file)
        file.close()