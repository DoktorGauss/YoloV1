# read the xml file from segmentation
# create a txt file for easier reading

import sys, getopt
import xml.etree.ElementTree as ET
import os

def readXML(path):
    xml_data = []
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(path, filename)
        tree = ET.parse(fullname)
        root = tree.getroot()
        xml_datas = []
        xml_metric = {}
        xml_metric['name'] = filename
        for child in root:
            if child.tag == 'size':
                for metric in child:
                    xml_metric[metric.tag] = metric.text
            if not child.tag == ('object'): continue
            child_xml_data = {}
            for childs in child:
                #if not myClasses(childs.tag): continue
                if not (childs.tag == 'name' or childs.tag == 'bndbox'): continue
                if childs.tag == 'name':
                    child_xml_data[childs.tag] = childs.text
                if childs.tag == 'bndbox':
                    for positions in childs:
                        child_xml_data[positions.tag] = positions.text
            xml_datas.append(child_xml_data)
        xml_data.append({'metric' : xml_metric, 'data' : xml_datas})
    print( path + 'data xml readed')
    return xml_data

# read xml data and convert it to np.array
def readXML_Train_Test( image_folder_path='./data' ):
    train_path = image_folder_path + '/train'
    test_path = image_folder_path + '/test'
    # for each file in files 
    train_xml_data = readXML(train_path)
    test_xml_data = readXML(test_path)
    return train_xml_data, test_xml_data
