from imageReader import readImages_Train_Test
from rotate_to_ordinal_line import rotateImageToOrdinalLine


train_image_data, test_image_data = readImages_Train_Test()


for train_image in train_image_data:
    img = rotateImageToOrdinalLine(train_image)
    img.save(img.name + 'rotated.JPG')