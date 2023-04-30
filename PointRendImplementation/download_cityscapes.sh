#!/bin/bash

# Login and download Cityscapes dataset packages
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=AditSharma&password=Adit@2801&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1

# Create VisualAid directory and extract dataset files
mkdir VisualAid
unzip /content/leftImg8bit_trainvaltest.zip -d ./VisualAid/
unzip /content/gtFine_trainvaltest.zip -d ./VisualAid/

