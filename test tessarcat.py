# -*- coding: utf-8 -*-
"""
Created on Thu May  1 21:42:29 2025

@author: rickr
"""

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"D:\tessarcat\tesseract.exe"

img = Image.new("RGB", (300, 100), color=(255, 255, 255))
from PIL import ImageDraw
d = ImageDraw.Draw(img)
d.text((10, 30), "Hello Tesseract!", fill=(0, 0, 0))

print(pytesseract.image_to_string(img))  # 应该输出: Hello Tesseract!
