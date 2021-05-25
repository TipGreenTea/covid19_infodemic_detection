"""
Google Transalte: https://pypi.org/project/googletrans/
Thai Sentence Segmentation: https://www.bualabs.com/archives/3740/python-word-tokenize-pythainlp-example-algorithm-deepcut-newmm-longest-python-pythainlp-ep-2/
"""

from google_trans_new import google_translator   
import warnings
from pythainlp import sent_tokenize, word_tokenize
warnings.filterwarnings('ignore')


translator = google_translator()  
translate_text = translator.translate('Hola mundo!', lang_src='es', lang_tgt='th')  
print(translate_text)

text = "เมืองเชียงรายมีประวัติศาสตร์อันยาวนาน        เป็นที่ตั้งของหิรัญนครเงินยางเชียงแสน"
print("sent_tokenize:", sent_tokenize(text))
print("word_tokenize:", word_tokenize(text))
print("no whitespace:", word_tokenize(text, keep_whitespace=False))
#-------Thai segmentation methods--------#
print("1. newmm    :", word_tokenize(text))  # default engine is "newmm"

print("2. longest  :", word_tokenize(text, engine="longest"))
print("3. multi_cut:", word_tokenize(text, engine="multi_cut"))
print("4. deepcut  :", word_tokenize(text, engine="deepcut"))

print("no whitespace:", word_tokenize('การสวมหน้ากากอนามัยสามารถป้องกันคุณจากไวรัสได้', keep_whitespace=False))
print("no whitespace:", word_tokenize('ถ้าคุณทาน Crocin สามครั้งต่อวันคุณจะปลอดภัย', keep_whitespace=False))