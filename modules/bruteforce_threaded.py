from functools import reduce
from PIL import Image
import numpy as np
from collections import Counter
import string
from concurrent.futures import ThreadPoolExecutor
import threading

def run(filename, treshold, max_workers=4):
    image = Image.open(filename)
    image = image.convert("RGB")
    lock = threading.Lock()
    
    def process_e(e):
        if e is not None:
            score1 = character_distribution_score(e)
            score2 = doubles_score(e)
            if score1 > treshold and score2 > treshold:
                with lock:
                    print("Potential text detected:", e, "scores:", score1, score2)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for a in BlockIter1(image):
            for b in OrderIter2(a):
                for c in ColorIter3(b):
                    for d in BitsIter4(c):
                        for e in DecodingIter5(d):
                            executor.submit(process_e, e)
    
    print("Bruteforce finished")

class BlockIter1:
  def __init__(self,image):
    self.a = 0
    self.img = np.array(image)
    self.hor, self.wer = image.size
    self.h = (self.hor // 3) * 3
    self.w = (self.wer // 3) * 3
    self.i = 0
    self.j = 0
    return None
  
  def __iter__(self):
    return self

  def __next__(self):
    print("iter1:",self.a)
    self.a += 1
    if self.a < 6:
        match self.a:
            case 1:
                return self.img
            case 2:
                return self.img[:self.hor//2,:self.wer//2]
            case 3:
                return self.img[self.hor//2:,:self.wer//2]
            case 4:
                return self.img[:self.hor//2,self.wer//2:]
            case 5:
                return self.img[self.hor//2:,self.wer//2:]
    else:
       raise StopIteration
       if self.j >= self.w: raise StopIteration
       res = self.img[self.i:self.i+3, self.j:self.j+3]
       self.i += 3
       if self.i >= self.w:
          self.i = 0
          self.j += 3
       return res

class OrderIter2:
   def __init__(self,array):
      self.arr = array
      self.i = 0
   def __iter__(self):
      return self
   def __next__(self):
      self.i += 1
      if self.i == 1:
         return self.arr.reshape(self.arr.shape[0] * self.arr.shape[1], self.arr.shape[2])
      elif self.i == 2:
         return self.arr.transpose(1,0,2).reshape(self.arr.shape[0] * self.arr.shape[1], self.arr.shape[2])
      else:
         raise StopIteration
         
class ColorIter3:
   def __init__(self,array):
      self.arr = array
      self.i = 0
   def __iter__(self):
      return self
   def __next__(self):
      self.i += 1
      if self.arr.shape[1] == 3:
        match self.i:
            case 1:
                return self.arr.reshape(-1)
            case 2:
                return np.concatenate([self.arr[:,0], self.arr[:,1],self.arr[:,2]])
            case 3:
                raise StopIteration
      else:
        match self.i:
            case 1:
                return self.arr.reshape(-1)
            case 2:
                return self.arr[:, :3].reshape(-1)
            case 3:
                return np.concatenate([self.arr[:,0], self.arr[:,1],self.arr[:,2]])
            case 4:
                return np.concatenate([self.arr[:,0], self.arr[:,1],self.arr[:,2],self.arr[:,3]])
            case 5:
              raise StopIteration

class BitsIter4:
   def __init__(self,array):
      self.arr = array
      self.i = 0
   def __iter__(self):
      return self
   def __next__(self):
      self.i += 1
      match self.i:
         case 1:
            return self.arr & 1
         case 2:
            return np.concatenate([self.arr & 1 , (self.arr >> 1) & 1])
         case 3:
            return np.concatenate([(self.arr >> 1) & 1, self.arr & 1])
         case 4:
            return np.column_stack([(self.arr >> 1) & 1, self.arr & 1 ]).flatten()
         case 5:
            return np.column_stack([self.arr & 1, (self.arr >> 1) & 1 ]).flatten()
         case 6:
            raise StopIteration

class DecodingIter5:
   def __init__(self,array):
      self.arr = array
      self.mismatch = len(self.arr) % 8
      self.i = 0
      self.j = 0
   def __iter__(self):
      return self
   def __next__(self):
      
      if self.i < 2:
         
         bytes_list = []
         for i in range(self.j, len(self.arr), 8):
            byte_bits = self.arr[i:i+8]
            if len(byte_bits) < 8:
                break
            byte = np.packbits(byte_bits)[0]
            bytes_list.append(byte)
        
         byte_data = bytes(bytes_list)
         self.j += 1

         if self.j >= self.mismatch:
            self.i += 1
            self.j = 0

         try:
            if self.i == 0:
                return byte_data.decode('utf-8')
            else:
               return byte_data.decode('ascii')
         except UnicodeDecodeError:
            return None
      else:
         raise StopIteration


def character_distribution_score(text):
    english_freq = 'etaoinshrdlcumwfgypbvkjxqz'
    text_chars = [c.lower() for c in text if c.isalpha()]
    if not text_chars:
        return 0
    most_common = [c for c, _ in Counter(text_chars).most_common()]
    # Compare to English frequency
    matches1 = sum(1 for i, c in enumerate(most_common[:6]) if c in english_freq[:6])
    matches2 = sum(1 for i, c in enumerate(most_common[:12]) if c in english_freq[:12])
    return ((matches1 / 6)/2) + ((matches2 / 12)/ 2)

def doubles_score(text):
    common_pairs = ["th" , "ar" ,"he" , "te" ,"an" , "se" ,"in" , "me" ,"er" , "sa" ,"nd" , 
                    "ne" ,"re" , "wa" ,"ed" , "ve" ,"es" , "le" ,"ou" , "no" ,"to" , "ta" ,
                    "ha" , "al" ,"en" , "de" ,"ea" , "ot" ,"st" , "so" ,"nt" , "dt" ,"on" ,
                    "ll" ,"at" , "tt" ,"hi" , "el" ,"as" , "ro" ,"it" , "ad" ,"ng" , "di" ,
                    "is" , "ew" ,"or" , "ra" ,"et" , "ri" ,"of" , "sh" ,"ti", "am", "i ",
                    " i", " t", "t " "e ", " e", " a", "m ", " o", "f ", " w", " s", "we", 
                    "s ", "do", "if", "my", "\'s", "by", "ho", ", ", ". ", "? ", "! ", ".."]
    text = text.lower()
    
    all_pairs = 0
    valid_pairs = 0
    for i in range(len(text)-1):
       all_pairs += 1
       if str(text[i:i+2]) in common_pairs:
          valid_pairs += 1

    return valid_pairs/ all_pairs if valid_pairs > 0 and all_pairs > 0 else 0.0



if __name__ == "__main__":
    #run("../normal/YaK5lgBy0sunsplash.png",0.1)
    run("../stego/LSB w BMP/zdj1sh.png",0.1)
    pass