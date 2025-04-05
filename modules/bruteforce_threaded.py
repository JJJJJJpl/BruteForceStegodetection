from functools import reduce
from itertools import permutations
from PIL import Image
import numpy as np
from collections import Counter
import string
from concurrent.futures import ThreadPoolExecutor
import threading

def run(filename, window_size, treshold, max_workers=4):
    image = Image.open(filename)
    if image.mode != 'RGBA':
      image = image.convert("RGB")
    lock = threading.Lock()
    
    with open("output.txt", 'w') as output_file:
      def process_e(e):
         if len(e) > 0:
            f = check(e, window_size, treshold, entropy_score)
            if len(f) > 0:
               g = check(e, window_size, treshold, character_distribution_score)
               if len(g) > 0:
                  h = check(e, window_size, treshold, doubles_score)
                  if len(h) > 0:
                     with lock:
                        output_file.write(h)
                        output_file.write("\n\n")
                        #print("Potential text detected:", h)
      
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
                raise StopIteration
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
    def __init__(self, array):
        self.arr = array
        self.modes = None
        self.mode_index = 0 # permutacje: RGB, BGR, BRG ...
        self.variant = 0  # 0 = interleaved, 1 = grouped
        if self.arr.shape[1] == 3:
            self.modes = list(permutations([0, 1, 2]))  # indeksy R=0, G=1, B=2
        else:
            self.modes = list(permutations([0, 1, 2, 3]))  # indeksy R=0, G=1, B=2, A=3

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode_index >= len(self.modes):
            raise StopIteration

        mode = self.modes[self.mode_index]

        if self.variant == 0:
            # Interleaved: R G B R G B ...
            data = self.arr[:, list(mode)].flatten()
        else:
            # Grouped: RRR... GGG... BBB...
            data = np.concatenate([self.arr[:, i] for i in mode])

        self.variant += 1
        if self.variant > 1:
            self.variant = 0
            self.mode_index += 1

        return data

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

         if self.j < self.mismatch:
            bytes_list = self.arr[self.j: -(self.mismatch - self.j)]
         else:
            bytes_list = self.arr[self.mismatch:]

         byte_data = np.packbits(bytes_list).tobytes()

         self.j += 1

         if self.j >= self.mismatch:
            self.i += 1
            self.j = 0

         if self.i == 0:
            return byte_data.decode('utf-8', errors='ignore')
         else:
            return byte_data.decode('ascii', errors='ignore')
      else:
         raise StopIteration


def check(text,window_size,treshold,function):
   if len(text) < window_size:
      if function(window) > treshold:
         return text

   result = ""
   for i in range(0,len(text),window_size):
      window = text[i:i+window_size]
      if function(window) > treshold:
         result += window
   return result

def entropy_score(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    unique_letters = set(letters)
    return len(unique_letters) / len(text)

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
    #run("../normal/YaK5lgBy0sunsplash.png",50,0.2)
    run("../stego/LSB w BMP/YaK5Long.png",20,0.2)
    pass