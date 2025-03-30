from PIL import Image
import numpy as np

def run(filename):
    image = Image.open(filename)
    image = image.convert("RGBA")
    for a in BlockIter1(image):
       for b in OrderIter2(a):
          for c in ColorIter3(b):
            print(c)
    pass

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
       #raise StopIteration
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



if __name__ == "__main__":
    run("test.png")
    pass