import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def analyze_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext == ".jpg" or ext == ".jpeg":
        analyzerJPG = JpegAnalyzer(image_path)
        return analyzerJPG.read_jpeg_params()
    elif ext == ".png" or ext == ".bmp":
        analyzerPNG = PngAnalyzer(image_path)
        return analyzerPNG.read_png_params()
    else:
        raise ValueError("Nieobsługiwany format pliku")


class PngAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.img is None:
            raise FileNotFoundError(f"Nie udało się otworzyć pliku: {image_path}")
    
    def read_png_params(self):
        #properties = self.get_image_properties()
        #histogram = self.get_histogram()
        blocks = self.dct_analysis()
        #print(properties)
        #self.display_histogram(histogram)
        print(blocks)
        
    
    def get_image_properties(self):
        height, width = self.img.shape[:2]
        depth = self.img.dtype
        channels = self.img.shape[2] if len(self.img.shape) == 3 else 1
        return {
            "Resolution": (height, width),
            "Bit Depth": depth,
            "Channels": channels
        }
    def get_histogram(self):
        if len(self.img.shape) == 3:
            # Histogram dla RGB
            color = ('b', 'g', 'r')
            histograms = {}
            for i, col in enumerate(color):
                hist = cv2.calcHist([self.img], [i], None, [256], [0, 256])
                histograms[col] = hist
            return histograms
        else:
            # Histogram dla obrazu grayscale
            hist = cv2.calcHist([self.img], [0], None, [256], [0, 256])
            return {"grayscale": hist}
    
    #def edge_detection(self):
    #    # Detekcja krawędzi przy użyciu algorytmu Canny
    #    edges = cv2.Canny(self.img, 100, 200)
    #    return edges

    def dct_analysis(self):
        # Przeprowadzanie analizy DCT na obrazie
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        dct_blocks = []

        for i in range(0, h - 7, 8):  
            for j in range(0, w - 7, 8):
                block = np.float32(img_gray[i:i+8, j:j+8]) - 128
                dct_block = cv2.dct(block)
                dct_blocks.append(dct_block)
        
        return np.array(dct_blocks)

    def display_histogram(self, histograms):
        for col, hist in histograms.items():
            plt.plot(hist, color=col)
        plt.title("Histogram")
        plt.show()

class JpegAnalyzer:
    def __init__(self, image_path):
        self.recompressed_path = "BruteForceStegodetection\\modules\\recompressed.jpg"
        self.image_path = image_path

    def read_jpeg_params(self):
        image_path = self.image_path
        dct_blocks = self.extract_dct_coeffs(image_path)
        #print(dct_blocks)
        print("Reading JPEG parameters...")
        self.recompress_image(quality=75)
        print("Recompressing image...")
        difference = self.comapare_params(dct_blocks)
        return difference

    def extract_dct_coeffs(self,image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Nie można otworzyć pliku: {image_path}. Sprawdź ścieżkę.")
        ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 

        Y, Cr, Cb = cv2.split(ycbcr)

        def compute_dct(channel):
            h, w = channel.shape
            dct_blocks = np.zeros((h // 8, w // 8, 8, 8))

            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    block = np.float32(channel[i:i+8, j:j+8]) - 128  
                    dct_blocks[i//8, j//8] = cv2.dct(block)  

            return dct_blocks

        return {
            "Y": compute_dct(Y),
            "Cb": compute_dct(Cb),
            "Cr": compute_dct(Cr)
        }

    def recompress_image(self, quality=75):
        image_path = self.image_path
        img = cv2.imread(image_path)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', img, encode_param)
        
        with open(self.recompressed_path, "wb") as f:
            f.write(encoded_img.tobytes())

        return "recompressed.jpg"


    def comapare_params(self, dct_blocks):
        original_dct = dct_blocks
        recompressed_dct = self.extract_dct_coeffs(self.recompressed_path)

        if recompressed_dct is None:
            raise ValueError("Błąd: Nie udało się odczytać współczynników DCT z recompressed.jpg")

        difference = {
            "Y": original_dct["Y"] - recompressed_dct["Y"],
            "Cb": original_dct["Cb"] - recompressed_dct["Cb"],
            "Cr": original_dct["Cr"] - recompressed_dct["Cr"],
        }
        print(difference)

analyze_image("BruteForceStegodetection\modules\\czystypng.png")