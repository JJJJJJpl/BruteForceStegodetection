import os
import cv2
import numpy as np
import json
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
import numpy as np


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
        #blocks = self.dct_analysis()
        #print(properties)
        #self.display_histogram(histogram)
        #print(blocks)
        #self.calculate_gefr()
        #self.is_stego_suspected_gefr()
        #self.gref_analysis()
        #print(self.is_stego_suspected_rs())
        #self.pov_analysis()
        print(self.is_stego_suspected_pov())
        
    
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


    #def display_histogram(self, histograms):
    #    for col, hist in histograms.items():
    #        plt.plot(hist, color=col)
    #    plt.title("Histogram")
    #    plt.show()

    def process_image_directory(self,directory_path):
        gefr_list = []
        pov_list = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png','.bmp')):
                image_path = os.path.join(directory_path, filename)
                print(f"Przetwarzanie: {image_path}")
                gefr = self.gref_analysis(str(image_path))
                if gefr:
                    gefr_list.append(gefr)
                pov = self.pov_analysis(str(image_path))
                if pov:
                    pov_list.append(pov)

        # Oblicz średnie i odchylenia standardowe dla każdej metryki
        metrics_gefr = ['mean_gefr', 'std_gefr', 'skewness_gefr', 'kurtosis_gefr']
        summary = {}
        for metric in metrics_gefr:
            values = [g[metric] for g in gefr_list]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
        metrics_pov = ['Blue_PDS', 'Green_PDS', 'Red_PDS', 'Grayscale_PDS']
        for metric in metrics_pov:
            values = [p[metric]["PoV Difference Sum"] for p in pov_list]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
        print(summary)


        with open("BruteForceStegodetection\\modules\\summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
        return summary
  

    def gref_analysis(self, image_path=None):
        
        def calculate_gefr(img):
            dct_blocks = dct_analysis(img)  # (n, 8, 8)

            # Pomijamy składową DC (0,0), bierzemy tylko AC coefficients
            ac_coeffs = []

            for block in dct_blocks:
                flat = block.flatten()
                ac = np.delete(flat, 0)  # usuwamy element [0]
                ac_coeffs.extend(ac)

            ac_coeffs = np.array(ac_coeffs)

            # Normalizacja
            if np.max(np.abs(ac_coeffs)) != 0:
                ac_coeffs /= np.max(np.abs(ac_coeffs))  # Skala [-1, 1]

            # Obliczanie statystyk
            gefr_metrics = {
                "mean_gefr": np.mean(ac_coeffs),
                "std_gefr": np.std(ac_coeffs),
                "skewness_gefr": skew(ac_coeffs),
                "kurtosis_gefr": kurtosis(ac_coeffs)
            }
            return gefr_metrics

        def dct_analysis(img):
            # Przeprowadzanie analizy DCT na obrazie
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img_gray.shape
            dct_blocks = []

            for i in range(0, h - 7, 8):  
                for j in range(0, w - 7, 8):
                    block = np.float32(img_gray[i:i+8, j:j+8]) - 128
                    dct_block = cv2.dct(block)
                    dct_blocks.append(dct_block)
            
            return np.array(dct_blocks)

        if image_path is None:
            image = self.img
        else:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
        resault = calculate_gefr(image)
        print(resault)
        return resault
    
    def is_stego_suspected_gefr(self, threshold=2):
        while True:
            if not os.path.exists("BruteForceStegodetection\\modules\\summary.json"):
                print("brak summary, generuje summary")
                self.process_image_directory("BruteForceStegodetection\\normal")
            else:
                with open ("BruteForceStegodetection\\modules\\summary.json", 'r') as summar:
                    reference_stats = json.load(summar)
                break
        gefr = self.gref_analysis()
        suspicion_score = 0
        for key in gefr:
            ref_mean = reference_stats[key]["mean"]
            ref_std = reference_stats[key]["std"]
            if abs(gefr[key] - ref_mean) > threshold * ref_std:
                suspicion_score += 1
        
        # Jeśli więcej niż 2 z 4 cech są podejrzane – uznajemy obraz za potencjalnie stego
        print(suspicion_score)
        return suspicion_score >= 2
          

    def rs_analysis(self):
        def lsb_flip(pixel):
            return pixel ^ 1  # XOR z 1 odwraca najmłodszy bit

        def group_smoothness(group):
            return sum(abs(int(group[i]) - int(group[i+1])) for i in range(len(group) - 1))
        
        def analyze_channel(channel, name="Channel"):
            h, w = channel.shape
            regular, singular = 0, 0

            # Przechodzimy liniowo po obrazie (grupy po 4 piksele)
            for i in range(h):
                for j in range(0, w - 4, 4):
                    group = channel[i, j:j+4]
                    original_smooth = group_smoothness(group)
                    
                    flipped_group = np.array([lsb_flip(p) for p in group])
                    flipped_smooth = group_smoothness(flipped_group)

                    if flipped_smooth > original_smooth:
                        singular += 1
                    elif flipped_smooth < original_smooth:
                        regular += 1
                    # jeśli equal – ignorujemy

            return {
                "regular": regular,
                "singular": singular,
                "R/S ratio": round(regular / singular, 3) if singular else float('inf')
            }

        results = {}

        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            # Dla RGB
            channels = cv2.split(self.img)
            names = ['Blue', 'Green', 'Red']
            for ch, name in zip(channels, names):
                results[name] = analyze_channel(ch, name)
            
            # Dla Grayscale
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            results['Grayscale'] = analyze_channel(gray, 'Grayscale')
        else:
            # Jeśli obraz już jest grayscale
            results['Grayscale'] = analyze_channel(self.img, 'Grayscale')
        return results

    def is_stego_suspected_rs(self, treshold=0.1):
        result = self.rs_analysis()

        for key, value in result.items():
            if value["R/S ratio"] > 1 -treshold and value["R/S ratio"] < 1 + treshold:
                print(f"Podejrzany kanał: {key}, R/S ratio: {value['R/S ratio']}")
                return True
        return False
                

    def pov_analysis(self, image_path=None):
        def analyze_channel(channel, name="Channel"):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist = hist.flatten()
            pov_diff_sum = 0

            for k in range(0, 256, 2):
                if k + 1 >= len(hist):
                    break
                p0 = hist[k]
                p1 = hist[k + 1]
                pov_diff_sum += abs(p0 - p1)

            return {
                "PoV Difference Sum": int(pov_diff_sum),
                "Channel": name
            }

        results = {}
        if image_path is None:
            image = self.img
        else:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB
            channels = cv2.split(image)
            names = ['Blue', 'Green', 'Red']
            names2 = ['Blue_PDS', 'Green_PDS', 'Red_PDS']
            for ch, name, name2 in zip(channels, names, names2):
                results[name2] = analyze_channel(ch, name)

            # Grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            results['Grayscale_PDS'] = analyze_channel(gray, 'Grayscale')
        else:
            # Już grayscale
            results['Grayscale_PDS'] = analyze_channel(image, 'Grayscale')

        print(results)
        return results
  
    def is_stego_suspected_pov(self, threshold=2):
        while True:
            if not os.path.exists("BruteForceStegodetection\\modules\\summary.json"):
                print("brak summary, generuje summary")
                self.process_image_directory("BruteForceStegodetection\\normal")
            else:
                with open ("BruteForceStegodetection\\modules\\summary.json", 'r') as summar:
                    reference_stats = json.load(summar)
                break
        
        PoV = self.pov_analysis()
        suspicion_score = 0
        for key in PoV:
            PoV_mean = reference_stats[key]["mean"]
            PoV_std = reference_stats[key]["std"]
            if abs(PoV[key]['PoV Difference Sum'] - PoV_mean) > threshold * PoV_std:
                suspicion_score += 1
        
        # Jeśli więcej niż 2 z 4 cech są podejrzane – uznajemy obraz za potencjalnie stego
        print(suspicion_score)
        return suspicion_score >=1
    

class JpegAnalyzer:
    def __init__(self, image_path):
        self.recompressed_path = "BruteForceStegodetection\\modules\\recompressed.jpg"
        self.image_path = image_path

    def read_jpeg_params(self):
        image_path = self.image_path
        dct_blocks = self.extract_dct_coeffs(image_path)
        self.recompress_image(quality=75)
        dct_anomalys = self.analyze_DCT(dct_blocks)
        artifacts_anomalys = self.analyze_artifacts(dct_blocks)
        anomaly = self.is_stego_suspected_dct(dct_anomalys, artifacts_anomalys)
        print(anomaly)
        return anomaly

    def is_stego_suspected_dct(self, dct_anomalys, artifacts_anomalys):
        histogram_anomaly, benford_anomaly = artifacts_anomalys
        print(histogram_anomaly, benford_anomaly, dct_anomalys)
        chanels = ['Y', 'Cb', 'Cr']
        for channel in chanels:
            if histogram_anomaly[channel] and benford_anomaly[channel] or dct_anomalys and histogram_anomaly[channel] or dct_anomalys and benford_anomaly[channel]:
                print(f"Podejrzany kanał: {channel}")
                return True


    def extract_dct_coeffs(self,image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Nie można otworzyć pliku: {image_path}. Sprawdź ścieżkę.")
        ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 

        Y, Cr, Cb = cv2.split(ycbcr)

        def compute_dct(channel):
            h, w = channel.shape

            # Zaokrągl w górę do wielokrotności 8
            new_h = ((h + 7) // 8) * 8
            new_w = ((w + 7) // 8) * 8

            # Padding obrazu
            padded = np.zeros((new_h, new_w), dtype=np.uint8)
            padded[:h, :w] = channel

            dct_blocks = []

            for i in range(0, new_h, 8):
                for j in range(0, new_w, 8):
                    block = np.float32(padded[i:i+8, j:j+8]) - 128
                    dct_blocks.append(cv2.dct(block))

            return np.array(dct_blocks)

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


    def analyze_DCT(self, dct_blocks):
        recompressed_dct = self.extract_dct_coeffs(self.recompressed_path)
        def comapare_params(dct_blocks, recompressed_dct):
            original_dct = dct_blocks
            if recompressed_dct is None:
                raise ValueError("Błąd: Nie udało się odczytać współczynników DCT z recompressed.jpg")

            difference = {
                "Y": original_dct["Y"] - recompressed_dct["Y"],
                "Cb": original_dct["Cb"] - recompressed_dct["Cb"],
                "Cr": original_dct["Cr"] - recompressed_dct["Cr"],
            }
            return difference

        def analyze_dct_differences(dct_blocks, threshold=2.0):
            suspicion_score = 0
            report = {}

            difference = comapare_params(dct_blocks, recompressed_dct)

            for channel in ["Y", "Cb", "Cr"]:
                diff = difference[channel]
                abs_mean = np.mean(np.abs(diff))
                std_dev = np.std(diff)
                max_diff = np.max(np.abs(diff))

                report[channel] = {
                    "MeanAbsDiff": abs_mean,
                    "StdDev": std_dev,
                    "MaxAbsDiff": max_diff
                }

                # Prosta reguła: jeśli średnia różnica przekracza próg, podnieś alert
                if abs_mean > threshold:
                    suspicion_score += 1

            for ch, stats in report.items():
                print(f"{ch} -> MeanAbsDiff: {stats['MeanAbsDiff']:.2f}, StdDev: {stats['StdDev']:.2f}, MaxAbsDiff: {stats['MaxAbsDiff']:.2f}")
            
            if suspicion_score >= 2:
                return True
            else:
                return False

        is_diffrance_anomaly = analyze_dct_differences(dct_blocks)
        return is_diffrance_anomaly
    
    def analyze_artifacts(self, dct_blocks):

        def analyze_dct_histogram_anomalies(dct_original, dct_recompressed, channel='Y'):
            # Wybieramy kanał
            orig = dct_original[channel].flatten()
            rec = dct_recompressed[channel].flatten()

            # Histogramy wartości DCT (w zakresie -100 do 100 dla lepszej czułości)
            hist_orig, _ = np.histogram(orig, bins=201, range=(-100, 100), density=True)
            hist_rec, _ = np.histogram(rec, bins=201, range=(-100, 100), density=True)

            # Obliczamy różnicę histogramów (np. L1 normę)
            diff = np.abs(hist_orig - hist_rec)
            anomaly_score = np.sum(diff)

            return bool(anomaly_score > 0.2)  # Próg do wykrywania anomalii

        def analyze_benford_law(dct_data, channel='Y'):

            dct_flat = np.abs(dct_data[channel].flatten())
            dct_flat = dct_flat[dct_flat > 0]  # pomiń zera

            # Ekstrakcja pierwszych cyfr
            first_digits = [int(str(int(x))[0]) for x in dct_flat if x >= 1]

            # Liczenie rozkładu cyfr 1-9
            digit_counts = np.zeros(9)
            for digit in first_digits:
                if 1 <= digit <= 9:
                    digit_counts[digit - 1] += 1

            # Normalizacja
            digit_distribution = digit_counts / np.sum(digit_counts)


            # Rozkład teoretyczny Benforda
            benford_dist = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

            # Porównanie rozkładów (L1 distance)
            differencee = np.abs(digit_distribution - benford_dist)
            

            mean_diff = np.mean(differencee)
            std_diff = np.std(differencee)
            
            # Dostosowanie progu na podstawie odchylenia standardowego
            threshold = mean_diff + 2 * std_diff  # np. ustawienie progu na 2 razy odchylenie standardowe
            
            # Porównanie z obliczonym progiem
            benford_score = np.sum(differencee)

            #benford_score = np.sum(differencee)
            print(f"Benford score for {channel}: {benford_score:.4f}: {threshold:.4f}")
            # Jeśli różnica jest większa niż próg, uznajemy to za anomalię
            return bool(benford_score > 0.12)

        is_histogram_anomaly = {}
        is_benford_anomaly = {}

        dct_decompressed = self.extract_dct_coeffs(self.recompressed_path)
        chanels = ['Y', 'Cb', 'Cr']
        for channel in chanels:
            is_histogram_anomaly[channel] = analyze_dct_histogram_anomalies(dct_blocks, dct_decompressed, channel=channel)
            is_benford_anomaly[channel] = analyze_benford_law(dct_blocks, channel=channel)
        
        return is_histogram_anomaly, is_benford_anomaly

analyze_image("BruteForceStegodetection\\normal\\sumup-2enEyX2MAvQ-unsplash.jpg")
#analyze_image("BruteForceStegodetection\\stego\\LSB w JPG\\cat.jpg")
#Image = PngAnalyzer("BruteForceStegodetection\\modules\\zakodowanylsb.png")
#Image.process_image_directory("BruteForceStegodetection\\normal")