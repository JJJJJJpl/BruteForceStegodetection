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
        self.recompressed_path = "modules\\recompressed.jpg"
        self.image_path = image_path


    def save_jpeg_params(self):
        folders = ["grouped_by_resolution\\res_1_250",
                   "grouped_by_resolution\\res_251_500",
                   "grouped_by_resolution\\res_501_750",
                   "grouped_by_resolution\\res_751_1000",
                   "grouped_by_resolution\\res_1001_1250"]
        for folder in folders:
            stats = self.analyze_folder(folder)
            print(f"Folder: {folder}")
            print(stats)
            with open(f"{folder}\\stats.json", 'w') as f:
                json.dump(stats, f, indent=4)
            
    def analyze_folder(self,folder_path):
        # Struktura danych do agregacji wyników
        dct_means = {'Y': [], 'Cb': [], 'Cr': []}
        artifact_scores = {'Y': [], 'Cb': [], 'Cr': []}

        # Iteruj przez pliki w folderze
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, filename)
                try:
                    dct_blocks = self.extract_dct_coeffs(image_path)
                    if dct_blocks is None:
                        print(f"{filename}: extract_dct_coeffs zwróciło None")
                        continue
                    self.recompress_image(quality=75,image_path=image_path)

                    dct_anomalys = self.analyze_DCT(dct_blocks)
                    if dct_anomalys is None:
                        print(f"{filename}: analyze_DCT zwróciło None")
                        continue
                    artifacts_anomalys = self.analyze_artifacts(dct_blocks)
                    if artifacts_anomalys is None:
                        print(f"{filename}: analyze_artifacts zwróciło None")
                        continue
                    # Sumowanie DCT i artefaktów dla każdego kanału
                    for channel in ['Y', 'Cb', 'Cr']:
                        dct_means[channel].append(dct_anomalys[channel]['MeanAbsDiff'])
                        artifact_scores[channel].append(artifacts_anomalys[channel])

                except Exception as e:
                    print(f"Błąd przy pliku {filename}: {e}")
                    continue

        # Obliczanie średnich i błędu standardowego
        stats = {}
        for channel in ['Y', 'Cb', 'Cr']:
            dct_mean = np.mean(dct_means[channel])
            dct_std = np.std(dct_means[channel])
            artifact_mean = np.mean(artifact_scores[channel])
            artifact_std = np.std(artifact_scores[channel])

            stats[channel] = {
                'DCT_MeanAbsDiff_Mean': dct_mean,
                'DCT_MeanAbsDiff_Std': dct_std,
                'Artifacts_Mean': artifact_mean,
                'Artifacts_Std': artifact_std
            }

        return stats
      

    def read_jpeg_params(self):
        image_path = self.image_path
        dct_blocks,folder_path = self.extract_dct_coeffs(image_path)
        self.recompress_image(quality=75)
        dct_anomalys = self.analyze_DCT(dct_blocks)
        artifacts_anomalys = self.analyze_artifacts(dct_blocks)
        
        anomaly = self.is_stego_suspected_dct(dct_anomalys, artifacts_anomalys)
        print(dct_anomalys,artifacts_anomalys)
        print(anomaly)
        return 

    def is_stego_suspected_dct(self, dct_anomalys, artifacts_anomalys):
        histogram_anomaly, benford_anomaly = artifacts_anomalys
        chanels = ['Y', 'Cb', 'Cr']
        for channel in chanels:
            if histogram_anomaly[channel] and benford_anomaly[channel] or dct_anomalys[channel] and histogram_anomaly[channel] or dct_anomalys[channel] and benford_anomaly[channel]:
                print(f"Podejrzany kanał: {channel}")
                return True
        return False

    def extract_dct_coeffs(self,image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Nie można otworzyć pliku: {image_path}. Sprawdź ścieżkę.")
        ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 
        height, width = img.shape[:2]
        max_dim = max(width, height)
        match max_dim:
            case h if h < 250 and h > 0:
                folder_path = "grouped_by_resolution\\res_1_250"
            case h if h < 500 and h > 250:
                folder_path = "grouped_by_resolution\\res_251_500"
            case h if h < 750 and h > 500:
                folder_path = "grouped_by_resolution\\res_501_750"
            case h if h < 1000 and h > 750:
                folder_path = "grouped_by_resolution\\res_751_1000"
            case h if h > 1000:
                folder_path = "grouped_by_resolution\\res_1001_1250"
            case _:
                raise ValueError("Nieprawidłowy rozmiar obrazu. Rozmiar musi być większy od 0.")

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
        },folder_path

    def recompress_image(self, quality=75, image_path=None):
        if image_path is None:
            image_path = self.image_path
        img = cv2.imread(image_path)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', img, encode_param)
        
        with open(self.recompressed_path, "wb") as f:
            f.write(encoded_img.tobytes())

        return "recompressed.jpg"


    def analyze_DCT(self, dct_blocks):
        recompressed_dct,folder_path = self.extract_dct_coeffs(self.recompressed_path)
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

        def analyze_dct_differences(dct_blocks, folder_path="grouped_by_resolution\\res_1_250"):
            suspicion_score = 0
            report = {}
            sus = {}
            stats = json.load(open(f"{folder_path}\\stats.json", 'r'))

            difference = comapare_params(dct_blocks, recompressed_dct)

            for channel in ["Y", "Cb", "Cr"]:
                diff = difference[channel]
                abs_mean = float(np.mean(np.abs(diff)))
                std_dev = float(np.std(diff))
                max_diff = float(np.max(np.abs(diff)))

                report[channel] = {
                    "MeanAbsDiff": abs_mean,
                    "StdDev": std_dev,
                    "MaxAbsDiff": max_diff
                }

            
            

                # Prosta reguła: jeśli średnia różnica przekracza próg, podnieś alert
                print("=====================")
                print(stats[channel]['DCT_MeanAbsDiff_Mean'] + 2 * stats[channel]['DCT_MeanAbsDiff_Std'])
                print (report[channel])
                print("=====================")
                sus[channel] = abs_mean > stats[channel]['DCT_MeanAbsDiff_Mean'] + 2 * stats[channel]['DCT_MeanAbsDiff_Std']

            for ch, stats in report.items():
                print(f"{ch} -> MeanAbsDiff: {stats['MeanAbsDiff']:.2f}, StdDev: {stats['StdDev']:.2f}, MaxAbsDiff: {stats['MaxAbsDiff']:.2f}")
            
            return sus

        is_diffrance_anomaly = analyze_dct_differences(dct_blocks,folder_path)
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
            #epsilon = 1e-10
            #kl_div = np.sum(hist_orig * np.log((hist_orig + epsilon) / (hist_rec + epsilon)))
            #anomaly_score = kl_div
            #weights = np.exp(-np.abs(np.linspace(-100, 100, 201)))  # większa waga przy 0
            #diff = weights * np.abs(hist_orig - hist_rec)
            #anomaly_score = np.sum(diff)

            return bool(anomaly_score > 0.2)  # Próg do wykrywania anomalii

        def analyze_benford_law(dct_blocks, channel='Y', folder_path="grouped_by_resolution\\res_1_250"):

            blocks = dct_blocks[channel]
            
            # Jeżeli blocks to np. (num_blocks, 8, 8) albo (H, W, 8, 8)
            if blocks.ndim == 4:
                all_blocks = blocks.reshape(-1, 8, 8)
            elif blocks.ndim == 3 and blocks.shape[1:] == (8, 8):
                all_blocks = blocks
            else:
                raise ValueError(f"Nieoczekiwany kształt tablicy bloków: {blocks.shape}")

            coefficients = []
            for block in all_blocks:
                ac_coeffs = np.delete(block.flatten(), 0)  # pomiń DC coefficient
                coefficients.extend(ac_coeffs)

            coeffs_abs = np.abs(coefficients)
            coeffs_abs = coeffs_abs[coeffs_abs > 0]

            if len(coeffs_abs) == 0:
                print(f"[{channel}] Zbyt mało danych do analizy.")
                return False

            # Ekstrakcja pierwszych cyfr
            first_digits = [int(str(int(val))[0]) for val in coeffs_abs if val >= 1]

            digit_counts = np.zeros(9)
            for digit in first_digits:
                if 1 <= digit <= 9:
                    digit_counts[digit - 1] += 1

            digit_distribution = digit_counts / np.sum(digit_counts)
            benford_dist = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
            difference = np.abs(digit_distribution - benford_dist)
            benford_score = float(np.sum(difference) / 9)  # normalizacja
            num_coeffs = len(first_digits)
            stats = json.load(open(f"{folder_path}\\stats.json", 'r'))
            thresholds = {
                'Y': stats['Y']['Artifacts_Mean'] + 2 * stats['Y']['Artifacts_Std'],
                'Cb': stats['Cb']['Artifacts_Mean'] + 2 * stats['Cb']['Artifacts_Std'],
                'Cr': stats['Cr']['Artifacts_Mean'] + 2 * stats['Cr']['Artifacts_Std']
            }
            


            print(f"Benford score for {channel}: {benford_score:.4f}")
            print(f"Rozkład: {digit_distribution.round(3)}")
            print(f"Benford: {benford_dist.round(3)}")
            print(f"Różnica: {difference.round(3)}")
            print(f"Liczba współczynników: {num_coeffs}")
            print(f"Próg: {thresholds[channel]:.4f}")
            return benford_score > thresholds[channel]  # Próg do wykrywania anomalii


        is_histogram_anomaly = {}
        is_benford_anomaly = {}

        dct_decompressed,folder_path = self.extract_dct_coeffs(self.recompressed_path)
        chanels = ['Y', 'Cb', 'Cr']
        for channel in chanels:
            is_histogram_anomaly[channel] = analyze_dct_histogram_anomalies(dct_blocks, dct_decompressed, channel=channel)
            is_benford_anomaly[channel] = analyze_benford_law(dct_blocks, channel=channel,folder_path=folder_path)
        
        return is_benford_anomaly,is_histogram_anomaly

#analyze_image("stego\\LSB w JPG\\AAAAA.jpg")
#analyze_image("normal\\pi9UmWpTId0-unsplash.jpg")

#analyze_image("modules\\Cat_November_2010-1.jpg")
analyze_image("stego\\LSB w JPG\\cat.jpg")
#Image = PngAnalyzer("BruteForceStegodetection\\modules\\zakodowanylsb.png")
#Image.process_image_directory("BruteForceStegodetection\\normal")