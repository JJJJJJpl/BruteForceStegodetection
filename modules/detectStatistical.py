import os
import cv2
from multiprocessing import Pool, cpu_count
import numpy as np
#import json
#from matplotlib import pyplot as plt
#from scipy.stats import skew, kurtosis
#import shutil
#from scipy.stats import entropy
#import multiprocessing
#from scipy.stats import chi2
#from numba import njit


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
        if self.img.shape[2] == 4:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGRA2BGR)
        if self.img is None:
            raise FileNotFoundError(f"Nie udało się otworzyć pliku: {image_path}")
    
    def read_png_params(self):
        print("Analiza metodą RS...")
        print(f"Wykrycie Steganografi: {self.is_stego_suspected_rs()}")
        #print(self.is_stego_suspected_pov())
        #print(self.chi_square_lsb())
        #print(self.pov_complementary_true())
        #print(self.pov_analysis())
    



    def analyze_pov_until_stable_std(self, pathr, output_folder="outliers", threshold=2.0, target_std=0.6):
        os.makedirs(output_folder, exist_ok=True)
        iteration = 0
        all_outliers = []

        while True:
            iteration += 1
            pov_list = []

            # Zbierz dane
            for filename in os.listdir(pathr):
                if filename.lower().endswith(('.png', '.bmp')):
                    image_path = os.path.join(pathr, filename)
                    pov = self.pov_analysis(image_path)
                    if pov and "Grayscale_PDS" in pov:
                        value = pov["Grayscale_PDS"]["PoV Difference Sum"]
                        pov_list.append((filename, value))

            if len(pov_list) < 3:
                print(" Za mało obrazów do dalszej analizy.")
                break

            values = np.array([v[1] for v in pov_list])
            mean = np.mean(values)
            std = np.std(values)

            print(f"\n Iteracja {iteration}: Średnia = {mean:.4f}, Odchylenie std = {std:.4f}")

            if std <= target_std:
                print(" Odchylenie standardowe osiągnęło docelową wartość.")
                break

            outliers = []
            for filename, value in pov_list:
                z_score = (value - mean) / std if std > 0 else 0
                if abs(z_score) > threshold:
                    outliers.append((filename, value, z_score))

            if not outliers:
                print("ℹ️ Brak nowych outlierów do usunięcia — zatrzymuję.")
                break

            # Przenieś i usuń z folderu wejściowego
            for filename, value, z_score in outliers:
                src = os.path.join(pathr, filename)
                dst = os.path.join(output_folder, filename)
                shutil.copy2(src, dst)
                os.remove(src)
                print(f" 🚫 Outlier: {filename} | Wartość: {value:.2f} | Z-score: {z_score:.2f}")

            all_outliers.extend(outliers)

        # Zapisz metadane wszystkich outlierów
        out_json = os.path.join(output_folder, "outliers_metadata.json")
        with open(out_json, "w") as f:
            json.dump([
                {"filename": name, "value": val, "z_score": z}
                for name, val, z in all_outliers
            ], f, indent=4)

        print(f"\n📦 Przeniesiono {len(all_outliers)} obrazów odstających do: {output_folder}")
        return all_outliers

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

        # Dodaj metryki GEFR
        for metric in metrics_gefr:
            values = [g[metric] for g in gefr_list]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }

        # Dodaj metryki POV (bez nadpisywania summary!)
        metrics_pov = ['Blue_PDS', 'Green_PDS', 'Red_PDS', 'Grayscale_PDS']

        for metric in metrics_pov:
            values = []

            for p in pov_list:
                if metric in p and "PoV Normalized" in p[metric]:
                    values.append(p[metric]["PoV Normalized"])

            if values:
                summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
            else:
                print(f"Brak danych dla metryki: {metric}")

        print(summary)

        path = os.path.join(directory_path, "summary.json")
        with open(f"{path}", 'w') as f:
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
            if len(img.shape) == 3:
                if img.shape[2] == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif img.shape[2] == 1:
                    img_gray = img[:, :, 0]
                else:
                    print(f"Niezwykły format obrazu: shape={img.shape}")
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # domyślnie próbuj przekonwertować
            else:
                img_gray = img  # Jeśli obraz jest już grayscale


            #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        resault = calculate_gefr(image)
        print(resault)
        return resault
    
    def is_stego_suspected_gefr(self, threshold=2):
        height, width = self.img.shape[:2]
        max_dim = max(width, height)
        match max_dim:
            case h if h <= 250 and h > 0:
                folder_path = "grouped_by_resolution\\res_1_250"
            case h if h <= 500 and h > 250:
                folder_path = "grouped_by_resolution\\res_251_500"
            case h if h <= 750 and h > 500:
                folder_path = "grouped_by_resolution\\res_501_750"
            case h if h <= 1000 and h > 750:
                folder_path = "grouped_by_resolution\\res_751_1000"
            case h if h <= 1250 and h > 1000:
                folder_path = "grouped_by_resolution\\res_1001_1250"
            case h if h <= 1500 and h > 1250:
                folder_path = "grouped_by_resolution\\res_1251_1500"
            case h if h <= 1750 and h > 1500:
                folder_path = "grouped_by_resolution\\res_1501_1750"
            case h if h > 1750:
                folder_path = "grouped_by_resolution\\res_1751_2000"
            case _:
                raise ValueError("Nieprawidłowy rozmiar obrazu. Rozmiar musi być większy od 0.")
        path = os.path.join(folder_path, "summary.json")
        while True:
            if not os.path.exists(f"{path}"):
                
                ValueError("Nie można znaleźć summary.json. Proszę najpierw uruchomić process_image_directory.")
            else:
                with open (f"{path}", 'r') as summar:
                    reference_stats = json.load(summar)
                break
        gefr = self.gref_analysis()
        suspicion_score = 0
        for key in gefr:
            ref_mean = reference_stats[key]["mean"]
            ref_std = reference_stats[key]["std"]
            if abs(gefr[key] - ref_mean) > threshold * ref_std:
                print(f"Podejrzany kanał: {key}, GEFR: {gefr[key]}")
                suspicion_score += 1
        
        # Jeśli więcej niż 2 z 4 cech są podejrzane – uznajemy obraz za potencjalnie stego
        print(suspicion_score)
        return suspicion_score >= 2
          
    def rs_analysis_single_debug(self):
        print("[INFO] Start RS analysis (debug single-thread)...")

        results = {}
        directions = ['horizontal', 'vertical', 'diagonal']
        masks = {
            'mask1': np.array([1, 0, 1, 0]),
            'mask2': np.array([0, 1, 0, 1]),
        }

        if len(self.img.shape) == 3:
            channels = cv2.split(self.img)
            channel_names = ['Blue', 'Green', 'Red']
        else:
            channels = [self.img]
            channel_names = ['Grayscale']

        tasks = []

        # 🔹 Przygotuj tylko jeden przypadek do testu!
        for ch_name, ch in zip(channel_names, channels):
            for direction in directions:
                for mask_name, mask in masks.items():
                    args = (ch, direction, mask_name, mask)
                    tasks.append(args)
                    break  # tylko 1 maska
                break  # tylko 1 kierunek
            break  # tylko 1 kanał

        print(f"[DEBUG] Number of tasks to process: {len(tasks)}")

        with multiprocessing.Pool(processes=1) as pool:
            results_list = pool.map(self.process_channel_direction_mask, tasks)

        for direction, mask_name, stats in results_list:
            print(f"[RESULT] {direction}, {mask_name}: {stats}")

        return results_list
    
    def rs_analysis(self):
        def analyze_channel_parallel(self, channel):
            directions = ["horizontal", "vertical", "diag_down", "diag_up"]
            masks = {"M1": [1, 1, 1, 1], "M2": [1, -1, 1, -1]}

            tasks = [(channel, dir, mname, mask) for dir in directions for mname, mask in masks.items()]

            with Pool(cpu_count()) as pool:
                results = pool.map(self.process_channel_direction_mask, tasks)

            combined = {}
            for direction, mask_name, res in results:
                if direction not in combined:
                    combined[direction] = {}
                combined[direction][mask_name] = res
            return combined



        def analyze_channel(channel):
            directions = ["horizontal", "vertical", "diag_down", "diag_up"]
            masks = {
                "M1": [1, 1, 1, 1],
                "M2": [1, -1, 1, -1],
            }

            result = {}

            for direction in directions:
                result[direction] = {}

                groups = self.extract_groups(channel, direction)
                for mname, mask in masks.items():
                    R, S = 0, 0

                    for group in groups:
                        flipped_group = self.apply_mask(group, mask)
                        orig_smooth = self.group_smoothness(group)
                        flip_smooth = self.group_smoothness(flipped_group)

                        if flip_smooth > orig_smooth:
                            S += 1
                        elif flip_smooth < orig_smooth:
                            R += 1
                        # Equal – ignorujemy

                    result[direction][mname] = {
                        "regular": R,
                        "singular": S,
                        "R/S ratio": round(R / S, 3) if S else float("inf")
                    }

            return result

        results = {}
        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            channels = cv2.split(self.img)
            names = ['Blue', 'Green', 'Red']
            for ch, name in zip(channels, names):
                results[name] = analyze_channel_parallel(self,ch)
            #gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            #results['Grayscale'] = analyze_channel_parallel(gray)
        else:
            results['Grayscale'] = analyze_channel_parallel(self.img)
        return results

    def extract_groups(self,channel, direction):
            h, w = channel.shape
            groups = []

            if direction == "horizontal":
                for i in range(h):
                    for j in range(0, w - 3):
                        groups.append(channel[i, j:j+4])
            elif direction == "vertical":
                for i in range(0, h - 3):
                    for j in range(w):
                        groups.append(channel[i:i+4, j])
            elif direction == "diag_down":  # ↘
                for i in range(h - 3):
                    for j in range(w - 3):
                        group = np.array([channel[i + k, j + k] for k in range(4)])
                        groups.append(group)
            elif direction == "diag_up":  # ↙
                for i in range(3, h):
                    for j in range(w - 3):
                        group = np.array([channel[i - k, j + k] for k in range(4)])
                        groups.append(group)

            return groups
    
    def group_smoothness(self,group):
            return sum(abs(int(group[i]) - int(group[i+1])) for i in range(len(group) - 1))

    def apply_mask(self,group, mask):
        return np.array([self.lsb_flip(p) if m == 1 else p for p, m in zip(group, mask)])

    def lsb_flip(self,p):
        return p ^ 1
    
    def process_channel_direction_mask(self,args):
            channel, direction, mask_name, mask = args
            groups = self.extract_groups(channel, direction)
            R, S = 0, 0
            for group in groups:
                flipped_group = self.apply_mask(group, mask)
                orig_smooth = self.group_smoothness(group)
                flip_smooth = self.group_smoothness(flipped_group)

                if flip_smooth > orig_smooth:
                    S += 1
                elif flip_smooth < orig_smooth:
                    R += 1
            return direction, mask_name, {"regular": R, "singular": S, "R/S ratio": round(R / S, 3) if S else float('inf')}

    

    def is_stego_suspected_rs(self, threshold=0.1):
        result = self.rs_analysis()

        for channel_name, directions in result.items():
            for direction_name, models in directions.items():
                for model_name, values in models.items():
                    rs_ratio = values.get("R/S ratio")
                    if rs_ratio is not None and (1 - threshold) < rs_ratio < (1 + threshold):
                        print(f"Podejrzany kanał: {channel_name}, kierunek: {direction_name}, R/S ratio: {rs_ratio}")
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

            # Normalizacja względem liczby pikseli
            num_pixels = channel.shape[0] * channel.shape[1]
            pov_per_pixel = pov_diff_sum / num_pixels if num_pixels > 0 else 0

            return {
                "PoV Difference Sum": int(pov_diff_sum),
                "PoV Normalized": float(pov_per_pixel),
                "Channel": name
            }

        results = {}
        if image_path is None:
            image = self.img
        else:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image is None:
            print(f" Błąd wczytywania obrazu: {image_path}")
            return results

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

    

    def calculate_lsb_entropy(self, image_path, channel='all', verbose=False):

        print(f"{image_path}")
        # Wczytaj obraz
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")

        if len(img.shape) != 3 or img.shape[2] < 3:
            raise ValueError("Obraz nie jest w formacie kolorowym RGB!")

        # Wyciągnij kanały
        b, g, r = cv2.split(img)


        # Funkcja wyciągająca LSB
        def extract_lsb(channel_data):
            return channel_data & 1  # operacja bitowa: zostawia tylko najmłodszy bit

        # Przygotuj dane do analizy
        if channel == 'r':
            lsb_data = extract_lsb(r)
        elif channel == 'g':
            lsb_data = extract_lsb(g)
        elif channel == 'b':
            lsb_data = extract_lsb(b)
        elif channel == 'all':
            lsb_data = np.concatenate([
                extract_lsb(r).flatten(),
                extract_lsb(g).flatten(),
                extract_lsb(b).flatten()
            ])
        else:
            raise ValueError("Kanał musi być jednym z: 'r', 'g', 'b', 'all'.")

        # Liczenie histogramu bitów (ile 0, ile 1)
        counts = np.bincount(lsb_data.flatten(), minlength=2)
        probs = counts / counts.sum()  # normalizacja do 1

        # Entropia
        lsb_entropy = entropy(probs, base=2)

        # Normalizacja entropii (bo maksymalna dla dwóch stanów = 1)
        normalized_entropy = lsb_entropy  # już jest w zakresie 0–1

        if verbose:
            print(f"Obraz: {image_path}")
            print(f"Liczba bitów: {counts.sum()}")
            print(f"Rozkład bitów: 0 -> {counts[0]}, 1 -> {counts[1]}")
            print(f"Entropia LSB: {normalized_entropy:.4f}")

        return normalized_entropy


    def pov_complementary_true(self, image_path=None):
        def is_regular(p1, p2):
            diff = abs(int(p1) - int(p2))
            return diff != 1

        def flip_pair(p1, p2):
            # Flip one up, the other down (simulate embedding)
            if p1 < 255 and p2 > 0:
                return p1 + 1, p2 - 1
            elif p1 > 0 and p2 < 255:
                return p1 - 1, p2 + 1
            return p1, p2

        def analyze_channel(channel):
            height, width = channel.shape
            original_regular = 0
            original_singular = 0
            flipped_regular = 0
            flipped_singular = 0

            # Przesuwamy po poziomych parach
            for i in range(height):
                for j in range(width - 1):
                    p1 = channel[i, j]
                    p2 = channel[i, j + 1]

                    if is_regular(p1, p2):
                        original_regular += 1
                    else:
                        original_singular += 1

                    # Teraz flipujemy i klasyfikujemy ponownie
                    fp1, fp2 = flip_pair(p1, p2)
                    if is_regular(fp1, fp2):
                        flipped_regular += 1
                    else:
                        flipped_singular += 1

            # Oblicz R/S
            rs_ratio = original_regular / original_singular if original_singular != 0 else 0
            flipped_rs_ratio = flipped_regular / flipped_singular if flipped_singular != 0 else 0

            return {
                "Original Regular": original_regular,
                "Original Singular": original_singular,
                "Original R/S": rs_ratio,
                "Flipped Regular": flipped_regular,
                "Flipped Singular": flipped_singular,
                "Flipped R/S": flipped_rs_ratio
            }

        results = {}
        image = self.img if image_path is None else cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print("Nie udało się wczytać obrazu.")
            return results

        if len(image.shape) == 3 and image.shape[2] == 3:
            channels = cv2.split(image)
            names = ['Blue', 'Green', 'Red']
            for ch, name in zip(channels, names):
                results[name] = analyze_channel(ch)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            results['Grayscale'] = analyze_channel(gray)

        return results



    def is_stego_suspected_pov(self, threshold=2):
        height, width = self.img.shape[:2]
        max_dim = max(width, height)
        match max_dim:
            case h if h <= 250 and h > 0:
                folder_path = "grouped_by_resolution\\res_1_250"
            case h if h <= 500 and h > 250:
                folder_path = "grouped_by_resolution\\res_251_500"
            case h if h <= 750 and h > 500:
                folder_path = "grouped_by_resolution\\res_501_750"
            case h if h <= 1000 and h > 750:
                folder_path = "grouped_by_resolution\\res_751_1000"
            case h if h <= 1250 and h > 1000:
                folder_path = "grouped_by_resolution\\res_1001_1250"
            case h if h <= 1500 and h > 1250:
                folder_path = "grouped_by_resolution\\res_1251_1500"
            case h if h <= 1750 and h > 1500:
                folder_path = "grouped_by_resolution\\res_1501_1750"
            case h if h > 1750:
                folder_path = "grouped_by_resolution\\res_1751_2000"
            case _:
                raise ValueError("Nieprawidłowy rozmiar obrazu. Rozmiar musi być większy od 0.")
        path = os.path.join(folder_path, "summary.json")
        while True:
            if not os.path.exists(f"{path}"):
                print("brak summary, generuje summary")
                ValueError("Nie można znaleźć summary.json. Proszę najpierw uruchomić process_image_directory.")
            else:
                with open (f"{path}", 'r') as summar:
                    reference_stats = json.load(summar)
                break
        
        PoV = self.pov_analysis()
        suspicion_score = 0
        for key in PoV:
            PoV_mean = reference_stats[key]["mean"]
            PoV_std = reference_stats[key]["std"]
            #print(f"Podejrzany kanał: {key}")
            #print("Pov Mean: " + f"{PoV_mean}")
            #print("Pov Std: " + f"{PoV_std}")
            #print("Wartość: " + f"{PoV[key]['PoV Normalized']}")
            #print("Diff: " + f"{abs(PoV[key]['PoV Normalized'] - PoV_mean)}")
            #print("Próg: " + f"{threshold * PoV_std}")
            if abs(PoV[key]['PoV Normalized'] - PoV_mean) > threshold * PoV_std:
                suspicion_score += 1
        
        # Jeśli więcej niż 2 z 4 cech są podejrzane – uznajemy obraz za potencjalnie stego
        print("Sus score:" + str(suspicion_score))
        return suspicion_score >=1
    



    def analaze_spa_block(self, image_path=None, threshold=0.05, block_size=(64, 64)):
        
        @njit  # Przyspieszenie obliczeń dzięki kompilacji do kodu maszynowego
        def estimate_spa_p_block_numba(channel, block_height, block_width):
            Z = 0
            C = 0
            rows, cols = channel.shape

            for i in range(0, rows, block_height):
                for j in range(0, cols, block_width):
                    block = channel[i:i + block_height, j:j + block_width]
                    bh, bw = block.shape

                    for x in range(bh - 1):
                        for y in range(1, bw - 1):  # zaczynamy od 1 by uniknąć indeksu -1 przy skosie

                            p1 = block[x, y]
                            # poziom i pion
                            p2 = block[x + 1, y]
                            p3 = block[x, y + 1]
                            # skosy
                            p4 = block[x + 1, y + 1]
                            p5 = block[x + 1, y - 1]

                            # Pionowe i poziome porównania
                            if abs(p1 - p2) == 1:
                                C += 1
                            elif p1 == p2:
                                Z += 1

                            if abs(p1 - p3) == 1:
                                C += 1
                            elif p1 == p3:
                                Z += 1

                            # Skośne porównania
                            if abs(p1 - p4) == 1:
                                C += 1
                            elif p1 == p4:
                                Z += 1

                            if abs(p1 - p5) == 1:
                                C += 1
                            elif p1 == p5:
                                Z += 1

            T = Z + C
            if T == 0:
                return 0.0

            a = 2 * T
            b = -2 * T + 4 * Z
            c = T - 4 * Z + 4 * C

            D = b * b - 4 * a * c
            if D < 0:
                return 0.0

            sqrt_D = np.sqrt(D)
            p1 = (-b + sqrt_D) / (2 * a)
            p2 = (-b - sqrt_D) / (2 * a)
            candidates = [p for p in (p1, p2) if 0 <= p <= 1]

            if len(candidates) == 0:
                return 0.0

            return min(candidates[0], 1.0)

        # ---- Główna część ----
        results = {}
        image = self.img if image_path is None else cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print("Nie udało się wczytać obrazu.")
            return None

        block_height, block_width = block_size

        if len(image.shape) == 3 and image.shape[2] == 3:
            channels = cv2.split(image)
            names = ['Blue_SPA', 'Green_SPA', 'Red_SPA']
            for ch, name in zip(channels, names):
                print(f"\nAnaliza kanału: {name}")
                p_value = estimate_spa_p_block_numba(ch.astype(np.int16), block_height, block_width)
                print(f"Wartość p dla {name}: {p_value}")
                print(f"Próg: {threshold}")
                if p_value > threshold:
                    results[name] = p_value
                else:
                    print(f"Wartość p poniżej progu ({threshold}). Nie dodano do wyników.")
        else:
            print("\nAnaliza w skali szarości:")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            p_value_gray = estimate_spa_p_block_numba(gray.astype(np.int16), block_height, block_width)
            if p_value_gray > threshold:
                results['Grayscale_SPA'] = p_value_gray
            else:
                print(f"Wartość p poniżej progu ({threshold}). Nie dodano do wyników.")

        return results




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
        #artifacts_anomalys = self.analyze_artifacts(dct_blocks)
        
        #anomaly = self.is_stego_suspected_dct(dct_anomalys, artifacts_anomalys)
        print(dct_anomalys)#,artifacts_anomalys)
        #print(anomaly)
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
        #print(f"Rozmiar obrazu: {height}x{width}")
        max_dim = max(width, height)
        #print(f"Max dim: {max_dim}")
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
                #print("=====================")
                #print("Próg: "+ str(stats[channel]['DCT_MeanAbsDiff_Mean'] + 2 * stats[channel]['DCT_MeanAbsDiff_Std']))
                #print (report[channel])
                #print("DCT ABSmean: " + str(abs_mean))
                #print("=====================")
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



def calculate_noise_level(image_path):
    # Wczytaj obraz w trybie szarości (jedno kanałowy)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Obraz nie został wczytany poprawnie")
    
    # Oblicz różnice między pikselami w poziomie i pionie
    diff_horizontal = np.diff(img, axis=1)  # Różnice w poziomie (wzdłuż osi x)
    diff_vertical = np.diff(img, axis=0)    # Różnice w pionie (wzdłuż osi y)
    
    # Jeśli różnica w liczbie kolumn to 1, dodajemy kolumnę zer do diff_vertical\
    if diff_vertical.shape[1] > diff_horizontal.shape[1]:

        # Dodajemy kolumnę zer na końcu
        diff_horizontal = np.hstack([diff_horizontal, np.zeros((diff_horizontal.shape[0], 1))])

    # Jeśli różnica w liczbie wierszy to 1, dodajemy ostatni wiersz do diff_vertical
    if diff_vertical.shape[0] < diff_horizontal.shape[0]:

        diff_vertical = np.vstack([diff_vertical, diff_vertical[-1, :]])

    # Połącz obie różnice w jedną macierz
    diff_combined = np.concatenate((diff_horizontal, diff_vertical), axis=0)
    
    # Oblicz odchylenie standardowe tych różnic (poziom szumu)
    noise_level = np.std(diff_combined)
    
    return noise_level


def analyze_histogram(image_path):
    # Wczytanie obrazu w trybie szarości
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Obraz nie został wczytany poprawnie")
    
    # Obliczenie histogramu
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    # Normalizacja histogramu
    hist = hist / hist.sum()
    
    # Wyświetlanie histogramu
    plt.figure(figsize=(8, 6))
    plt.plot(hist)
    plt.title("Histogram obrazu")
    plt.xlabel("Poziomy szarości")
    plt.ylabel("Prawdopodobieństwo")
    plt.show()

    return hist

def compare_histograms(hist1, hist2):
    # Obliczanie różnicy pomiędzy dwoma histogramami (np. na podstawie odległości Manhattan)
    diff = np.sum(np.abs(hist1 - hist2))
    
    return diff


if __name__ == "__main__":
    #analyze_image("stego\\LSB w JPG\\AAAAA.jpg")
    #analyze_image("normal\\pi9UmWpTId0-unsplash.jpg")

    #analyze_image("modules\\Cat_November_2010-1.jpg")
    #analyze_image("stego\\LSB w JPG\\cat.jpg")
    #analyze_image("modules\\testjpg.jpg")
    #analyze_image("stego\LSB w JPG\YaK5Short.jpg")

    #test.png i hidden2.png , YaK5Short.png nie wykrywalne
    #test.png,hidden2 nie bedzie wykrywalne przez POV zbyt mała ingenrencja

    #analyze_image("stego\\LSB w BMP\\autohiden4.png")
    #analyze_image("normal\\auto.png")

    #analyze_image("normal\IMG_105540755_HDR.bmp")
    #print("======================================")
    #analyze_image("stego\LSB w BMP\hidden1.bmp")


    #analyze_image("stego\LSB w JPG\kwiatek000.jpg")
    #print("================aaaaaaaaaaaaaaaaaaaaaaaaaaaaa======================")

    analyze_image("stego\\LSB_w_BMP\\snowboard005.png")

    analyze_image("normal\-dNN6qAo9ptY-unsplash.png")

    analyze_image("normal\cv8cepOsAg-unsplash.png")
    #print("======================================")
    #analyze_image("stego\LSB w BMP\hidden1.bmp")
    #folder_path1 = "grouped_by_resolution"
    #Image.analyze_pov_outliers(pathr = "grouped_by_resolution\\res_1_250", output_folder= "outliers")


    #Image = PngAnalyzer("modules\\zakodowanylsb.png")
    #print(Image.calculate_lsb_entropy("normal\\cd566e20-3851-4b88-9960-268250b88302.png"))
    #print(Image.calculate_lsb_entropy("stego\\LSB w BMP\\hidden3.png"))
    #print(calculate_noise_level("normal\\cd566e20-3851-4b88-9960-268250b88302.png"))
    #print(calculate_noise_level("stego\\LSB w BMP\\hidden3.png"))
    #Image.pov_analysis("normal\\cd566e20-3851-4b88-9960-268250b88302.png")
    #Image.pov_analysis("stego\\LSB w BMP\\hidden3.png")
    #print(Image.calculate_lsb_entropy("normal\\6zxkS9PVE-unsplash.png"))
    #a=analyze_histogram("normal\\6zxkS9PVE-unsplash.png")
    #aa=analyze_histogram("stego\\LSB w BMP\\hidden3.png")
    #b=analyze_histogram("normal\\cd566e20-3851-4b88-9960-268250b88302.png")
    #difference = compare_histograms(aa, b)
    #print(f"Różnica między histogramami: {difference}")

    #difference = compare_histograms(aa, b)
    #print(f"Różnica między histogramami: {difference}")
    #folder_path2 = "grouped_precise_v2\\B0_E6.8"
    #folder_path3 = "grouped_advanced\\bright_high_entropy_high_contrast_textured_green_dominant"
    #Image.process_image_directory(f"{folder_path2}")
    #Image.process_image_directory(f"{folder_path3}")

    #folder_path1 = "grouped_by_entropy_brightness"
    #Image = PngAnalyzer("modules\\zakodowanylsb.png")
    #for foldername in os.listdir(folder_path1):
    #    folder_path2 = os.path.join(folder_path1, foldername)
    #    Image.process_image_directory(f"{folder_path2}")




    #for images in os.listdir("normal"):
    #    print(images)
    #    if images.lower().endswith(('.png', '.bmp')):
    #        analyze_image(os.path.join("normal", images))
    #    else:
    #        continue
        
        