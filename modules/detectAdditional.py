# EXIF (Exchangeable Image File Format) - to metadane osadzone w obrazach, zawierające informacje o aparacie, ustawieniach, dacie wykonania zdjęcia,
# a czasem również o lokalizacji GPS. Mogą również zawierać dane edycyjne, takie jak program używany do obróbki zdjęcia.

import os
from PIL import Image


def check_exif_data(image_path):
    """Checks for the presence of EXIF metadata in an image"""
    try:
        with Image.open(image_path) as img:
            exif_data = img.info.get("exif")  # A safer method to retrieve EXIF data
            return bool(exif_data)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def check_trailing_data(image_path):
    """Checks for extra data after the JPEG end marker (0xFFD9) or PNG end marker (IEND)"""
    try:
        with open(image_path, 'rb') as f:
            content = f.read()

            if image_path.lower().endswith(('.jpg', '.jpeg')):
                eoi_pos = content.find(b'\xff\xd9')
                if eoi_pos == -1:
                    return False  # Not a JPEG or a corrupted file
                trailing_data = content[eoi_pos + 2:]
                return len(trailing_data) > 0 and not all(
                    b == 0 or b == 32 for b in trailing_data)  # Ensure non-empty, meaningful data

            elif image_path.lower().endswith('.png'):
                iend_pos = content.rfind(b'IEND')
                if iend_pos == -1:
                    return False  # Not a valid PNG file
                trailing_data = content[iend_pos + 8:]  # IEND chunk is 8 bytes long
                return len(trailing_data) > 0 and not all(b == 0 or b == 32 for b in trailing_data)

            return False
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return False


def scan_images(directory):
    """Scans a directory and checks images for extra data"""
    results = {}

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            full_path = os.path.join(directory, filename)

            exif_detected = check_exif_data(full_path)
            trailing_detected = check_trailing_data(full_path) if filename.lower().endswith(
                ('.jpg', '.jpeg', '.png')) else False

            if exif_detected or trailing_detected:
                results[filename] = {
                    'EXIF': exif_detected,
                    'Trailing_data': trailing_detected
                }

    return results


if __name__ == "__main__":
    scan_directory = "../stego/nadmiarowe dane"

    if not os.path.isdir(scan_directory):
        print("The specified directory does not exist!")
        exit()

    results = scan_images(scan_directory)

    if results:
        print("\nImages with additional data detected:")
        for filename, data in results.items():
            print(f"\nFile: {filename}")
            print(f"EXIF Data: {'Yes' if data['EXIF'] else 'No'}")
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Trailing Data: {'Yes' if data['Trailing_data'] else 'No'}")
    else:
        print("\nNo extra data detected in images.")
