SIGNATURE = b'STEGOv1'

def decrypt_message(data):
    return bytes([(b - 3) % 256 for b in data]).decode('utf-8', errors='ignore')

def extract_text_from_steganography(image_path):
    try:
        with open(image_path, 'rb') as img_file:
            data = img_file.read()

        pos = data.find(SIGNATURE)
        if pos == -1:
            print("[!] No hidden message found.")
            return None

        length_bytes = data[pos + len(SIGNATURE):pos + len(SIGNATURE) + 2]
        message_length = int.from_bytes(length_bytes, 'big')

        encrypted_message = data[pos + len(SIGNATURE) + 2 : pos + len(SIGNATURE) + 2 + message_length]

        return decrypt_message(encrypted_message)

    except Exception as e:
        print(f"[!] Error extracting: {e}")
        return None

def extract_stego_from_jpegx(jpeg_path):
    with open(jpeg_path, "rb") as f:
        data = f.read()
    signature = bytes([0x5B, 0x3B, 0x31, 0x53, 0x00])
    pos = data.find(signature)
    if pos == -1:
        return None
    warning_len = len(b"Warning! Modification of this file will result in no longer working. JPegX 1.0.6")
    start_len = pos + len(signature) + warning_len
    length_bytes = data[start_len:start_len+2]
    if len(length_bytes) < 2:
        return None
    msg_len = int.from_bytes(length_bytes, "little")
    encrypted_msg_start = start_len + 2
    encrypted_msg_end = encrypted_msg_start + msg_len
    encrypted_msg = data[encrypted_msg_start:encrypted_msg_end]
    return decrypt(encrypted_msg)

import os
from PIL import Image
from PIL.ExifTags import TAGS

def check_exif_data(image_path):
    """Return full EXIF metadata dictionary or None"""
    try:
        with Image.open(image_path) as img:
            exif_data = img.getexif()
            if not exif_data:
                return None
            exif_dict = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
            return exif_dict
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def decrypt(data):
    decrypted = []
    for i, b in enumerate(data):
        base = 0xBB + i
        res = b - base
        if res < 0:
            res = (res + 256) - 1
        decrypted.append(res & 0xFF)
    return bytes(decrypted).decode("utf-8", errors="ignore")

def check_trailing_data(image_path):
    """Check for extra data after JPEG end marker (0xFFD9) or PNG end marker (IEND)"""
    try:
        with open(image_path, 'rb') as f:
            content = f.read()

            def is_stego1_data(s):
                return s.startswith("STEGOv1") or s == "<Encrypted or binary data>"

            if image_path.lower().endswith(('.jpg', '.jpeg')):
                eoi_pos = content.find(b'\xff\xd9')
                if eoi_pos == -1:
                    return None
                trailing_data = content[eoi_pos + 2:]

                if trailing_data and not all(b in (0, 32) for b in trailing_data):
                    try:
                        decoded = trailing_data.decode('utf-8').strip()
                        if is_stego1_data(decoded):
                            stego_msg = extract_text_from_steganography(image_path)
                            return stego_msg if stego_msg else decoded
                        return decoded
                    except UnicodeDecodeError:
                        hidden_msg = extract_stego_from_jpegx(image_path)
                        if hidden_msg:
                            if is_stego1_data(hidden_msg):
                                stego_msg = extract_text_from_steganography(image_path)
                                return stego_msg if stego_msg else hidden_msg
                            return hidden_msg
                        stego_msg = extract_text_from_steganography(image_path)
                        if stego_msg:
                            return stego_msg
                        return "<Encrypted or binary data>"

            elif image_path.lower().endswith('.png'):
                iend_pos = content.rfind(b'IEND')
                if iend_pos == -1:
                    return None
                trailing_data = content[iend_pos + 8:]

                if trailing_data and not all(b in (0, 32) for b in trailing_data):
                    try:
                        decoded = trailing_data.decode('utf-8').strip()
                        return decoded
                    except UnicodeDecodeError:
                        return "<Encrypted or binary data>"

            else:
                return None

        return None
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def scan_image(image_path):
    if image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        filename = os.path.basename(image_path)
    else:
        print(f"Unsupported file format: {image_path}")
        return {}

    results = {}
    exif_data = check_exif_data(image_path)
    hidden_message = check_trailing_data(image_path)

    results[filename] = {
        'EXIF': exif_data,
        'Hidden Message': hidden_message
    }
    if exif_data:
        print("EXIF Data:")
        for key, value in exif_data.items():
            print(f"  {key}: {value}")
    else:
        print("EXIF Data: None")

    if hidden_message:
        print("Hidden Message: Found")
        if hidden_message == "<Encrypted or binary data>":
            print("  -> Possibly encrypted or binary data.")
        else:
            print(f"  Content: {hidden_message}")
    else:
        print("Hidden Message: None")

    return results

def scan_images(directory):
    """Scan a directory and analyze all images"""
    results = {}
    print(f"\nScanning files in: {directory}")

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            full_path = os.path.join(directory, filename)

            exif_data = check_exif_data(full_path)
            hidden_message = check_trailing_data(full_path)

            results[filename] = {
                'EXIF': exif_data,
                'Hidden Message': hidden_message
            }

            print(f"\nFile: {filename}")
            if exif_data:
                print("EXIF Data:")
                for key, value in exif_data.items():
                    print(f"  {key}: {value}")
            else:
                print("EXIF Data: None")

            if hidden_message:
                print("Hidden Message: Found")
                if hidden_message == "<Encrypted or binary data>":
                    print("  -> Possibly encrypted or binary data.")
                else:
                    print(f"  Content: {hidden_message}")
            else:
                print("Hidden Message: None")

    return results
