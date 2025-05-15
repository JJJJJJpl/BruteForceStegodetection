import argparse
from detectAdditional import scan_image as run_additional
from detectStatistical import analyze_image as run_statistical
from bruteforce import run as run_brute

def main():
    parser = argparse.ArgumentParser(
        description="CLI do uruchamiania modułów analizy zdjęć"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Komenda "check"
    check_parser = subparsers.add_parser(
        "check", help="Wykonuje analizę na podanym zdjęciu, po wiecej infiormacji użyj check -h"
    )
    check_parser.add_argument(
        "image_path", type=str, help="Ścieżka do zdjęcia"
    )
    check_parser.add_argument(
        "-a", "--additional", action="store_true",
        help="Uruchamia analizę nadmiarowych danych w celu wykrycia steganografii"
    )
    check_parser.add_argument(
        "-s", "--statistical", action="store_true",
        help="Uruchamia analizę statystyczną w celu wykrycia steganografii"
    )
    check_parser.add_argument(
        "-b", "--brute", action="store_true",
        help="Uruchamia próbę wykrycia steganografii metodą brute force i odczytanie wiadomości"
    )

    args = parser.parse_args()

    if args.command == "check":
        print(f"Analiza zdjęcia: {args.image_path}")

        if args.additional:
            print("Uruchamiam DetectAditional...")
            run_additional(args.image_path)

        if args.statistical:
            print("Uruchamiam DetectStatistical...")
            run_statistical(args.image_path)

        if args.brute:
            print("Uruchamiam BruteForce...")
            run_brute(args.image_path, 2)

        if not any([args.additional, args.statistical, args.brute]):
            print("Nie wybrano żadnej metody analizy. Użyj -a, -s lub -b.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
