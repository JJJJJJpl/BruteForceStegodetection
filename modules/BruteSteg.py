import argparse
from detectAdditional import scan_image as run_additional
from detectStatistical import analyze_image as run_statistical
from bruteforce_png_threaded import run as run_brute

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
        help="Uruchamia próbę odczytania steganograficznie ukrytych danych. Dodaje flagi: w,dt,mw,v,all"
    )

    check_parser.add_argument(
        "-w", "--window", type=int,
        help="Określa szerokość okna dla funkcji wykrywającej tekst. Wymaga -b. Domyślnie 20."
    )
    check_parser.add_argument(
        "-dt", "--treshold", type=float,
        help="Określa czułość dla funkcji wykrywającej tekst. Wymaga -b. Domyślnie 0.3."
    )
    check_parser.add_argument(
        "-mw", "--max_workers", type=int,
        help="Okresla ilość wątków które użyje bruteforce. Wymaga -b. Domyślnie 4."
    )
    check_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Bruteforce będzie wypisywał postępy podczas pracy."
    )
    check_parser.add_argument(
        "-all", "--print_all", action="store_true",
        help="Bruteforce nie będzie używał funkcji wykrywającej tekst, tylko zwróci wszystkie kombinacje. Dział identycznie co -dt 0.0."
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
            window = 20
            treshold = 0.3

            all = args.print_all
            if all:
                if args.window is not None: print("Uwaga: Ignoruję -w przez -all")
                if args.treshold is not None: print("Uwaga: Ignoruję -dt przez -all")
            else:
                
                if args.window is not None:
                    if args.window <= 1:
                        print("Uwaga: Ignoruję -w ponieważ musi być > 1")
                    else:
                        window = args.window
                
                if args.treshold is not None:
                    if args.treshold < 0.0 or args.treshold > 1.0:
                        print("Uwaga: Ignoruję -dt ponieważ musi być w zakresie (0.0, 1.0)")
                    else:
                        treshold = args.treshold
            
            max_workers = 4
            if args.max_workers is not None:
                if args.max_workers < 1:
                    print("Uwaga: Ignoruję -mw ponieważ musi być > 0")
                else:
                    max_workers = args.max_workers
            
            silent = True
            if args.verbose: silent = False

            print("Uruchamiam BruteForce...")
            run_brute(args.image_path,window,treshold,max_workers,silent,all)
        else:
            if args.window is not None: print("Uwaga: Ignoruję -w przez brak -b")
            if args.treshold is not None: print("Uwaga: Ignoruję -dt przez brak -b")
            if args.max_workers is not None: print("Uwaga: Ignoruję -mw przez brak -b")
            if args.verbose: print("Uwaga: Ignoruję -v przez brak -b")
            if args.print_all: print("Uwaga: Ignoruję -all przez brak -b")

        if not any([args.additional, args.statistical, args.brute]):
            print("Nie wybrano żadnej metody analizy. Użyj -a, -s lub -b.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
