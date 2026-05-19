# Simple Gaussian Blur mit OpenCL

OpenCL-basierter Gaussian Blur Filter fuer die HPC-Vorlesung (SS2026, FH Technikum Wien).

Das Programm liest ein TGA-Bild ein, wendet einen Gaussian Blur auf der GPU an und speichert das Ergebnis als BMP-Datei.

## Voraussetzungen

Bevor das Projekt kompiliert werden kann, muessen zwei Dinge installiert sein:

### 1. Visual Studio mit C++ Unterstuetzung

- **Visual Studio 2022 oder neuer** (Community Edition reicht)
- Im Visual Studio Installer die Workload **"Desktopentwicklung mit C++"** auswaehlen und installieren
- Das liefert den MSVC Compiler (`cl.exe`), Linker und die C++ Standard-Header

### 2. OpenCL SDK (Intel oneAPI)

- **Intel oneAPI Base Toolkit** installieren: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
- Bei der Installation reicht die Komponente **"Intel oneAPI DPC++/C++ Compiler"** — diese enthaelt die OpenCL Header und Libraries
- Nach der Installation sollten folgende Pfade existieren:
  - Header: `C:\Program Files (x86)\Intel\oneAPI\<version>\include\CL\cl.h`
  - Library: `C:\Program Files (x86)\Intel\oneAPI\<version>\lib\OpenCL.lib`

### 3. GPU mit OpenCL-Unterstuetzung

- Jede halbwegs aktuelle AMD, NVIDIA oder Intel GPU sollte funktionieren
- Falls keine GPU verfuegbar ist, faellt das Programm automatisch auf die CPU zurueck
- Die jeweiligen GPU-Treiber muessen installiert sein (enthalten den OpenCL Runtime)

## Build

### build.bat anpassen

Die Datei `build.bat` enthaelt die Pfade zu Visual Studio und dem OpenCL SDK. Diese muessen ggf. an die eigene Installation angepasst werden:

```bat
@echo off

rem === PFADE ANPASSEN ===
rem Pfad zu vcvarsall.bat (Visual Studio)
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

rem Pfad zum OpenCL SDK (Intel oneAPI)
set OPENCL_INC=C:\Program Files (x86)\Intel\oneAPI\2026.0\include
set OPENCL_LIB=C:\Program Files (x86)\Intel\oneAPI\2026.0\lib

rem === AB HIER NICHTS AENDERN ===
cl.exe /EHsc /W3 /I"%OPENCL_INC%" /Icpptga main.cpp cppTga\tga.cpp /Fe:gaussian_blur.exe /link "%OPENCL_LIB%\OpenCL.lib"
```

**So findet man die richtigen Pfade:**

- **vcvarsall.bat**: Im Visual Studio Installationsordner unter `VC\Auxiliary\Build\`. Typische Pfade:
  - `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat`
  - `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat`
- **OpenCL SDK**: Im Intel oneAPI Ordner. Die Versionsnummer (`2026.0`) muss an die installierte Version angepasst werden.

### Kompilieren

Einfach `build.bat` per Doppelklick ausfuehren oder in der Kommandozeile:

```
build.bat
```

Bei Erfolg wird `gaussian_blur.exe` erstellt.

## Ausfuehrung

Das Programm wird im selben Verzeichnis ausgefuehrt, in dem auch `shuttle.tga` und `gaussian_blur.cl` liegen:

```
gaussian_blur.exe [input.tga] [filterSize] [sigma]
```

Alle Parameter sind optional:

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| input.tga | shuttle.tga | Eingabebild im TGA-Format (24 oder 32 Bit) |
| filterSize | 5 | Groesse des Gauss-Filters (1, 3, 5, 7 oder 9 — muss ungerade sein) |
| sigma | 1.0 | Standardabweichung der Gauss-Funktion (hoeher = staerker verwischt) |

### Beispiele

```
gaussian_blur.exe                          # Default: shuttle.tga, 5x5, sigma 1.0
gaussian_blur.exe shuttle.tga 3 1.0        # kleiner Filter, leichte Weichzeichnung
gaussian_blur.exe shuttle.tga 9 2.0        # grosser Filter, starke Weichzeichnung
```

Das Ergebnis wird als `output.bmp` gespeichert und kann direkt in Windows per Doppelklick geoeffnet werden.

## Projektstruktur

```
├── main.cpp                  Host-Programm (OpenCL Setup, Bild I/O, Kernel-Ausfuehrung)
├── gaussian_blur.cl          OpenCL Kernel (Blur-Berechnung auf der GPU)
├── build.bat                 Build-Script
├── cppTga/                   TGA Bild-Library (Laden/Speichern)
│   ├── tga.h
│   └── tga.cpp
├── shuttle.tga               Test-Eingabebild
└── gauss_filter_kernel_generator.cpp   Hilfsprogramm zum Generieren von Gauss-Kernels
```

## Wie funktioniert das Programm?

### Ablauf

1. **TGA-Bild laden** — Das Eingabebild wird ueber die cppTga-Library gelesen. Die Pixeldaten landen als Byte-Array im Speicher (RGB, 3 Bytes pro Pixel).

2. **Gauss-Filter berechnen** — Ein 2D Gauss-Kernel wird basierend auf Groesse und Sigma generiert. Die Werte werden normalisiert (Summe = 1), damit die Gesamthelligkeit des Bildes erhalten bleibt.

3. **OpenCL initialisieren** — Das Programm sucht eine GPU (oder faellt auf CPU zurueck), erstellt einen OpenCL Context und eine Command Queue.

4. **Daten auf die GPU kopieren** — Drei Buffer werden erstellt:
   - `inputBuffer`: das Eingabebild (read-only)
   - `outputBuffer`: Platz fuer das Ergebnis (write-only)
   - `filterBuffer`: die Gauss-Filter-Gewichte (read-only)

5. **Kernel ausfuehren** — Der OpenCL Kernel wird mit einem 2D NDRange (Breite x Hoehe) gestartet. Jeder Work-Item berechnet genau einen Pixel im Ausgabebild.

6. **Ergebnis zuruecklesen und speichern** — Die GPU-Daten werden zurueck in den Host-Speicher kopiert und als BMP gespeichert.

### Der OpenCL Kernel (`gaussian_blur.cl`)

Jeder Work-Item (= ein Pixel):
- Bestimmt seine Position im Bild ueber `get_global_id(0)` (x) und `get_global_id(1)` (y)
- Iteriert ueber alle Nachbarpixel im Filterbereich
- **Border Handling**: Wenn ein Nachbarpixel ausserhalb des Bildes liegt, wird der naechste gueltige Randpixel verwendet (Clamping)
- Berechnet fuer jeden Farbkanal (R, G, B) die gewichtete Summe
- Schreibt das Ergebnis in den Output-Buffer

## Troubleshooting

| Problem | Loesung |
|---------|---------|
| `'cl.exe' is not recognized` | `vcvarsall.bat`-Pfad in `build.bat` pruefen, C++ Workload in VS installiert? |
| `fatal error: 'CL/cl.h' file not found` | OpenCL SDK Pfad in `build.bat` pruefen |
| `OpenCL.lib not found` (Linker Error) | OpenCL SDK Library-Pfad in `build.bat` pruefen |
| `No OpenCL platforms found` | GPU-Treiber installieren/aktualisieren |
| `Failed to open gaussian_blur.cl` | Programm muss aus dem Projektverzeichnis gestartet werden |
| Bild sieht komisch aus | Nur unkomprimierte 24-Bit oder 32-Bit TGA-Dateien werden unterstuetzt |
