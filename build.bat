@echo off

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

set OPENCL_INC=C:\Program Files (x86)\Intel\oneAPI\2026.0\include
set OPENCL_LIB=C:\Program Files (x86)\Intel\oneAPI\2026.0\lib

cl.exe /EHsc /W3 /I"%OPENCL_INC%" /Icpptga main.cpp cppTga\tga.cpp /Fe:gaussian_blur.exe /link "%OPENCL_LIB%\OpenCL.lib"

if %ERRORLEVEL% EQU 0 (
    echo Build successful: gaussian_blur.exe
) else (
    echo Build failed!
)
