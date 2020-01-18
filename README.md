# GMU - Grafické a multimediální procesory - Urychlení zpracování obrazu založené na modifikaci histogramu

### Jména řešitelů:

- Jan Vybíral, xvybir05
- Filip Zapletal, xzaple27
- Piják Lukáš, xpijak00

### Zadání:

Implementujte výpočet histogramu pomocí CUDA/OpenCL a jednu až tři (v závislosti na počtu řešitelů) metody zpracování obrazu, které využívají histogram. Výsledky porovnejte s CPU implementací (například OpenCV).
Dokumentace by měla obsahovat způsob a zdůvodnění paralelizace a přizpůsobení algoritmu pro výpočet na grafické kartě.

### Popis cíle:

Zpracujeme výpočet histogramu vstupního obrázku pomocí OpenCL.
Potom každý člen týmu zpracuje jednu metodu využívající tento histogram, a
to jak s pomocí OpenCL tak CPU implementaci. Výsledný program umožní vybrat
vstupní obrázek a metodu, kterou provede oběma způsoby, a vypíše čas
provedení každého z nich a zobrazí výstupní obrázek.

### Způsob vyhodnocení:

Dosažené urychlení OpenCL implementace oproti CPU implementaci a
korektnost výstupu.

### Názvy použitých metod a způsob jejich využití:

- výpočet histogramu vstupního obrázku (z toho budou vycházet všechny tři následující metody)
- ekvalizace histogramu - Jan Vybíral
- segmentace obrazu na základě prahování histogramu (Image Segmentation by Histogram Thresholding) - Filip Zapletal
- prahování s pomocí metody Otsu - Lukáš Piják

### Nástroje:

- MS Visual Studio
- C++
- OpenCL
- SDL (jako ve cvičeních)

### Reference na články a jiné zdroje:

- http://en.wikipedia.org/wiki/Histogram_equalization
- http://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf
- http://fourier.eng.hmc.edu/e161/lectures/contrast_transform/node2.html
- http://www.cis.temple.edu/~latecki/Courses/CIS750-03/Lectures/Venu_project2.ppt
- http://en.wikipedia.org/wiki/Otsu's_method
        
### Dokumentace:

[doc/dokumentace.pdf](doc/dokumentace.pdf)

### Hodnocení: 

**23/28**
