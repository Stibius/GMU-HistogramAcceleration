Název zadání:
        Urychlení zpracování obrazu založené na modifikaci histogramu

Jména øešitelù:
        Jan Vybíral, xvybir05
        Filip Zapletal, xzaple27
        Piják Lukáš, xpijak00

Popis problému:
        Zpracovat výpoèet histogramu a 3 metody zpracování obrazu
využívající tento histogram.

Popis cíle:
        Zpracujeme výpoèet histogramu vstupního obrázku pomocí OpenCL.
Potom každý èlen týmu zpracuje jednu metodu využívající tento histogram, a
to jak s pomocí OpenCL tak CPU implementaci. Výsledný program umožní vybrat
vstupní obrázek a metodu, kterou provede obìma zpùsoby, a vypíše èas
provedení každého z nich a zobrazí výstupní obrázek.

Zpùsob vyhodnocení:
        Dosažené urychlení OpenCL implementace oproti CPU implementaci a
korektnost výstupu.

Názvy použitých metod a zpùsob jejich využití:
        - výpoèet histogramu vstupního obrázku (z toho budou vycházet
všechny tøi následující metody)
        - ekvalizace histogramu - Jan Vybíral
        - segmentace obrazu na základì prahování histogramu (Image
Segmentation by Histogram Thresholding) - Filip Zapletal
           (doplòte vaše vybrané metody)
        - prahování s pomocí metody Otsu - Lukáš Piják

Nástroje:
        MS Visual Studio, C++, OpenCL, SDL (jako ve cvièeních)

Reference na èlánky a jiné zdroje:
        - http://en.wikipedia.org/wiki/Histogram_equalization
        - http://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf
        - http://fourier.eng.hmc.edu/e161/lectures/contrast_transform/node2.html
        - http://www.cis.temple.edu/~latecki/Courses/CIS750-03/Lectures/Venu_project2.ppt
        - http://en.wikipedia.org/wiki/Otsu's_method