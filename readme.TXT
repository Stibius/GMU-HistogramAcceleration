Název zadání:
        Urychlení zpracování obrazu zaloené na modifikaci histogramu

Jména řeitelů:
        Jan Vybíral, xvybir05
        Filip Zapletal, xzaple27
        Piják Luká, xpijak00

Popis problému:
        Zpracovat výpočet histogramu a 3 metody zpracování obrazu
vyuívající tento histogram.

Popis cíle:
        Zpracujeme výpočet histogramu vstupního obrázku pomocí OpenCL.
Potom kadý člen týmu zpracuje jednu metodu vyuívající tento histogram, a
to jak s pomocí OpenCL tak CPU implementaci. Výsledný program umoní vybrat
vstupní obrázek a metodu, kterou provede oběma způsoby, a vypíe čas
provedení kadého z nich a zobrazí výstupní obrázek.

Způsob vyhodnocení:
        Dosaené urychlení OpenCL implementace oproti CPU implementaci a
korektnost výstupu.

Názvy pouitých metod a způsob jejich vyuití:
        - výpočet histogramu vstupního obrázku (z toho budou vycházet
vechny tři následující metody)
        - ekvalizace histogramu - Jan Vybíral
        - segmentace obrazu na základě prahování histogramu (Image
Segmentation by Histogram Thresholding) - Filip Zapletal
           (doplňte vae vybrané metody)
        - prahování s pomocí metody Otsu - Luká Piják

Nástroje:
        MS Visual Studio, C++, OpenCL, SDL (jako ve cvičeních)

Reference na články a jiné zdroje:
        - http://en.wikipedia.org/wiki/Histogram_equalization
        - http://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf
        - http://fourier.eng.hmc.edu/e161/lectures/contrast_transform/node2.html
        - http://www.cis.temple.edu/~latecki/Courses/CIS750-03/Lectures/Venu_project2.ppt
        - http://en.wikipedia.org/wiki/Otsu's_method
        
Dobry den,
Predpokladam ze na vypoctu histogramu budete pracovat vsetci. Vyhybajte sa
prosim pri rieseni pouziti atomickych instrukcii - prip. ich pouzite co
najmenej. Atomicky increment v globalnej pameti nestaci do projektu ako
"vypocet histogramu". Vypocet histogramu ked to riesite traja, by malo byt
co najoptimalnejsi!
Co sa tyka literatury, na MGM u projektoch aj u diplomkach Wikipedia nie je
prijatelny zdroj. Na wiki strankach dole mate aj pouzitelne odkazy - bud
pouzivajte tie alebo najdete vedecke clanky alebo knihy. To iste plati aj na
prezentacie.
Pri vyhodnoteni vytvorte aj grafy v zavislosti na poctu pixelov.
S pozdravom,

ISz 