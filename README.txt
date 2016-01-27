Juraj Holas
<<<<<<< HEAD
Projekt c.2 - Viacvrstvovy perceptron
Neuronove siete, 2014/2015

Obsah archivu:
-> zdrojove subory k programu su v priecinku src/. Program nevyuziva ziadne externu kniznicu EJML, ktorej .jar subory su v ejml/.
-> v priecinku data/ su okrem vstupnych suborov aj hrube data z vysledkov vsetkych testovani
-> v subore grafy.xlsx su vsetky zozbierane hrube data + z nich vygenerovane kontingencne tabulky a grafy
-> subor holas.juraj.proj2.pdf je pisomny report ku projektu, ktory bol odovzdany aj samostatne

Spustanie programu
-> spusta sa hlavna trieda Main, bez argumentov
-> v aktualnom stave sa spusta testovanie najlepsieho modelu, pre ine testy spomenute v projekte staci len spustit prislusnu funkciu
-> pre vlastne testovanie sa riadte podla sablony:
       MLPParameters params = new MLPParameters();
       params.[nazov parametra] = [hodnota parametra];
       ... [nastavenie ostatnych parametrov]
       CrossValidation cv = new CrossValidation(params);
       cv.run(trainingDataSet);
   -> v poli cv.results budu nasledne ulozene vysledky jednotlivych instancii
   -> cv.getBest() vrati vysledky najlepsieho modelu
   -> kazdy z vysledkov obsahuje polia:
      - result.mlp - prislusny MLP
      - results.eErrors[] - vyvoj estimacnej chyby pocas trenovania
      - results.vErrors[] - vyvoj validacnej chyby pocas trenovania
   -> klasifikacia jedneho testovacieho prikladu prebieba pomocou mlp.classify(dataSet, index)
   -> testovanie mnoziny prikladov prebieha pomocou mlp.test(dataSet), metoda vracia primernu chybu
=======
Projekt c.1 -  Jednoduchy perceptron
Neuronove siete, 2014/2015

-> zdrojove subory k programu su v priecinku src/. Program nevyuziva ziadne
   externe kniznice ci balicky. Spusta sa hlavna trieda Main, bez argumentov.
-> v subore trainLog.txt je zaznamenany vystup z demonstracneho testovania
   perceptronu
-> v subore grafy.xlsx su vsetky zozbierane hrube data + z nich vygenerovane
   grafy
-> subor holas.juraj.proj1.pdf je pisomny report ku projektu, ktory bol
   odovzdany aj samostatne
>>>>>>> origin/master
