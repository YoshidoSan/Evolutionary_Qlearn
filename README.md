Temat Projektu :
Zastosowanie uczenia ze wzmocnieniem do ustawienia prawdopodobieństwa krzyżowania i sposobu krzyżowania w algorytmie ewolucyjnym. <br>
Niech stanem będzie procent sukcesów (ile potomków było lepszych niż rodzice) oraz średnia odległość pomiędzy osobnikami w
aktualnej populacji, a akcją wybór wskazanych parametrów algorytmu. Zarówno stany jak i
akcje można zdyskretyzować. Funkcje do optymalizacji należy pobrać z benchmarku
CEC2017, którego kod da się znaleźć w Pythonie, R i C.


\begin{itemize}
    \item model.py - zawiera implementację algorytmu ewolucyjnego
    \item ewo\_calculating\_params.py - skrypt służący do wywołania badań nad parametrami algorytmu ewolucyjnego
    \item qlearn.py - zawiera implementację q-learning połączonego z algorytmem ewolucyjnym
    \item train.py - skrypt służący do badań porównawczych działania algorytmu ewolucyjnego kierowanego przez q-learning i do wywołania badań nad parametrami qlearning
\end{itemize}

Omówienie w dokumentacji końcowej
