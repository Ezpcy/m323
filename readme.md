# 323 - Funktional Programmieren

## Projekt - Machine learning Algorithmen

[Projekt Ordner](./assignments/project/ml-classifiers/)

Für dieses Projekt werden binäre Machine Learning Klassifikatoren funktionell in **Scala** implementiert. Als Datensatz wird der [Iris Datensatz](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) verwendet.

---

## Perceptron

[Perceptron](assignments/project/ml-classifiers/src/main/scala/algos/Perceptron.scala)

Der Perceptron-Algorithmus ist ein einfacher Lernalgorithmus, der auf der Funktionsweise biologischer Neuronen basiert. 

Es wird eine Entscheidungsfunktion definiert, σ(z), die eine lineare Kombination bestimmter Input-Werte, x, und eines Gewichtungsvektors, w, annimmt.

Ist der Netto-Input eines bestimmten Beispiels, $x^{(i)}$, grösser als ein vordefinierter Schwellwert, $\theta$, wird die Klasse 1 geschätzt, andernfalls Klasse 0.

---

## Adaline

[AdalineGD](assignments/project/ml-classifiers/src/main/scala/algos/AdalineGD.scala)

Der **Adaline (ADaptive LInead NEuron)** basiert auf dem [Perceptron](#Perceptron) Algorithmus und illustriert Konzepte der Definition und kontinuierte Minimierung einer Verlust Funktion. 


---

> **Quellen**:
> Sebastian Rascka und Vahid Mirjalili, Machine Learning with PyTorch and Scikit-Learn, 2019, Packt Publishing
> 
> **Ressourcen:**
> [Miro Board](https://miro.com/app/board/uXjVGCMHdFU=/)
> [Module Übersicht](https://docs.google.com/document/d/1nLG-KSTFBL7-Wcgilpr6mH3uZvVkrVVNwSujTTcfqzQ/edit?tab=t.0)