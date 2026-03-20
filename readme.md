# 323 - Funktional Programmieren

## Projekt - Machine learning Algorithmen

[Projekt Ordner](./assignments/project/ml-classifiers/)

Für dieses Projekt werden binäre Machine Learning Klassifikatoren funktionell in **Scala** implementiert. Als Datensatz wird der [Iris Datensatz](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) verwendet.

Verwendung:

```bash
sbt "run <perceptron | adaline | adalinesgd>"
```

---

## 1. Perceptron

[Perceptron](assignments/project/ml-classifiers/src/main/scala/classifiers/Perceptron.scala)

Der Perceptron-Algorithmus ist ein einfacher Lernalgorithmus, der auf der Funktionsweise biologischer Neuronen basiert. Eine **Net Input**-Funktion berechnet eine Linearkombination aus den Input-Werten $\mathbf{x}$ und den dazugehörigen Gewichtungen $\mathbf{w}$:

$$\mathbf{w} = \begin{bmatrix} w_1 \\ \vdots \\ w_m\end{bmatrix} , \mathbf{x} = \begin{bmatrix} x_1 \\ \vdots \\ x_m\end{bmatrix}$$

Es wird eine Entscheidungsfunktion $\sigma(z)$ definiert, wobei $z$ der **Net Input** ist ($z = w_1x_1 + w_2x_2 + ... + w_mx_m$). Ist der Netto-Input eines bestimmten Beispiels $\mathbf{x}^{(i)}$ grösser oder gleich einem vordefinierten Schwellenwert $\theta$, wird die Klasse 1 vorhergesagt, andernfalls Klasse 0.

$$\sigma(z) = \begin{cases} 1 & \text{if } z \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

Wir definieren einen Bias-Wert $b$ und integrieren diesen in den **Net Input** $z$:

$$z = w_1x_1 + w_2x_2 + ... + w_mx_m + b = \mathbf{w}^T\mathbf{x} + b$$

### 1.1 Lernprozess

Für jedes Trainingsbeispiel $\mathbf{x}^{(i)}$ berechnen wir den vorhergesagten Output-Wert $\hat{y}$. Anschliessend werden der Gewichtungsvektor und der Bias-Wert entsprechend angepasst:

$$\begin{align*} w_j &:= w_j + \Delta w_j \\ b &:= b + \Delta b \end{align*}$$

Die Update-Werte ($\Delta$) berechnen sich wie folgt:

$$\begin{align*} \Delta w_j &= \eta(y^{(i)} - \hat{y}^{(i)})x_j^{(i)} \\ \Delta b &= \eta(y^{(i)} - \hat{y}^{(i)}) \end{align*}$$

Hierbei ist $\eta$ die Lernrate (Learning Rate), welche die Schrittgrösse des Updates bestimmt. Da das Perceptron eine einfache Sprungfunktion nutzt, werden die Gewichte nur angepasst, wenn die Vorhersage falsch ist.

### 1.2 Ergebnisse


![](./x_ressources/2026-03-13/image/readme-1773396895544.png)

_Abbildung 1.1: Falsch Vorhersagen je Iteration im Perceptron Algorithmus_


![](./x_ressources/2026-03-13/image/readme-1773396909172.png)

_Abbildung 1.2: Resultierende Entscheidungs Region durch den Perceptron Algorithmus_


---

## 2. Adaline

[AdalineGD](assignments/project/ml-classifiers/src/main/scala/classifiers/AdalineGD.scala)

Das **Adaline (Adaptive Linear Neuron)** Modell basiert auf dem [Perceptron](https://www.google.com/search?q=%231-perceptron)-Algorithmus, führt jedoch ein entscheidendes neues Konzept ein: die Definition und kontinuierliche Minimierung einer **Verlustfunktion (Loss Function)**.

Im Gegensatz zum Perceptron, das die Klassenschätzungen (0 oder 1) für den Fehlerabgleich nutzt, vergleicht Adaline die wahren Klassenlabels mit dem _kontinuierlichen_ linearen Output der Aktivierungsfunktion $\phi(z) = z$.

### 2.1 Lernprozess (Batch Gradient Descent)

Um die Gewichte zu optimieren, definieren wir eine Verlustfunktion $L$, oft den mittleren quadratischen Fehler (Mean Squared Error, MSE), die minimiert werden soll:

$$L = \frac{1}{2n} \sum_{i=1}^{n} \left( y^{(i)} - \phi(z^{(i)}) \right)^2$$

Die Gewichts-Updates erfolgen mittels **Gradient Descent** (Gradientenabstieg). Dabei werden die Gewichte basierend auf dem Gradienten der gesamten Verlustfunktion über das _komplette_ Trainingsset (Batch) aktualisiert:

$$\Delta w_j = \eta \sum_{i=1}^{n} \left( y^{(i)} - \phi(z^{(i)}) \right) x_j^{(i)}$$

Da alle Datenpunkte gleichzeitig in das Update einfliessen, ist dieser Algorithmus rechenintensiver pro Iteration, konvergiert aber stetig in Richtung des globalen Minimums (vorausgesetzt die Lernrate $\eta$ ist passend gewählt). Die finale Klassifizierung (0 oder 1) findet erst nach dem Training durch eine Schwellenwertfunktion statt.

### 2.2 Ergebnisse


![](./x_ressources/2026-03-13/image/readme-1773396865166.png)

_Abbildung 2.1: Kontinuierliche Minimierung der Fehler im Adaline Algorithmus_

![](./x_ressources/2026-03-13/image/readme-1773396854975.png)

_Abbildung 2.2: Resultierende Entscheidungs Region durch den Adaline Algorithmus_

---

## 3. AdalineSGD

[AdalineSGD](assignments/project/ml-classifiers/src/main/scala/classifiers/AdalineSGD.scala)

Beim **Stochastic Gradient Descent (SGD)** handelt es sich um eine Optimierung des klassischen Adaline-Algorithmus, die besonders bei grossen Datensätzen signifikante Performance-Vorteile bietet.

### 3.1 Lernprozess (Stochastischer Ansatz)

Anstatt die Gewichte wie beim AdalineGD erst am Ende einer Epoche für alle Datenpunkte auf einmal anzupassen (Batch-Update), werden die Gewichte hier **inkrementell nach jedem einzelnen Datenpunkt** (also _online_) aktualisiert.

Das Update für ein einzelnes Trainingsbeispiel $\mathbf{x}^{(i)}$ sieht folgendermassen aus:

$$\Delta w_j = \eta \left( y^{(i)} - \phi(z^{(i)}) \right) x_j^{(i)}$$

**Besonderheiten des SGD:**

1. **Shuffling:** Die Trainingsdaten müssen vor jeder Epoche gemischt (`shuffle`) werden, um zu verhindern, dass der Algorithmus Zyklen bildet oder sich an eine bestimmte Reihenfolge der Daten anpasst.
2. **Konvergenz:** Der SGD erreicht das Minimum der Verlustfunktion nicht auf einem glatten Weg, sondern schwankt (Rauschen/Noise). Dies hilft ihm jedoch, lokalen Minima in komplexeren Modellen zu entkommen, und führt oft viel schneller zu einer guten Lösung als der Batch-Ansatz.

### 3.2 Ergebnisse

![](./x_ressources/2026-03-13/image/readme-1773396830035.png)

_Abbildung 3.1: Fehler Verlauf im AdalineSGD Algorithmus_

![](./x_ressources/2026-03-13/image/readme-1773396814730.png)

_Abbildung 3.2: Resultierende Entscheidungs Region durch den AdalineSGD Algorithmus_

---

> **Quellen**:
> Sebastian Rascka und Vahid Mirjalili, Machine Learning with PyTorch and Scikit-Learn, 2019, Packt Publishing
> 
> **Ressourcen:**
> [Miro Board](https://miro.com/app/board/uXjVGCMHdFU=/)
> [Module Übersicht](https://docs.google.com/document/d/1nLG-KSTFBL7-Wcgilpr6mH3uZvVkrVVNwSujTTcfqzQ/edit?tab=t.0)
>
> **Text Anpassung:**
> [Gemini](https://gemini.google.com/)