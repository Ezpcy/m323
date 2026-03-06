@main def main(): Unit =
  // perceptronExample()
  adalineExample()

def perceptronExample() = {
  val (x, y) = IrisLoader.loadData(
    "/home/ezpz/Nextcloud/ObsidianVault/Schule/Module/323 - Funktional programmieren/assignments/project/ml-classifiers/learningData/iris.csv"
  )

  val (perceptron, errors) = Perceptron.fit(x, y, 0.1, 10, 1)

  for (iteration <- errors.indices) {
    println(s"Iteration ${iteration + 1}: Fehler = ${errors(iteration)}")
  }

// Korrektur (funktional sauberer):
  x.zip(y).foreach { case (features, label) =>
    val guess = Perceptron.predict(features, perceptron)
    println(if (guess == label) "Korrekt geraten" else "Falsch geraten")
  }
}

def adalineExample() = {
  val (x, yInit) = IrisLoader.loadData(
    "/home/ezpz/Nextcloud/ObsidianVault/Schule/Module/323 - Funktional programmieren/assignments/project/ml-classifiers/learningData/iris.csv"
  )

  val y = yInit.map(v => v.toDouble)

  val (adaline, errors) = AdalineGD.fit(x, y, 0.0001, 15, 1)

  for (iteration <- errors.indices) {
    println(s"Iteration ${iteration + 1}: Fehler = ${errors(iteration)}")
  }
}
