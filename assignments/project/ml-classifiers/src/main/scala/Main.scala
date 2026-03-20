@main def runAlgorithmus(alg: String): Unit = {
  // .toLowerCase stellt sicher, dass auch "Perceptron" oder "PERCEPTRON" funktioniert
  alg.toLowerCase match {
    case "perceptron" =>
      perceptronExample()

    case "adalinegd" =>
      adalineExample()

    case "adalinesgd" =>
      adalineSGDExample()

    case _ => // Der Wildcard-Operator fängt alles andere ab
      println(s"Fehler: Der Algorithmus '$alg' ist unbekannt.")
      println("Bitte wähle: perceptron, adalinegd oder adalinesgd")
  }
}

def perceptronExample() = {
  val (x, y) = IrisLoader.loadData(
    "/home/ezpz/Nextcloud/ObsidianVault/Schule/Module/323 - Funktional programmieren/assignments/project/ml-classifiers/learningData/iris.csv"
  )

  val (perceptron, errors) = Perceptron.fit(x, y, 0.1, 10, 1)

  for (iteration <- errors.indices) {
    println(s"Iteration ${iteration + 1}: Fehler = ${errors(iteration)}")
  }
}

def adalineExample() = {
  val (x, yInit) = IrisLoader.loadData(
    "/home/ezpz/Nextcloud/ObsidianVault/Schule/Module/323 - Funktional programmieren/assignments/project/ml-classifiers/learningData/iris.csv"
  )
  val xStd = x.map(xi => FeatureScaling.standardize(xi));

  val y = yInit.map(v => v.toDouble)

  val (adaline, errors) = AdalineGD.fit(xStd, y, 0.0001, 15, 1)

  for (iteration <- errors.indices) {
    println(s"Iteration ${iteration + 1}: Fehler = ${errors(iteration)}")
  }
}

def adalineSGDExample() = {
  val (x, yInit) = IrisLoader.loadData(
    "/home/ezpz/Nextcloud/ObsidianVault/Schule/Module/323 - Funktional programmieren/assignments/project/ml-classifiers/learningData/iris.csv"
  )

  val xStd = x.map(xi => FeatureScaling.standardize(xi));

  val y = yInit.map(v => v.toDouble)

  val (adaline, errors) = AdalineSGD.fit(xStd, y, 0.01, 15, 1);

  for (iteration <- errors.indices) {
    println(s"Iteration ${iteration + 1}: Fehler = ${errors(iteration)}")
  }
}
