import scala.util.Random
import scala.annotation.init
import scala.annotation.tailrec
// Case class is a immutable. When weights are being updated, a new Perceptron class is created.
case class PerceptronModel(weights: Vector[Double], bias: Double);

object Perceptron {

  // Pure Funktions Implementationen

  def netInput(x: Vector[Double], model: PerceptronModel): Double = {
    x.zip(model.weights).map { case (xi, wi) => xi * wi }.sum + model.bias
  }

  def predict(x: Vector[Double], model: PerceptronModel): Int = {
    if (netInput(x, model) >= 0.0) 1 else 0
  }

  def fit(
      X: Vector[Vector[Double]],
      y: Vector[Int],
      eta: Double,
      nIter: Int,
      randomState: Int
  ): (PerceptronModel, Vector[Int]) = {
    val random = new Random(randomState);
    val initialWeights = Vector.fill(X.headOption.map(_.length).getOrElse(0))(
      random.nextGaussian() * 0.01
    )
    val initialModel = PerceptronModel(initialWeights, 0.0)

    @tailrec
    def trainEpoch(
        currentEpoch: Int,
        currentModel: PerceptronModel,
        errorList: Vector[Int]
    ): (PerceptronModel, Vector[Int]) = {
      if (currentEpoch >= nIter) {
        (currentModel, errorList) // end condition reached
      } else {

        // Anstatt Variablen zu überschreiben, wird foldLeft genutzt.
        // foldLeft reicht einen "Akkumulator" (aktuelle Modell und die Fehler)
        // von Datenpunkt zu Datenpunkt weiter.

        val initialAccumulator = (currentModel, 0)

        val (modelAfterEpoch, errorsinEpoch) =
          X.zip(y).foldLeft(initialAccumulator) {
            case ((modelAcc, errorAcc), (xi, target)) =>
              val prediction = predict(xi, modelAcc)
              val update = eta * (target - prediction)

              // Bei einem Fehler erzeugen wir ein neues Modell
              if (update != 0.0) {
                val newWeights = modelAcc.weights.zip(xi).map {
                  case (w, x_val) => w + update * x_val
                }

                val newBias = modelAcc.bias + update
                val newModel = PerceptronModel(newWeights, newBias)

                (
                  newModel,
                  errorAcc + 1
                ) // Neues Modell wird weitergegeben und Fehler erhöht
              } else {
                (modelAcc, errorAcc) // Mit dem altem Modell fortfahren
              }
          }

        // Rekursiver Aufruf für die nächste Epoche
        trainEpoch(
          currentEpoch + 1,
          modelAfterEpoch,
          errorList :+ errorsinEpoch
        )
      }
    }

    trainEpoch(0, initialModel, Vector.empty[Int])
  }
}
