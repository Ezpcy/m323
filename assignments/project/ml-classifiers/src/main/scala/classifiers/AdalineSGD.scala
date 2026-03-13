import scala.util.Random
import scala.annotation.tailrec
case class AdalineSGDModel(weights: Vector[Double], bias: Double)

object AdalineSGD {
  def netInput(x: Vector[Double], model: AdalineSGDModel): Double = {
    x.zip(model.weights).map { case (xi, wi) => xi * wi }.sum + model.bias
  }

  def predict(x: Vector[Double], model: AdalineSGDModel): Int = {
    if (netInput(x, model) >= 0.5) 1 else 0
  }

  def updateWeights(
      xi: Vector[Double],
      target: Double,
      model: AdalineSGDModel,
      eta: Double
  ): (AdalineSGDModel, Double) = {
    val output = netInput(xi, model)
    val error = target - output

    val newWeights = model.weights.zip(xi).map { case (w, xVal) =>
      w + (eta * 2.0 * xVal * error)
    }

    val newBias = model.bias + (eta * 2.0 * error)
    val loss = error * error

    (AdalineSGDModel(newWeights, newBias), loss)
  }

  def fit(
      X: Vector[Vector[Double]],
      y: Vector[Double],
      eta: Double,
      nIter: Int,
      randomState: Int,
      shuffle: Boolean = true
  ): (AdalineSGDModel, Vector[Double]) = {
    val random = new Random(randomState)
    val nFeatures = X.headOption.map(_.length).getOrElse(0)

    val initialWeights = Vector.fill(nFeatures)(random.nextGaussian() * 0.01)
    val initialModel = AdalineSGDModel(initialWeights, 0.0)

    @tailrec
    def trainEpoch(
        currentEpoch: Int,
        currentModel: AdalineSGDModel,
        lossList: Vector[Double]
    ): (AdalineSGDModel, Vector[Double]) = {
      if (currentEpoch >= nIter) {
        (currentModel, lossList)
      } else {
        val dataset = X.zip(y)
        val epochData = if (shuffle) random.shuffle(dataset) else dataset

        val initalAccumulator = (currentModel, Vector.empty[Double])

        val (modelAfterEpoch, lossesInEpoch) =
          epochData.foldLeft(initalAccumulator) {
            case ((modelAcc, lossAcc), (xi, target)) =>
              val (updateModel, singleLoss) =
                updateWeights(xi, target, modelAcc, eta)
              (updateModel, lossAcc :+ singleLoss)
          }

        val avgLoss =
          if (lossesInEpoch.nonEmpty) lossesInEpoch.sum / lossesInEpoch.length
          else 0.0

        trainEpoch(currentEpoch + 1, modelAfterEpoch, lossList :+ avgLoss)
      }
    }

    trainEpoch(0, initialModel, Vector.empty[Double])
  }

}
