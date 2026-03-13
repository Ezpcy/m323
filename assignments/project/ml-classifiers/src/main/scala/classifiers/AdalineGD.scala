import scala.util.Random
import scala.annotation.tailrec
case class AdalineGDModel(weights: Vector[Double], bias: Double)

object AdalineGD {

  def netInput(x: Vector[Double], model: AdalineGDModel): Double = {
    x.zip(model.weights).map { case (xi, wi) => xi * wi }.sum + model.bias
  }

  def predict(x: Vector[Double], model: AdalineGDModel): Int = {
    if (netInput(x, model) >= 0.5) 1 else 0
  }

  def fit(
      X: Vector[Vector[Double]],
      y: Vector[Double],
      eta: Double,
      nIter: Int,
      randomState: Int
  ): (AdalineGDModel, Vector[Double]) = {
    val random = new Random(randomState)
    val nFeatures = X.headOption.map(_.length).getOrElse(0)
    val nSamples = X.length

    val initialWeights = Vector.fill(nFeatures)(random.nextGaussian() * 0.01)
    val initialModel = AdalineGDModel(initialWeights, 0.0)

    @tailrec
    def trainEpoch(
        currentEpoch: Int,
        currentModel: AdalineGDModel,
        errorList: Vector[Double]
    ): (AdalineGDModel, Vector[Double]) = {
      if (currentEpoch >= nIter) {
        (currentModel, errorList)
      } else {
        val output = X.map(xi => netInput(xi, currentModel))
        val errors = y.zip(output).map { case (yi, outi) => yi - outi }

        val newWeights = currentModel.weights.zipWithIndex.map { case (wj, j) =>
          val gradientJ =
            X.zip(errors).map { case (xi, err) => err * xi(j) }.sum
          wj + (eta * 2.0 * gradientJ / nSamples)
        }

        val newBias = currentModel.bias + (eta * 2.0 * errors.sum / nSamples)

        val epochLoss = errors.map(e => e * e).sum / nSamples

        val newModel = AdalineGDModel(newWeights, newBias)

        trainEpoch(
          currentEpoch + 1,
          newModel,
          errorList :+ epochLoss
        )
      }
    }
    trainEpoch(0, initialModel, Vector.empty[Double])
  }
}
