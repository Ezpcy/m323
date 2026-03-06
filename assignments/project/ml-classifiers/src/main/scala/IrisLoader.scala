import scala.io.Source
import scala.util.Using

object IrisLoader {
  def loadData(filePath: String): (Vector[Vector[Double]], Vector[Int]) = {

    val lines: Vector[String] = Using(Source.fromFile(filePath)) { source =>
      source.getLines().toVector
    }.getOrElse(Vector.empty)

    val dataLines = lines.drop(1).filter(_.trim.nonEmpty)

    val parsedData: Vector[(Vector[Double], Int)] = dataLines.map { line =>
      val columns = line.split(',')

      val features = columns.take(4).map(_.toDouble).toVector

      val labelString = columns.last.trim

      val labelInt = if (labelString == "Iris-setosa") 0 else 1

      (features, labelInt)
    }

    // unzip teilt die Tupeln in zwei seperate Vektoren auf
    val (x, y) = parsedData.unzip

    (x, y)
  }
}
