object FeatureScaling {
  def standardize(column: Vector[Double]): Vector[Double] = {
    val n = column.length

    if (n == 0) return column

    val mean = column.sum / n

    val stdDev = column
  }
}
