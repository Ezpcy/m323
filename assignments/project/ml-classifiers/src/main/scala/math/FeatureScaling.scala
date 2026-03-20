object FeatureScaling {
  def standardize(column: Vector[Double]): Vector[Double] = {
    val n = column.length

    if (n == 0) return column

    val mean = column.sum / n

    val variance = column.map(x => math.pow(x - mean, 2)).sum / n
    val stdDev = math.sqrt(variance)

    if (stdDev == 0.0) {
      return Vector.fill(n)(0.0)
    } else {
      column.map(x => (x - mean) / stdDev)
    }
  }
}
