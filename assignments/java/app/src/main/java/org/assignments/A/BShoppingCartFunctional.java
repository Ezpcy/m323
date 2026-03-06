package org.assignments.A;

import java.util.List;

public class BShoppingCartFunctional {

  public static int getDiscountPercentage(List<String> items) {
    boolean containsBook = items.stream()
        .anyMatch(item -> item.equalsIgnoreCase("book"));

    return containsBook ? 5 : 0;
  }

}
