package org.assignments.A;

import java.util.List;
import java.util.ArrayList;

public class BShoppingCart {
  private List<String> cartItems = new ArrayList<>();
  private boolean bookAdded = false;

  public void addItem(String item) {
    cartItems.add(item);

    if (item.equalsIgnoreCase("book")) {
      bookAdded = true;
    }
  }

  public void removeItem(String item) {
    cartItems.remove(item);

    // Zustand neu berechnen!
    if (item.equalsIgnoreCase("book")) {
      bookAdded = cartItems.stream()
          .anyMatch(i -> i.equalsIgnoreCase("book"));
    }
  }

  public List<String> getItems() {
    return new ArrayList<>(cartItems);
  }

  public int getDiscountPercentage() {
    return bookAdded ? 5 : 0;
  }

}
