package org.assignments.A;

public class A {

  // Imperativ
  public static int calculateScore(String word) {
    int score = 0;

    for (char c : word.toCharArray()) {
      if (c == 'a') {
        continue;
      }

      score++;
    }
    return score;
  }

  // Declarative
  public static int wordScore(String word) {
    word = word.replaceAll("[a]", "");
    return word.length();
  }

  public A() {
    System.out.println(A.calculateScore("imperative"));
    System.out.println(A.calculateScore("no"));
    System.out.println(A.wordScore("imperative"));
    System.out.println(A.wordScore("no"));
  }
}
