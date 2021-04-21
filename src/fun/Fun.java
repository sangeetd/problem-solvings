/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 *
 * @author RAVI
 */
public class Fun {

    public static long factorial(long n) {

        if (n == 0l || n == 1l) {
            return 1l;
        } else {
            return n * factorial(n - 1);
        }

    }

    public static long permutationCalculation(List<Character> charList) {

        long numerator = charList.size();
        Map<Character, Long> charOccr = new HashMap<>();
        for (char c : charList) {

            if (charOccr.containsKey(c)) {
                long counter = charOccr.get(c);
                charOccr.put(c, counter + 1l);
            } else {
                charOccr.put(c, 1l);
            }

        }

        //find permutation using factorial;
        long numeratorFactorial = factorial(numerator);
        long denominatorFactorial = 1l;
        Iterator keySet = charOccr.keySet().iterator();
        while (keySet.hasNext()) {
            denominatorFactorial *= factorial(charOccr.get(keySet.next()));
        }

        return (numeratorFactorial / denominatorFactorial);
    }

    public static List<String> stringPermutation(String input) {

        List<Character> charList = new ArrayList<>();
        for (char c : input.toCharArray()) {
            charList.add(c);
        }

        Set<String> setComb = new HashSet<>();

        //permutation combination for input
        long permutationPossibility = permutationCalculation(charList);
        while (setComb.size() != permutationPossibility) {

            try {

                Collections.shuffle(charList);
                StringBuilder sb = new StringBuilder();
                for (char c : charList) {
                    sb.append(c);
                }

                setComb.add(sb.toString());

            } catch (Exception e) {
                e.printStackTrace();
            }

        }

        return setComb.stream()
                .collect(Collectors.toList());
    }

    public static void main(String[] args) {
        // TODO code application logic here
        
        List<String> getCombination = stringPermutation("MISSISSIPPI");
//        System.out.println(getCombination.size());
//        for(String s: getCombination){
//            System.out.println(s);
//        }
        
        BinaryHeap<Integer> minHeap = new BinaryHeap<>((a, b) -> a.compareTo(b));
        minHeap.add(2);
        minHeap.add(10);
        minHeap.add(1);
        minHeap.print();
        minHeap.add(13);
        minHeap.add(0);
        minHeap.add(7);
        minHeap.add(100);
        minHeap.print();
    }

}
