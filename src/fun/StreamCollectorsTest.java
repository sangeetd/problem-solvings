/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 *
 * @author RAVI
 */
public class StreamCollectorsTest {

    public static void main(String[] args) {

        //https://medium.com/swlh/java-collectors-and-its-20-methods-2fc422920f18
        
        List<Integer> intList = new ArrayList(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8));
        System.out.println(intList);

        System.out.println("--Int list filter for even int and toList");
        List<Integer> evenList = intList.stream()
                .filter(x -> x % 2 == 0)
                .collect(Collectors.toList());

        evenList.stream().forEach(System.out::println);

        System.out.println("--Int list filter for odd int and convert the result into a specified Collection class");
        List<Integer> oddList = intList.stream()
                .filter(x -> x % 2 == 1)
                //specified Collection class
                .collect(Collectors.toCollection(LinkedList::new));

        oddList.stream().forEach(System.out::println);

        //create duplicate of the list
        intList.addAll(intList);
        System.out.println(intList);

        System.out.println("--Int list to remove duplicates and toSet");
        Set<Integer> set = intList.stream()
                .collect(Collectors.toSet());
        set.stream().forEach(System.out::println);

        System.out.println("--Int list to remove duplicates and convert the result into a specified Collection class");
        //shuffle the main list to show below how treeset will sort the shuffled list
        Collections.shuffle(intList);
        System.out.println("Shuffled : " + intList);
        set = intList.stream()
                //tree set to show the shuffled List above is now sorted
                //specified Collection class
                .collect(Collectors.toCollection(TreeSet::new));
        set.stream().forEach(System.out::println);

        System.out.println("--Int list to count duplicates occurenecs and toMap");
        Map<Integer, Integer> map = intList.stream()
                .collect(Collectors.toMap(Function.identity(), 
                        x -> x*x, 
                        (y1, y2) -> y1));
        map.entrySet().stream().forEach(System.out::println);
        
        System.out.println("--Int list to count duplicates occurenecs and groupingBy");
        Map<Integer, Long> map2 = intList.stream()
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        map2.entrySet().stream().forEach(System.out::println);
        
        System.out.println("--set to divide list as map of even or odd and groupingBy | two variants");
        
        Map<Integer, List<Integer>> map3 = set.stream()
                .collect(Collectors.groupingBy(x -> x%2));
        map3.entrySet().stream().forEach(System.out::println);
        
        Map<String, List<Integer>> map4 = set.stream()
                .collect(Collectors.groupingBy(x -> x%2 == 0? "even" : "odd"));
        map4.entrySet().stream().forEach(System.out::println);

        System.out.println("--Simple count elements in collection");
        System.out.println("Count main list: " + intList.stream().collect(Collectors.counting()));
        System.out.println("Count set: " + set.stream().collect(Collectors.counting()));

        System.out.println("--Simple min elements in collection | natural order");
        System.out.println("Min in main list: " + intList.stream()
                .collect(Collectors
                        .minBy(Comparator.naturalOrder())
                )
                .get());

        List<String> stringList = new ArrayList<>(Arrays.asList("Alpha", "Beta", "Gamma", "Omega"));
        System.out.println("Min in String list: " + stringList.stream()
                .collect(Collectors
                        .minBy(Comparator.naturalOrder())
                )
                .get());

        //revrse of above
        System.out.println("--Simple min elements in collection | reverse order");
        System.out.println("Min in main list: " + intList.stream()
                .collect(Collectors
                        .minBy(Comparator.reverseOrder())
                )
                .get());

        System.out.println("Min in String list: " + stringList.stream()
                .collect(Collectors
                        .minBy(Comparator.reverseOrder())
                )
                .get());

        System.out.println("--Simple Max elements in collection | natural order");
        System.out.println("Max in main list: " + intList.stream()
                .collect(Collectors
                        .maxBy(Comparator.naturalOrder())
                )
                .get());

        System.out.println("Max in String list: " + stringList.stream()
                .collect(Collectors
                        .maxBy(Comparator.naturalOrder())
                )
                .get());

        System.out.println("--Simple Max elements in collection | reverse order");
        System.out.println("Max in main list: " + intList.stream()
                .collect(Collectors
                        .maxBy(Comparator.reverseOrder())
                )
                .get());

        System.out.println("Max in String list: " + stringList.stream()
                .collect(Collectors
                        .maxBy(Comparator.reverseOrder())
                )
                .get());

        System.out.println("--Partionting in collection");
        List<String> strings = Arrays.asList("a", "alpha", "beta", "gamma");
        Map<Boolean, List<String>> partionedMap = strings
                .stream()
                .collect(Collectors.partitioningBy(x -> x.length() > 2));
        partionedMap.entrySet().stream().forEach(System.out::println);

        System.out.println("--Joining elements in collection");
        System.out.println("Joining in String list: " + stringList.stream()
                .collect(Collectors.joining(/*delimiter*/"%")));
        System.out.println("Joining in String list: " + stringList.stream()
                .distinct()
                .collect(Collectors.joining(/*delimiter*/"%")));
        System.out.println("Joining in String list: " + stringList.stream()
                .collect(Collectors.joining(/*delimiter*/"%", /*prefix*/ "[-", /*suffix*/ "-]")));

        System.out.println("--Average in collection");
        List<Long> longValues = Arrays.asList(100l, 200l, 300l);
        System.out.println("Average Long: "+longValues
                .stream()
                .collect(Collectors.averagingLong(x -> x * 2)));
        
        List<Integer> intValues = Arrays.asList(1,2,3,4,5,6,6);
        System.out.println("Average Long: "+intValues
                .stream()
                .collect(Collectors.averagingInt(x -> x * 2)));
        
        List<Double> doubleValues = Arrays.asList(1.1,2.0,3.0,4.0,5.0,5.0);
        System.out.println("Average Long: "+doubleValues
                .stream()
                .collect(Collectors.averagingDouble(x -> x)));
        
        System.out.println("--Summing in collection");
        System.out.println("Summing string by length: "+stringList
                .stream()
                .collect(Collectors.summingInt(str -> str.length())));
        System.out.println("Summing even list: "+evenList
                .stream()
                .collect(Collectors.summingInt(x -> x)));
        System.out.println("Summing even list with given Funtion: "+evenList
                .stream()
                .collect(Collectors.summingInt(x -> x*10)));
        System.out.println("Summing double list: "+doubleValues
                .stream()
                .collect(Collectors.summingDouble(x -> x)));
        System.out.println("Summing long list: "+longValues
                .stream()
                .collect(Collectors.summingLong(x -> x)));
        
        System.out.println("--Summarizing in collection");
        System.out.println("Summarizing int list: "+intList
                .stream()
                .collect(Collectors.summarizingInt(x -> x)));

    }

}
