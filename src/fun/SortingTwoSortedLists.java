/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 *
 * @author RAVI
 */
public class SortingTwoSortedLists {
    
    public static List<Integer> intersectionSort(List<Integer> l1, List<Integer> l2){
        
        Set<Integer> intersection = new HashSet<>();
        
        l1.forEach(i -> intersection.add(i));
        
        l2.forEach(j -> intersection.add(j));
        
        System.out.println("Intersection is done...");
        System.out.println(intersection);
        
        //converting set of intersection to list of unique sorted int
        List<Integer> backToList = intersection.stream()
                                    .collect(Collectors.toList());
        
        System.out.println("Result so far....");
        System.out.println(backToList);
        
        //perform sortng on this
        Collections.sort(backToList);
        
        
        return backToList;
    }
    
    public static void main(String[] args) {
        
        long start, end;
        
        //random list of int with un-sorted data
        List<Integer> l1 = new ArrayList<>(
                Arrays.asList(2, 3, 10, 0, 34, 3, 12, 1, 5)
        );
        
        //random list of int with un-sorted data
        List<Integer> l2 = new ArrayList<>(
                Arrays.asList(5, 16, 19, 4, 30, 7, 22, 100, 5)
        );
        
        //sorting them explicitly
        Collections.sort(l1);
        Collections.sort(l2);
        
        System.out.println("2 Sorted List as Input...");
        
        System.out.println("List l1: "+l1);
        System.out.println("List l2: "+l2);
        
        start = System.currentTimeMillis();
        
        List<Integer> result = intersectionSort(l1, l2);
        
        System.out.println("Final output...");
        System.out.println("output: "+result);
        
        end = System.currentTimeMillis();
        
        System.out.println("time: "+((end-start)/1000l)+"sec");
        
    }
    
}
