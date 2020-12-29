/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *
 * @author sangeetdas
 */
public class DSA450Questions {

    public void reverseArray(int[] a) {

        int len = a.length;

        //.....................O(N)
        for (int i = 0; i < len / 2; i++) {
            int temp = a[i];
            a[i] = a[len - i - 1];
            a[len - i - 1] = temp;
        }

        //output
        for (int x : a) {
            System.out.print(x + " ");
        }
        System.out.println();

    }

    public String reverseString(String str) {

        int len = str.length();
        char[] ch = str.toCharArray();

        //.....................reverse by length
        //.....................O(N)
        for (int i = 0; i < len / 2; i++) {
            char temp = ch[i];
            ch[i] = ch[len - i - 1];
            ch[len - i - 1] = temp;
        }

        //output
        System.out.println("output reverse by length: " + String.valueOf(ch));

        //....................reverse by two pointer
        //....................O(N)
        int f = 0;
        int l = len - 1;
        ch = str.toCharArray();

        while (f < l) {

            char temp = ch[f];
            ch[f] = ch[l];
            ch[l] = temp;
            f++;
            l--;

        }

        //output
        System.out.println("output reverse by two pointer: " + String.valueOf(ch));

        //............................reverse by STL
        String output = new StringBuilder(str)
                .reverse()
                .toString();
        System.out.println("output reverse by STL: " + output);

        return output;

    }

    public boolean isStringPallindrome(String str) {

        return str.equals(reverseString(str));

    }
    
    public void printDuplicatesCharInString(String str){
        System.out.println("For: "+str);
        
        Map<Character, Integer> countMap = new HashMap<>();
        for(char c: str.toCharArray()){
            countMap.put(c, countMap.getOrDefault(c, 0) + 1);
        }
        
        countMap.entrySet().stream()
                .filter(e -> e.getValue() > 1)
                .forEach(e -> System.out.println(e.getKey()+" "+e.getValue()));
        
    }
    
    public void reverseLinkedList_Iterative(Node<Integer> node){
        System.out.println("Reverse linked list iterative");
        //actual
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(node);
        ll.print();
        
        Node<Integer> curr = node;
        Node<Integer> prev = null;
        Node<Integer> next = null;
        
        while(curr != null){
            
            next = curr.getNext();
            curr.setNext(prev);
            prev = curr;
            curr = next;
            
        }
        
        node = prev;
        
        //output
        LinkedListUtil<Integer> output = new LinkedListUtil<>(node);
        output.print();
        
    }
    
    Node<Integer> reverseLinkedList_Recursive_NewHead;
    private Node<Integer> reverseLinkedList_Recursive_Helper(Node<Integer> node){
        
        if(node.getNext() == null){
            reverseLinkedList_Recursive_NewHead = node;
            return node;
        }
        
        Node<Integer> revNode = reverseLinkedList_Recursive_Helper(node.getNext());
        revNode.setNext(node);
        node.setNext(null);
        
        return node;
        
    }
    
    public void reverseLinkedList_Recursive(Node<Integer> node){
        System.out.println("Reverse linked list recursive");
        //actual
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(node);
        ll.print();
        
        reverseLinkedList_Recursive_Helper(node);
        
        //output
        LinkedListUtil<Integer> output = new LinkedListUtil<>(reverseLinkedList_Recursive_NewHead);
        output.print();
        
    }

    public static void main(String[] args) {

        //Object to access method
        DSA450Questions obj = new DSA450Questions();

        //......................................................................
//        Row: 6
//        System.out.println("Reverse array");
//        int[] a1 = {1, 2, 3, 4, 5};
//        obj.reverseArray(a1);
//        int[] a2 = {1, 2, 3, 4};
//        obj.reverseArray(a2);
        //......................................................................
//        Row: 56
//        System.out.println("Reverse string");
//        String str1 = "Sangeet";
//        obj.reverseString(str1);
//        String str2 = "ABCD";
//        obj.reverseString(str2);
        //......................................................................
//        Row: 57 
//        System.out.println("Is string pallindrome");
//        String str3 = "Sangeet";
//        System.out.println(str3+" "+obj.isStringPallindrome(str3));
//        String str4 = "ABBA";
//        System.out.println(str4+" "+obj.isStringPallindrome(str4));
        //......................................................................
//        Row: 58
//        System.out.println("Print duplicates char in string");
//        String str5 = "AABBCDD";
//        obj.printDuplicatesCharInString(str5);
//        String str6 = "XYZPQRS";
//        obj.printDuplicatesCharInString(str6);
        //......................................................................
//        Row: 139
        System.out.println("Reverse a linked list iterative/recursive");
        Node<Integer> node1 = new Node<>(1);
        node1.setNext(new Node<>(2));
        node1.getNext().setNext(new Node<>(3));
        obj.reverseLinkedList_Iterative(node1);
        Node<Integer> node2 = new Node<>(1);
        node2.setNext(new Node<>(2));
        node2.getNext().setNext(new Node<>(3));
        node2.getNext().getNext().setNext(new Node<>(4));
        node2.getNext().getNext().getNext().setNext(new Node<>(5));
        obj.reverseLinkedList_Recursive(node2);
    }

}
