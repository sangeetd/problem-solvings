/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Stack;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javafx.util.Pair;

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

    public void printDuplicatesCharInString(String str) {
        System.out.println("For: " + str);

        Map<Character, Integer> countMap = new HashMap<>();
        for (char c : str.toCharArray()) {
            countMap.put(c, countMap.getOrDefault(c, 0) + 1);
        }

        countMap.entrySet().stream()
                .filter(e -> e.getValue() > 1)
                .forEach(e -> System.out.println(e.getKey() + " " + e.getValue()));

    }

    public void reverseLinkedList_Iterative(Node<Integer> node) {
        System.out.println("Reverse linked list iterative");
        //actual
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(node);
        ll.print();

        Node<Integer> curr = node;
        Node<Integer> prev = null;
        Node<Integer> next = null;

        while (curr != null) {

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

    private Node<Integer> reverseLinkedList_Recursive_Helper(Node<Integer> node) {

        if (node.getNext() == null) {
            reverseLinkedList_Recursive_NewHead = node;
            return node;
        }

        Node<Integer> revNode = reverseLinkedList_Recursive_Helper(node.getNext());
        revNode.setNext(node);
        node.setNext(null);

        return node;

    }

    public void reverseLinkedList_Recursive(Node<Integer> node) {
        System.out.println("Reverse linked list recursive");
        //actual
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(node);
        ll.print();

        reverseLinkedList_Recursive_Helper(node);

        //output
        LinkedListUtil<Integer> output = new LinkedListUtil<>(reverseLinkedList_Recursive_NewHead);
        output.print();

    }

    public void levelOrderTraversal_Iterative(TreeNode root) {

        if (root == null) {
            return;
        }

        //actuals
        BinaryTree bt = new BinaryTree(root);
        bt.treeBFS();

        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        Queue<TreeNode> intQ = new LinkedList<>();

        List<List> levels = new ArrayList<>();
        List nodes = new ArrayList<>();

        while (!q.isEmpty()) {

            TreeNode t = q.poll();
            nodes.add(t.getData());

            if (t.getLeft() != null) {
                intQ.add(t.getLeft());
            }
            if (t.getRight() != null) {
                intQ.add(t.getRight());
            }

            if (q.isEmpty()) {
                levels.add(nodes);
                nodes = new ArrayList<>();
                q.addAll(intQ);
                intQ.clear();
            }

        }

        //output
        System.out.println();
        for (List l : levels) {
            System.out.println(l);
        }

    }
    
    public void inOrderTraversal_Iterative(TreeNode root){
        
        if(root == null){
            return;
        }
        
        Stack<Pair<TreeNode, Integer>> stack = new Stack<>();
        stack.push(new Pair<>(root, 0));
        
        while(!stack.isEmpty()){
            
            Pair<TreeNode, Integer> p = stack.pop(); 
            TreeNode n = p.getKey();
            int status = p.getValue();
            
            if(n == null || status == 3){
                continue;
            }
            
            stack.push(new Pair<>(n, status + 1));
            
            if(status == 0){
                stack.push(new Pair<>(n.getLeft(), 0));
            }
            
            if(status == 1){
                System.out.print(n.getData() + " ");
            }
            
            if(status == 2){
                stack.push(new Pair<>(n.getRight(), 0));
            }
            
        }
        
        System.out.println();
        
    }
    
    public void inOrderTraversal_Recursive(TreeNode root){
        
        if(root == null){
            return;
        }
        
        inOrderTraversal_Recursive(root.getLeft());
        System.out.print(root.getData() + " ");
        inOrderTraversal_Recursive(root.getRight());
    }
    
    public void preOrderTraversal_Iterative(TreeNode root){
        
        if(root == null){
            return;
        }
        
        Stack<Pair<TreeNode, Integer>> stack = new Stack<>();
        stack.push(new Pair<>(root, 0));
        
        while(!stack.isEmpty()){
            
            Pair<TreeNode, Integer> p = stack.pop();
            TreeNode n = p.getKey();
            int status = p.getValue();
            
            if(n == null || status == 3){
                continue;
            }
            
            stack.push(new Pair<>(n, status + 1));
            
            if(status == 0){
                System.out.print(n.getData() + " ");
            }
            
            if(status == 1){
                stack.push(new Pair<>(n.getLeft(), 0));
            }
            
            if(status == 2){
                stack.push(new Pair<>(n.getRight(), 0));
            }
            
        }
        
        System.out.println();
        
    }
    
    public void preOrderTraversal_Recursive(TreeNode root){
        
        if(root == null){
            return;
        }
        
        System.out.print(root.getData() + " ");
        preOrderTraversal_Recursive(root.getLeft());
        preOrderTraversal_Recursive(root.getRight());
        
    }
    
    public void postOrderTraversal_Iterative(TreeNode root){
        
        if(root == null){
            return;
        }
        
        Stack<Pair<TreeNode, Integer>> stack = new Stack<>();
        stack.push(new Pair<>(root, 0));
        
        while(!stack.isEmpty()){
            
            Pair<TreeNode, Integer> p = stack.pop();
            TreeNode n = p.getKey();
            int status = p.getValue();
            
            if(n == null || status == 3){
                continue;
            }
            
            stack.push(new Pair<>(n, status + 1));
            
            if(status == 0){
                stack.push(new Pair<>(n.getLeft(), 0));
            }
            
            if(status == 1){
                stack.push(new Pair<>(n.getRight(), 0));
            }
            
            if(status == 2){
                System.out.print(n.getData() + " ");
            }
            
        }
        
        System.out.println();
    }
    
    public void postOrderTraversal_recursive(TreeNode root){
        
        if(root == null){
            return;
        }
        
        postOrderTraversal_recursive(root.getLeft());
        postOrderTraversal_recursive(root.getRight());
        System.out.print(root.getData() + " ");
        
    }

    public int heightOfTree(TreeNode root) {

        if (root == null) {
            return -1;
        }

        return Math.max(heightOfTree(root.getLeft()),
                heightOfTree(root.getRight())) + 1;

    }

    public TreeNode mirrorOfTree(TreeNode root) {

        if (root == null) {
            return null;
        }

        TreeNode left = mirrorOfTree(root.getLeft());
        TreeNode right = mirrorOfTree(root.getRight());
        root.setLeft(right);
        root.setRight(left);

        return root;

    }
    
    int middleElementInStack_Element = Integer.MIN_VALUE;
    private void middleElementInStack_Helper(Stack<Integer> s, int n, int index) {

        if (n == index || s.isEmpty()) {
            return;
        }

        int ele = s.pop();
        middleElementInStack_Helper(s, n, index + 1);
        if (index == n / 2) {
            middleElementInStack_Element = ele;
        }
        s.push(ele);
    }

    public void middleElementInStack(Stack<Integer> stack) {
        int n = stack.size();
        int index = 0;
        middleElementInStack_Helper(stack, n, index);
        //outputs
        System.out.println("Middle eleement of the stack: "+middleElementInStack_Element);
        //just reseting
        middleElementInStack_Element = Integer.MIN_VALUE;
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
//        System.out.println("Reverse a linked list iterative/recursive");
//        Node<Integer> node1 = new Node<>(1);
//        node1.setNext(new Node<>(2));
//        node1.getNext().setNext(new Node<>(3));
//        obj.reverseLinkedList_Iterative(node1);
//        Node<Integer> node2 = new Node<>(1);
//        node2.setNext(new Node<>(2));
//        node2.getNext().setNext(new Node<>(3));
//        node2.getNext().getNext().setNext(new Node<>(4));
//        node2.getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.reverseLinkedList_Recursive(node2);
        //......................................................................
//        Row: 177
//        System.out.println("Level order traversal of tree iterative");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.levelOrderTraversal_Iterative(root1);
        //......................................................................
//        Row: 179
//        System.out.println("Height of tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        System.out.println(obj.heightOfTree(root1));
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode(2));
//        System.out.println(obj.heightOfTree(root2));
        //......................................................................
//        Row: 181
//        System.out.println("Mirror of tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        //actual
//        BinaryTree bt = new BinaryTree<>(root1);
//        bt.treeBFS();
//        obj.mirrorOfTree(root1);
//        System.out.println();
//        //output
//        bt = new BinaryTree<>(root1);
//        bt.treeBFS();
        //......................................................................
//        Row: 299
//        System.out.println("Middle element in the stack");
//        Stack<Integer> stack = new Stack<>();
//        stack.addAll(Arrays.asList(1, 2, 3, 4, 5, 6, 7));
//        obj.middleElementInStack(stack);
//        stack.clear();
//        stack.addAll(Arrays.asList(1, 2, 3, 4));
//        obj.middleElementInStack(stack);
//        stack.clear();
//        //empty stack!!
//        obj.middleElementInStack(stack);
        //......................................................................
//        Row: 182
//        System.out.println("Inorder traversal of tree Iterative/recursive");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        //actual
//        BinaryTree bt = new BinaryTree<>(root1);
//        bt.treeBFS();
//        System.out.println();
//        obj.inOrderTraversal_Iterative(root1);
//        obj.inOrderTraversal_Recursive(root1);
        //......................................................................
//        Row: 183
//        System.out.println("Preorder traversal of tree Iterative/recursive");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        //actual
//        BinaryTree bt = new BinaryTree<>(root1);
//        bt.treeBFS();
//        System.out.println();
//        obj.preOrderTraversal_Iterative(root1);
//        obj.preOrderTraversal_Recursive(root1);
        //......................................................................
//        Row: 184
        System.out.println("Postsorder traversal of tree Iterative/recursive");
        TreeNode<Integer> root1 = new TreeNode<>(6);
        root1.setLeft(new TreeNode(2));
        root1.getLeft().setLeft(new TreeNode(0));
        root1.getLeft().setRight(new TreeNode(4));
        root1.getLeft().getRight().setLeft(new TreeNode(3));
        root1.getLeft().getRight().setRight(new TreeNode(5));
        root1.setRight(new TreeNode(8));
        root1.getRight().setLeft(new TreeNode(7));
        root1.getRight().setRight(new TreeNode(9));
        //actual
        BinaryTree bt = new BinaryTree<>(root1);
        bt.treeBFS();
        System.out.println();
        obj.postOrderTraversal_Iterative(root1);
        obj.postOrderTraversal_recursive(root1);
    }

}
