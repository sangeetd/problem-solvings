/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Stack;
import java.util.TreeMap;
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
    
    public void arrayElementMoreThan_NDivK(int[] a, int K){
        
        int N = a.length;
        int count = N/K;
        Map<Integer, Integer> map = new HashMap<>();
        for(int x: a){
            map.put(x, map.getOrDefault(x, 0) + 1);
        }
        
        map.entrySet().stream()
                .filter(e -> e.getValue() > count)
                .collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue()))
                .entrySet()
                .stream()
                .forEach(e -> System.out.println(e.getKey()));
        
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
    
    public void romanStringToDecimal(String str){
        
        //actual
        System.out.println("roman: "+str);
        
        Map<Character, Integer> roman = new HashMap<>();
        roman.put('I', 1);
        roman.put('V', 5);
        roman.put('X', 10);
        roman.put('L', 50);
        roman.put('C', 100);
        roman.put('D', 500);
        roman.put('M', 1000);
        
        int decimal = 0;
        for(int i=0; i<str.length(); i++){
            
            char c = str.charAt(i);
            if(i>0 && roman.get(str.charAt(i-1)) < roman.get(c)){
                decimal += roman.get(c) - 2*roman.get(str.charAt(i-1));
            }else {
                decimal += roman.get(c);
            }
            
        }
        
        //output
        System.out.println("Decimal: "+decimal);
        
    }
    
    public void longestCommonSubsequence(String a, String b){
        
        //memoization
        int[][] memo = new int[a.length()+1][b.length()+1];
        //base cond
        for(int[] x: memo){
            Arrays.fill(x, 0);
        }
        
        for(int x = 1; x<a.length()+1; x++){
            for(int y = 1; y<b.length()+1; y++){
                if(a.charAt(x - 1) == b.charAt(y - 1)){
                    memo[x][y] = memo[x-1][y-1] + 1;
                }else {
                    memo[x][y] = Math.max(memo[x-1][y], memo[x][y-1]);
                }
            }
        }
        
        int l = a.length();
        int m = b.length();
        StringBuilder sb = new StringBuilder();
        while(l>0 && m>0){
            
            if(a.charAt(l-1) == b.charAt(m - 1)){
                sb.insert(0,a.charAt(l-1));
                l--;
                m--;
            }else{
                
                if(memo[l-1][m] > memo[l][m-1]){
                    l--;
                }else{
                    m--;
                }
                
            }
            
        }
        
        //output
        System.out.println("Longest common subseq: "+sb.toString());
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
    
    private Stack<Integer> sumOfNumbersAsLinkedList_ToStack(Node<Integer> node){
        
        Stack<Integer> s = new Stack<>();
        Node<Integer> temp = node;
        while(temp != null){
            
            s.push(temp.getData());
            temp = temp.getNext();
            
        }
        
        return s;
        
    }
    
    public void sumOfNumbersAsLinkedList(Node<Integer> n1, Node<Integer> n2){
        
        Stack<Integer> nS1 = sumOfNumbersAsLinkedList_ToStack(n1);
        Stack<Integer> nS2 = sumOfNumbersAsLinkedList_ToStack(n2);
        
        int carry = 0;
        LinkedListUtil<Integer> ll = new LinkedListUtil<>();
        while(!nS1.isEmpty() || !nS2.isEmpty()){
            
            int sum = carry;
            
            if(!nS1.isEmpty()){
                sum += nS1.pop();
            }
            
            if(!nS2.isEmpty()){
                sum += nS2.pop();
            }
            
            carry = sum/10;
            ll.addAtHead(sum%10);
            
        }
        
        if(carry > 0){
            ll.addAtHead(carry);
        }
        
        //output
        ll.print();
        
    }
    
    public void removeDuplicateFromSortedLinkedList(Node<Integer> node){
        
        //actual
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(node);
        ll.print();
        
        Node<Integer> curr = node;
        Node<Integer> temp = node.getNext();
        
        while(temp != null){
            
            if(curr.getData() != temp.getData()){
                curr.setNext(temp);
                curr = temp;
            }
            
            temp = temp.getNext();
            
        }
        
        curr.setNext(temp);
        
        //output
        ll = new LinkedListUtil<>(node);
        ll.print();
        
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
    
    public void reverseLevelOrderTraversal(TreeNode<Integer> root){
        
        //actuals
        BinaryTree bt = new BinaryTree(root);
        bt.treeBFS();
        
        List<Integer> singleListReverseLevelOrder = new ArrayList<>();
        
        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);
        Queue<TreeNode<Integer>> intQ = new LinkedList<>();
        
        List<List<Integer>> level = new ArrayList<>();
        List<Integer> nodes = new ArrayList<>();
        
        while(!q.isEmpty()){
            
            TreeNode<Integer> temp = q.poll();
            nodes.add(temp.getData());
            
            if(temp.getLeft() != null){
                intQ.add(temp.getLeft());
            }
            
            if(temp.getRight() != null){
                intQ.add(temp.getRight());
            }
            
            if(q.isEmpty()){
                level.add(nodes);
                nodes = new ArrayList<>();
                q.addAll(intQ);
                intQ.clear();
            }
            
        }
        
        //output
        System.out.println();
        Collections.reverse(level);
        System.out.println("Level wise: " + level);
        
        for(List l: level){
            singleListReverseLevelOrder.addAll(l);
        }
        System.out.println("Single node list: " + singleListReverseLevelOrder);
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
    
    private void leftViewOfTree_Helper(TreeNode<Integer> root, int level, Map<Integer, Integer> result){
        if(root == null){
            return;
        }
        
        if(!result.containsKey(level)){
            result.put(level, root.getData());
        }
        
        //for left view
        leftViewOfTree_Helper(root.getLeft(), level+1, result);
        leftViewOfTree_Helper(root.getRight(), level+1, result);
    }
    
    public void leftViewOfTree(TreeNode<Integer> root){
        Map<Integer, Integer> result = new TreeMap<>();
        leftViewOfTree_Helper(root, 0, result);
        
        result.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue()+ " ");
        });
        
        System.out.println();
    }
    
    private void rightViewOfTree_Helper(TreeNode<Integer> root, int level, Map<Integer, Integer> result){
        
        if(root == null){
            return;
        }
        
        if(!result.containsKey(level)){
            result.put(level, root.getData());
        }
        
        //for right view
        rightViewOfTree_Helper(root.getRight(), level+1, result);
        rightViewOfTree_Helper(root.getLeft(), level+1, result);
    }
    
    public void rightViewOfTree(TreeNode<Integer> root){
        
        Map<Integer, Integer> result = new TreeMap<>();
        rightViewOfTree_Helper(root, 0, result);
        
        result.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue()+ " ");
        });
        
        System.out.println();
        
    }
    
    public void topViewOfTree(TreeNode<Integer> root){
        
        Queue<Pair<TreeNode<Integer>, Integer>> q = new LinkedList<>();
        q.add(new Pair<>(root, 0));
        
        Map<Integer, Integer> result = new TreeMap<>();
        
        while(!q.isEmpty()){
            
            Pair<TreeNode<Integer>, Integer> p = q.poll();
            TreeNode<Integer> n = p.getKey();
            int vLevel = p.getValue();
            
            if(!result.containsKey(vLevel)){
                result.put(vLevel, n.getData());
            }
            
            if(n.getLeft() != null){
                q.add(new Pair<>(n.getLeft(), vLevel-1));
            }
            if(n.getRight() != null){
                q.add(new Pair<>(n.getRight(), vLevel+1));
            }
        }
        
        result.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue()+ " ");
        });
        
        System.out.println();
        
    }
    
    public void bottomViewOfTree(TreeNode<Integer> root){
        
        //pair: node,vlevels
        Queue<Pair<TreeNode<Integer>, Integer>> q = new LinkedList<>();
        q.add(new Pair<>(root, 0));
        
        Map<Integer, Integer> bottomView = new TreeMap<>();
        
        while(!q.isEmpty()){
            
            Pair<TreeNode<Integer>, Integer> p = q.poll();
            TreeNode<Integer> n = p.getKey();
            int vLevel = p.getValue();
            
            //updates the vlevel with new node data, as we go down the tree in level order wise
            bottomView.put(vLevel, n.getData());
            
            if(n.getLeft() != null){
                q.add(new Pair<>(n.getLeft(), vLevel - 1));
            }
            
            if(n.getRight() != null){
                q.add(new Pair<>(n.getRight(), vLevel + 1));
            }
            
        }
        
        bottomView.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue()+ " ");
        });
        
        System.out.println();
    }
    
    public void zigZagTreeTraversal(TreeNode<Integer> root, boolean ltr){
        
        Stack<TreeNode<Integer>> s = new Stack<>();
        s.push(root);
        Stack<TreeNode<Integer>> intS = new Stack<>();
        
        List<List<Integer>> level = new ArrayList<>();
        List<Integer> zigZagNodes = new ArrayList<>();
        
        while(!s.isEmpty()){
            
            TreeNode<Integer> t = s.pop();
            zigZagNodes.add(t.getData());
            
            if(ltr){
               
                if(t.getRight() != null){
                    intS.push(t.getRight());
                }
                
                if(t.getLeft() != null){
                    intS.push(t.getLeft());
                }
                
            }else {
                
                if(t.getLeft() != null){
                    intS.push(t.getLeft());
                }
                
                if(t.getRight() != null){
                    intS.push(t.getRight());
                }
                
            }
            
            if(s.isEmpty()){
                
                ltr = !ltr;
                level.add(zigZagNodes);
                zigZagNodes = new ArrayList<>();
                s.addAll(intS);
                intS.clear();
            }
            
        }
        
        //output
        System.out.println("Output: "+level);
    }
    
    TreeNode treeToDoublyLinkedList_Prev;
    TreeNode treeToDoublyLinkedList_HeadOfDLL;
    private void treeToDoublyLinkedList_Helper(TreeNode root){
        if(root == null){
            return;
        }
        
        treeToDoublyLinkedList_Helper(root.getLeft());
        
        if(treeToDoublyLinkedList_Prev == null){
            treeToDoublyLinkedList_HeadOfDLL = root;
        }else{
            root.setLeft(treeToDoublyLinkedList_Prev);
            treeToDoublyLinkedList_Prev.setRight(root);
        }
        
        treeToDoublyLinkedList_Prev = root;
        
        treeToDoublyLinkedList_Helper(root.getRight());
    }
    
    private void treeToDoublyLinkedList_Print(){
        
        while(treeToDoublyLinkedList_HeadOfDLL != null){
            
            System.out.print(treeToDoublyLinkedList_HeadOfDLL.getData()+ " ");
            treeToDoublyLinkedList_HeadOfDLL = treeToDoublyLinkedList_HeadOfDLL.getRight();
            
        }
        System.out.println();
    }
    
    public void treeToDoublyLinkedList(TreeNode root){
        treeToDoublyLinkedList_Helper(root);
        treeToDoublyLinkedList_Print();
        //just resetting
        treeToDoublyLinkedList_Prev = null;
        treeToDoublyLinkedList_HeadOfDLL = null;
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
//        System.out.println("Postsorder traversal of tree Iterative/recursive");
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
//        obj.postOrderTraversal_Iterative(root1);
//        obj.postOrderTraversal_recursive(root1);
        //......................................................................
//        Row: 148
//        System.out.println("Add two numbers represented by linked list");
//        Node<Integer> n1 = new Node<>(4);
//        n1.setNext(new Node<>(5));
//        Node<Integer> n2 = new Node<>(3);
//        n2.setNext(new Node<>(4));
//        n2.getNext().setNext(new Node<>(5));
//        obj.sumOfNumbersAsLinkedList(n1, n2);
        //......................................................................
//        Row: 178
//        System.out.println("Reverse level order traversal");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.reverseLevelOrderTraversal(root1);
        //......................................................................
//        Row: 185
//        System.out.println("Left view of tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.leftViewOfTree(root1);
        //......................................................................
//        Row: 186
//        System.out.println("Right view of tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.rightViewOfTree(root1);
        //......................................................................
//        Row: 187
//        System.out.println("Top view of tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.topViewOfTree(root1);
        //......................................................................
//        Row: 188
//        System.out.println("Bottom view of tree");
//        //https://practice.geeksforgeeks.org/problems/bottom-view-of-binary-tree/1
//        TreeNode<Integer> root1 = new TreeNode<>(20);
//        root1.setLeft(new TreeNode(8));
//        root1.getLeft().setLeft(new TreeNode(5));
//        root1.getLeft().setRight(new TreeNode(3));
//        root1.getLeft().getRight().setLeft(new TreeNode(10));
//        root1.setRight(new TreeNode(22));
//        root1.getRight().setLeft(new TreeNode(4));
//        root1.getRight().setRight(new TreeNode(25));
//        root1.getRight().getLeft().setRight(new TreeNode(14));
//        obj.bottomViewOfTree(root1);
        //......................................................................
//        Row: 189
//        System.out.println("Zig zag traversal of tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.zigZagTreeTraversal(root1, true);
//        root1 = new TreeNode<>(20);
//        root1.setLeft(new TreeNode(8));
//        root1.getLeft().setLeft(new TreeNode(5));
//        root1.getLeft().setRight(new TreeNode(3));
//        root1.getLeft().getRight().setLeft(new TreeNode(10));
//        root1.setRight(new TreeNode(22));
//        root1.getRight().setLeft(new TreeNode(4));
//        root1.getRight().setRight(new TreeNode(25));
//        root1.getRight().getLeft().setRight(new TreeNode(14));
//        obj.zigZagTreeTraversal(root1, false);
        //......................................................................
//        Row: 30
//        System.out.println("All the element from array[N] and given K that occurs more than N/K times");
//        obj.arrayElementMoreThan_NDivK(new int[]{3, 1, 2, 2, 1, 2, 3, 3}, 4);
        //......................................................................
//        Row: 81
//        System.out.println("Roman numeral string to decimal");
//        obj.romanStringToDecimal("III");
//        obj.romanStringToDecimal("CI");
//        obj.romanStringToDecimal("IM");
//        obj.romanStringToDecimal("V");
        //......................................................................
//        Row: 86
//        System.out.println("Longest commn subsequence");
//        obj.longestCommonSubsequence("ababcba", "ababcba");
//        obj.longestCommonSubsequence("abxayzbcpqba", "kgxyhgtzpnlerq");
//        obj.longestCommonSubsequence("abcd", "pqrs");
        //......................................................................
//        Row: 144
//        System.out.println("Remove duplicates from sorted linked list");
//        Node<Integer> node1 = new Node<>(1);
//        node1.setNext(new Node<>(1));
//        node1.getNext().setNext(new Node<>(2));
//        node1.getNext().getNext().setNext(new Node<>(2));
//        node1.getNext().getNext().getNext().setNext(new Node<>(2));
//        node1.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(4));
//        node1.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        node1.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.removeDuplicateFromSortedLinkedList(node1);
//        node1 = new Node<>(1);
//        node1.setNext(new Node<>(2));
//        node1.getNext().setNext(new Node<>(2));
//        node1.getNext().getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(4));
//        node1.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(4));
//        node1.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.removeDuplicateFromSortedLinkedList(node1);
//        node1 = new Node<>(1);
//        node1.setNext(new Node<>(2));
//        node1.getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().setNext(new Node<>(4));
//        node1.getNext().getNext().getNext().setNext(new Node<>(5));
//        node1.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.removeDuplicateFromSortedLinkedList(node1);
        //......................................................................
//        Row: 194
        System.out.println("Convert tree to doubly linked list");
        TreeNode<Integer> root1 = new TreeNode<>(6);
        root1.setLeft(new TreeNode(2));
        root1.getLeft().setLeft(new TreeNode(0));
        root1.getLeft().setRight(new TreeNode(4));
        root1.getLeft().getRight().setLeft(new TreeNode(3));
        root1.getLeft().getRight().setRight(new TreeNode(5));
        root1.setRight(new TreeNode(8));
        root1.getRight().setLeft(new TreeNode(7));
        root1.getRight().setRight(new TreeNode(9));
        obj.treeToDoublyLinkedList(root1);
        root1 = new TreeNode<>(20);
        root1.setLeft(new TreeNode(8));
        root1.getLeft().setLeft(new TreeNode(5));
        root1.getLeft().setRight(new TreeNode(3));
        root1.getLeft().getRight().setLeft(new TreeNode(10));
        root1.setRight(new TreeNode(22));
        root1.getRight().setLeft(new TreeNode(4));
        root1.getRight().setRight(new TreeNode(25));
        root1.getRight().getLeft().setRight(new TreeNode(14));
        obj.treeToDoublyLinkedList(root1);
    }

}
