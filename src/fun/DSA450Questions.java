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
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
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

    public void arrayElementMoreThan_NDivK(int[] a, int K) {

        int N = a.length;
        int count = N / K;
        Map<Integer, Integer> map = new HashMap<>();
        for (int x : a) {
            map.put(x, map.getOrDefault(x, 0) + 1);
        }

        map.entrySet().stream()
                .filter(e -> e.getValue() > count)
                .collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue()))
                .entrySet()
                .stream()
                .forEach(e -> System.out.println(e.getKey()));

    }

    public void minMaxInArray_1(int[] a) {

        //...................T: O(N)
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;

        for (int i = 0; i < a.length; i++) {

            max = Math.max(max, a[i]);
            min = Math.min(min, a[i]);

        }

        //outpur
        System.out.println("Min and max value in array: " + min + " " + max);

    }

    public void minMaxInArray_2(int[] a) {

        //https://www.geeksforgeeks.org/maximum-and-minimum-in-an-array/
        //...................T: O(N)
        //Min no of comparision
        int min = Integer.MIN_VALUE;
        int max = Integer.MAX_VALUE;

        int n = a.length;
        int itr = 0;
        //check if array is even/odd
        if (n % 2 == 0) {
            max = Math.max(a[0], a[1]);
            min = Math.min(a[0], a[1]);
            //in case of even choose min & max from first tw element
            //and set itr to start from 2nd index i.e(3rd element) in pair wise
            itr = 2;
        } else {
            max = a[0];
            min = a[0];
            //in case of odd choose first element as min & max both
            //set itr to 1 i.e, 2nd element
            itr = 1;
        }

        //since we checking itr and itr+1 value in loop 
        //so run loop to n-1 so that itr+1th element corresponds to n-1th element
        while (itr < n - 1) {

            //check current itr and itr+1 element
            if (a[itr] > a[itr + 1]) {
                max = Math.max(max, a[itr]);
                min = Math.min(min, a[itr + 1]);
            } else {
                max = Math.max(max, a[itr + 1]);
                min = Math.min(min, a[itr]);
            }
            itr++;
        }

        //outpur
        System.out.println("Min and max value in array: " + min + " " + max);

    }

    public void kThSmallestElementInArray(int[] a, int K) {

        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(
                (o1, o2) -> o2.compareTo(o1)
        );

        for (int x : a) {
            maxHeap.add(x);
            if (maxHeap.size() > K) {
                maxHeap.poll();
            }
        }

        //output
        System.out.println(K + " th smallest element: " + maxHeap.peek());

    }

    public void kThLargestElementInArray(int[] a, int K) {

        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        for (int x : a) {
            minHeap.add(x);
            if (minHeap.size() > K) {
                minHeap.poll();
            }
        }

        //output
        System.out.println(K + " th largest element: " + minHeap.peek());

    }

    public void sortArrayOf012_1(int[] a) {

        //.............T: O(N)
        //.............S: O(3)
        Map<Integer, Integer> map = new TreeMap<>();
        for (int x : a) {
            map.put(x, map.getOrDefault(x, 0) + 1);
        }

        //creating array
        int k = 0;
        for (Map.Entry<Integer, Integer> e : map.entrySet()) {
            for (int i = 0; i < e.getValue(); i++) {
                a[k++] = e.getKey();
            }
        }

        //output
        for (int x : a) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    private void swapIntArray(int[] a, int x, int y) {
        int temp;
        temp = a[x];
        a[x] = a[y];
        a[y] = temp;
    }

    public void sortArrayOf012_2(int[] a) {

        //.............T: O(N)
        //.............S: O(1)
        //based on Dutch National Flag Algorithm
        //https://www.geeksforgeeks.org/sort-an-array-of-0s-1s-and-2s/
        int lo = 0;
        int hi = a.length - 1;
        int mid = 0, temp = 0;
        while (mid <= hi) {
            switch (a[mid]) {
                case 0: {
                    swapIntArray(a, lo, mid);
                    lo++;
                    mid++;
                    break;
                }
                case 1:
                    mid++;
                    break;
                case 2: {
                    swapIntArray(a, mid, hi);
                    hi--;
                    break;
                }
            }
        }

        //output
        for (int x : a) {
            System.out.print(x + " ");
        }
        System.out.println();

    }

    public void rotateMatrixClockWise90Deg(int[][] mat) {

        int row = mat.length;
        int col = mat[0].length;
        int N = mat.length;
        for (int x = 0; x < N / 2; x++) {
            for (int y = x; y < N - x - 1; y++) {

                int temp = mat[x][y];
                mat[x][y] = mat[y][N - 1 - x];
                mat[y][N - 1 - x] = mat[N - 1 - x][N - 1 - y];
                mat[N - 1 - x][N - 1 - y] = mat[N - 1 - y][x];
                mat[N - 1 - y][x] = temp;

            }
        }

        //output
        for (int[] r : mat) {
            for (int c : r) {
                System.out.print(c + "\t");
            }
            System.out.println();
        }

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

    public void romanStringToDecimal(String str) {

        //actual
        System.out.println("roman: " + str);

        Map<Character, Integer> roman = new HashMap<>();
        roman.put('I', 1);
        roman.put('V', 5);
        roman.put('X', 10);
        roman.put('L', 50);
        roman.put('C', 100);
        roman.put('D', 500);
        roman.put('M', 1000);

        int decimal = 0;
        for (int i = 0; i < str.length(); i++) {

            char c = str.charAt(i);
            if (i > 0 && roman.get(str.charAt(i - 1)) < roman.get(c)) {
                decimal += roman.get(c) - 2 * roman.get(str.charAt(i - 1));
            } else {
                decimal += roman.get(c);
            }

        }

        //output
        System.out.println("Decimal: " + decimal);

    }

    public void longestCommonSubsequence(String a, String b) {

        //memoization
        int[][] memo = new int[a.length() + 1][b.length() + 1];
        //base cond
        for (int[] x : memo) {
            Arrays.fill(x, 0);
        }

        for (int x = 1; x < a.length() + 1; x++) {
            for (int y = 1; y < b.length() + 1; y++) {
                if (a.charAt(x - 1) == b.charAt(y - 1)) {
                    memo[x][y] = memo[x - 1][y - 1] + 1;
                } else {
                    memo[x][y] = Math.max(memo[x - 1][y], memo[x][y - 1]);
                }
            }
        }

        int l = a.length();
        int m = b.length();
        StringBuilder sb = new StringBuilder();
        while (l > 0 && m > 0) {

            if (a.charAt(l - 1) == b.charAt(m - 1)) {
                sb.insert(0, a.charAt(l - 1));
                l--;
                m--;
            } else {

                if (memo[l - 1][m] > memo[l][m - 1]) {
                    l--;
                } else {
                    m--;
                }

            }

        }

        //output
        System.out.println("Longest common subseq: " + sb.toString());
    }

    public String countAndSay_Helper(int n) {

        //https://leetcode.com/problems/count-and-say/
        //base cond
        if (n == 1) {
            return "1";
        }

        String ans = countAndSay_Helper(n - 1);

        StringBuilder sb = new StringBuilder();

        char ch = ans.charAt(0);
        int counter = 1;
        //for i==ans.length() i.e very last itr of loop
        //this itr will only invoke else cond below
        for (int i = 1; i <= ans.length(); i++) {
            //i<ans.length() bound the calculations upto string length
            if (i < ans.length() && ans.charAt(i) == ch) {
                counter++;
            } else {
                sb.append(counter).append(ch);
                //i<ans.length() bound the calculations upto string length
                if (i < ans.length()) {
                    ch = ans.charAt(i);
                }

                counter = 1;
            }

        }

        return sb.toString();

    }

    public void countAndSay(int n) {
        System.out.println("Count and say: " + countAndSay_Helper(n));
    }

    public void removeConsecutiveDuplicateInString(String str) {

        //https://www.geeksforgeeks.org/remove-consecutive-duplicates-string/
        char[] ch = str.toCharArray();
        int f = 0;
        int l = 1;

        while (l < ch.length) {

            if (ch[f] == ch[l]) {
                l++;
            } else {
                ch[f + 1] = ch[l];
                f++;
            }

        }

        System.out.println("output: " + String.valueOf(ch, 0, f + 1));

    }

    private void printSentencesFromCollectionOfWords_Propagte_Recursion(String[][] words,
            int m, int n,
            String[] output) {
        // Add current word to output array
        output[m] = words[m][n];

        // If this is last word of 
        // current output sentence, 
        // then print the output sentence
        if (m == words.length - 1) {
            for (int i = 0; i < words.length; i++) {
                System.out.print(output[i] + " ");
            }
            System.out.println();
            return;
        }

        // Recur for next row
        for (int i = 0; i < words.length; i++) {
            if (words[m + 1][i] != "" && m < words.length) {
                printSentencesFromCollectionOfWords_Propagte_Recursion(words, m + 1, i, output);
            }
        }
    }

    public void printSentencesFromCollectionOfWords(String[][] words) {

        //https://www.geeksforgeeks.org/recursively-print-all-sentences-that-can-be-formed-from-list-of-word-lists/
        String[] output = new String[words.length];

        // Consider all words for first 
        // row as starting points and
        // print all sentences
        for (int i = 0; i < words.length; i++) {
            if (words[0][i] != "") {
                printSentencesFromCollectionOfWords_Propagte_Recursion(words, 0, i, output);
            }
        }
    }

    public void longestPrefixAlsoSuffixInString_KMPAlgo(String s) {
        int N = s.length();
        int[] lps = new int[N];
        lps[0] = 1;
        int len = 0;
        int i = 1;
        while (i < N) {

            if (s.charAt(i) == s.charAt(len)) {
                len++;
                lps[i] = len;
                i++;
            } else {

                if (len != 0) {
                    len = lps[len - 1];
                } else {
                    lps[i] = 0;
                    i++;
                }

            }

        }

        int res = lps[N - 1];

        System.out.println("Length of longest prefix: " + ((res > N / 2) ? N / 2 : res));

    }

    class CharCount {

        //reorganizeString
        char letter;
        int count;

        public CharCount(char letter, int count) {
            this.letter = letter;
            this.count = count;
        }
    }

    public String reorganizeString(String S) {

        //https://leetcode.com/problems/reorganize-string/
        int N = S.length();
        int[] charCount = new int[26];
        for (char c : S.toCharArray()) {
            charCount[c - 'a']++;
        }

        PriorityQueue<CharCount> heap = new PriorityQueue<>(
                (o1, o2) -> o1.count == o2.count ? o1.letter - o2.letter : o2.count - o1.count
        );

        for (int i = 0; i < 26; i++) {

            if (charCount[i] > 0) {
                if (charCount[i] > (N + 1) / 2) {
                    return "";
                }
                heap.add(new CharCount((char) (i + 'a'), charCount[i]));
            }

        }

        StringBuilder sb = new StringBuilder();
        while (heap.size() >= 2) {

            CharCount chC1 = heap.poll();
            CharCount chC2 = heap.poll();

            sb.append(chC1.letter);
            sb.append(chC2.letter);

            if (--chC1.count > 0) {
                heap.add(chC1);
            }
            if (--chC2.count > 0) {
                heap.add(chC2);
            }

        }

        if (heap.size() > 0) {
            sb.append(heap.poll().letter);
        }

        return sb.toString();

    }

    public void longestCommonPrefix(String[] strs) {

        //https://leetcode.com/problems/longest-common-prefix/
        if (strs == null || strs.length == 0) {
            return;
        }

        if (strs.length == 1) {
            System.out.println("Longest common prefix in list of strings: " + strs[0]);
            return;
        }

        String prefix = strs[0]; // first string as starting point
        int minLenPrefix = Integer.MAX_VALUE;
        for (int i = 1; i < strs.length; i++) {
            String s = strs[i];
            int index = 0;
            while (index < Math.min(s.length(), prefix.length())) {

                if (prefix.charAt(index) != s.charAt(index)) {
                    break;
                }
                index++;

            }

            minLenPrefix = Math.min(minLenPrefix, (index - 0));
        }

        System.out.println("Longest common prefix in list of strings: " + (prefix.length() >= minLenPrefix ? prefix.substring(0, minLenPrefix) : ""));

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

    private Stack<Integer> sumOfNumbersAsLinkedList_ToStack(Node<Integer> node) {

        Stack<Integer> s = new Stack<>();
        Node<Integer> temp = node;
        while (temp != null) {

            s.push(temp.getData());
            temp = temp.getNext();

        }

        return s;

    }

    public void sumOfNumbersAsLinkedList(Node<Integer> n1, Node<Integer> n2) {

        Stack<Integer> nS1 = sumOfNumbersAsLinkedList_ToStack(n1);
        Stack<Integer> nS2 = sumOfNumbersAsLinkedList_ToStack(n2);

        int carry = 0;
        LinkedListUtil<Integer> ll = new LinkedListUtil<>();
        while (!nS1.isEmpty() || !nS2.isEmpty()) {

            int sum = carry;

            if (!nS1.isEmpty()) {
                sum += nS1.pop();
            }

            if (!nS2.isEmpty()) {
                sum += nS2.pop();
            }

            carry = sum / 10;
            ll.addAtHead(sum % 10);

        }

        if (carry > 0) {
            ll.addAtHead(carry);
        }

        //output
        ll.print();

    }

    public void removeDuplicateFromSortedLinkedList(Node<Integer> node) {

        //actual
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(node);
        ll.print();

        Node<Integer> curr = node;
        Node<Integer> temp = node.getNext();

        while (temp != null) {

            if (curr.getData() != temp.getData()) {
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

    public void mergeKSortedLinkedList(Node<Integer>[] nodes) {

        //HEAP based method
        PriorityQueue<Node<Integer>> minHeap = new PriorityQueue<>(
                (o1, o2) -> o1.getData().compareTo(o2.getData())
        );

        for (Node<Integer> x : nodes) {
            while (x != null) {
                minHeap.add(x);
                x = x.getNext();
            }
        }

        //head to point arbitary infinite value to start with
        Node<Integer> head = new Node<>(Integer.MIN_VALUE);
        //saving the actual head's ref
        Node<Integer> copyHead = head;
        while (!minHeap.isEmpty()) {

            copyHead.setNext(minHeap.poll());
            copyHead = copyHead.getNext();

        }

        //actual merged list starts with next of arbitary head pointer
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(head.getNext());
        ll.print();
    }

    public void kThNodeFromEndOfLinkedList_1(Node node, int K) {

        //1. Approach
        //using additional space (Stack)
        //................O(N)+O(K)
        //time O(N) creating stack of N nodes from linked list + O(K) reaching out to Kth node
        //in the stack.
        //.......................space complexity O(N)
        Stack<Node> s = new Stack<>();
        Node temp = node;
        //T: O(N)
        //S: O{N}
        while (temp != null) {
            s.push(temp);
            temp = temp.getNext();
        }

        //T: O(K)
        while (!s.isEmpty()) {

            K--;
            Object element = s.pop().getData();
            if (K == 0) {
                System.out.println("Kth node from end is: " + element);
            }

        }

    }

    public void kThNodeFromEndOfLinkedList_2(Node node, int K) {

        //2. Approach
        //using Len - K + 1 formula
        //calculate the full length of the linked list frst 
        //then move the head pointer upto (Len - K + 1) limit which
        // is Kth node from the end
        //.................T: O(N) + O(Len - K + 1)
        //1. calculating Len O(N)
        //2. moving to Len - k + 1 pointer is O(Len - K + 1)
        int len = 0;
        Node temp = node;
        while (temp != null) {
            temp = temp.getNext();
            len++;
        }

        //Kth node from end = len - K + 1
        temp = node;
        //i=1 as we consider the first node from 1 onwards
        for (int i = 1; i < (len - K + 1); i++) {
            temp = temp.getNext();
        }

        //output
        System.out.println("Kth node from end is: " + temp.getData());

    }

    public void kThNodeFromEndOfLinkedList_3(Node node, int K) {

        //3. Approach (OPTIMISED)
        //Two pointer method
        //Theory: 
        //maintain ref pointer, main pointer
        //both start from the head ref
        //move ref pointer to K dist. Once ref pointer reaches the K dist from main pointer
        //start moving the ref and main pointer one by one.
        //at the time ref pointer reaches the end of linked list
        //main pointer will be K dist behind the ref pointer(already at end now)
        //print the main pointer that will be answer
        //............T: O(N) S: O(1)
        Node ref = node;
        Node main = node;

        while (K-- != 0) {
            ref = ref.getNext();
        }

        //now ref is K dist ahead of main pointer
        //now move both pointer one by one
        //until ref reaches end of linked list
        //bt the time main pointer will be K dist behind the ref pointer
        while (ref != null) {

            main = main.getNext();
            ref = ref.getNext();

        }

        //output
        System.out.println("Kth node from end is: " + main.getData());

    }

    public Node<Integer> reverseLinkedListInKGroups(Node<Integer> node, int K) {

        //https://www.geeksforgeeks.org/reverse-a-list-in-groups-of-given-size/
        Node current = node;
        Node next = null;
        Node prev = null;

        int count = 0;

        /* Reverse first k nodes of linked list */
        while (count < K && current != null) {
            next = current.getNext();
            current.setNext(prev);
            prev = current;
            current = next;
            count++;
        }

        /* next is now a pointer to (k+1)th node  
         Recursively call for the list starting from current. 
         And make rest of the list as next of first node */
        if (next != null) {
            node.setNext(reverseLinkedListInKGroups(next, K));
        }

        // prev is now head of input list 
        return prev;

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

    public void reverseLevelOrderTraversal(TreeNode<Integer> root) {

        //actuals
        BinaryTree bt = new BinaryTree(root);
        bt.treeBFS();

        List<Integer> singleListReverseLevelOrder = new ArrayList<>();

        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);
        Queue<TreeNode<Integer>> intQ = new LinkedList<>();

        List<List<Integer>> level = new ArrayList<>();
        List<Integer> nodes = new ArrayList<>();

        while (!q.isEmpty()) {

            TreeNode<Integer> temp = q.poll();
            nodes.add(temp.getData());

            if (temp.getLeft() != null) {
                intQ.add(temp.getLeft());
            }

            if (temp.getRight() != null) {
                intQ.add(temp.getRight());
            }

            if (q.isEmpty()) {
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

        for (List l : level) {
            singleListReverseLevelOrder.addAll(l);
        }
        System.out.println("Single node list: " + singleListReverseLevelOrder);
    }

    public void inOrderTraversal_Iterative(TreeNode root) {

        if (root == null) {
            return;
        }

        Stack<Pair<TreeNode, Integer>> stack = new Stack<>();
        stack.push(new Pair<>(root, 0));

        while (!stack.isEmpty()) {

            Pair<TreeNode, Integer> p = stack.pop();
            TreeNode n = p.getKey();
            int status = p.getValue();

            if (n == null || status == 3) {
                continue;
            }

            stack.push(new Pair<>(n, status + 1));

            if (status == 0) {
                stack.push(new Pair<>(n.getLeft(), 0));
            }

            if (status == 1) {
                System.out.print(n.getData() + " ");
            }

            if (status == 2) {
                stack.push(new Pair<>(n.getRight(), 0));
            }

        }

        System.out.println();

    }

    public void inOrderTraversal_Recursive(TreeNode root) {

        if (root == null) {
            return;
        }

        inOrderTraversal_Recursive(root.getLeft());
        System.out.print(root.getData() + " ");
        inOrderTraversal_Recursive(root.getRight());
    }

    public void preOrderTraversal_Iterative(TreeNode root) {

        if (root == null) {
            return;
        }

        Stack<Pair<TreeNode, Integer>> stack = new Stack<>();
        stack.push(new Pair<>(root, 0));

        while (!stack.isEmpty()) {

            Pair<TreeNode, Integer> p = stack.pop();
            TreeNode n = p.getKey();
            int status = p.getValue();

            if (n == null || status == 3) {
                continue;
            }

            stack.push(new Pair<>(n, status + 1));

            if (status == 0) {
                System.out.print(n.getData() + " ");
            }

            if (status == 1) {
                stack.push(new Pair<>(n.getLeft(), 0));
            }

            if (status == 2) {
                stack.push(new Pair<>(n.getRight(), 0));
            }

        }

        System.out.println();

    }

    public void preOrderTraversal_Recursive(TreeNode root) {

        if (root == null) {
            return;
        }

        System.out.print(root.getData() + " ");
        preOrderTraversal_Recursive(root.getLeft());
        preOrderTraversal_Recursive(root.getRight());

    }

    public void postOrderTraversal_Iterative(TreeNode root) {

        if (root == null) {
            return;
        }

        Stack<Pair<TreeNode, Integer>> stack = new Stack<>();
        stack.push(new Pair<>(root, 0));

        while (!stack.isEmpty()) {

            Pair<TreeNode, Integer> p = stack.pop();
            TreeNode n = p.getKey();
            int status = p.getValue();

            if (n == null || status == 3) {
                continue;
            }

            stack.push(new Pair<>(n, status + 1));

            if (status == 0) {
                stack.push(new Pair<>(n.getLeft(), 0));
            }

            if (status == 1) {
                stack.push(new Pair<>(n.getRight(), 0));
            }

            if (status == 2) {
                System.out.print(n.getData() + " ");
            }

        }

        System.out.println();
    }

    public void postOrderTraversal_recursive(TreeNode root) {

        if (root == null) {
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

    private void leftViewOfTree_Helper(TreeNode<Integer> root, int level, Map<Integer, Integer> result) {
        if (root == null) {
            return;
        }

        if (!result.containsKey(level)) {
            result.put(level, root.getData());
        }

        //for left view
        leftViewOfTree_Helper(root.getLeft(), level + 1, result);
        leftViewOfTree_Helper(root.getRight(), level + 1, result);
    }

    public void leftViewOfTree(TreeNode<Integer> root) {
        Map<Integer, Integer> result = new TreeMap<>();
        leftViewOfTree_Helper(root, 0, result);

        result.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue() + " ");
        });

        System.out.println();
    }

    private void rightViewOfTree_Helper(TreeNode<Integer> root, int level, Map<Integer, Integer> result) {

        if (root == null) {
            return;
        }

        if (!result.containsKey(level)) {
            result.put(level, root.getData());
        }

        //for right view
        rightViewOfTree_Helper(root.getRight(), level + 1, result);
        rightViewOfTree_Helper(root.getLeft(), level + 1, result);
    }

    public void rightViewOfTree(TreeNode<Integer> root) {

        Map<Integer, Integer> result = new TreeMap<>();
        rightViewOfTree_Helper(root, 0, result);

        result.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue() + " ");
        });

        System.out.println();

    }

    public void topViewOfTree(TreeNode<Integer> root) {

        Queue<Pair<TreeNode<Integer>, Integer>> q = new LinkedList<>();
        q.add(new Pair<>(root, 0));

        Map<Integer, Integer> result = new TreeMap<>();

        while (!q.isEmpty()) {

            Pair<TreeNode<Integer>, Integer> p = q.poll();
            TreeNode<Integer> n = p.getKey();
            int vLevel = p.getValue();

            if (!result.containsKey(vLevel)) {
                result.put(vLevel, n.getData());
            }

            if (n.getLeft() != null) {
                q.add(new Pair<>(n.getLeft(), vLevel - 1));
            }
            if (n.getRight() != null) {
                q.add(new Pair<>(n.getRight(), vLevel + 1));
            }
        }

        result.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue() + " ");
        });

        System.out.println();

    }

    public void bottomViewOfTree(TreeNode<Integer> root) {

        //pair: node,vlevels
        Queue<Pair<TreeNode<Integer>, Integer>> q = new LinkedList<>();
        q.add(new Pair<>(root, 0));

        Map<Integer, Integer> bottomView = new TreeMap<>();

        while (!q.isEmpty()) {

            Pair<TreeNode<Integer>, Integer> p = q.poll();
            TreeNode<Integer> n = p.getKey();
            int vLevel = p.getValue();

            //updates the vlevel with new node data, as we go down the tree in level order wise
            bottomView.put(vLevel, n.getData());

            if (n.getLeft() != null) {
                q.add(new Pair<>(n.getLeft(), vLevel - 1));
            }

            if (n.getRight() != null) {
                q.add(new Pair<>(n.getRight(), vLevel + 1));
            }

        }

        bottomView.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue() + " ");
        });

        System.out.println();
    }

    public void zigZagTreeTraversal(TreeNode<Integer> root, boolean ltr) {

        Stack<TreeNode<Integer>> s = new Stack<>();
        s.push(root);
        Stack<TreeNode<Integer>> intS = new Stack<>();

        List<List<Integer>> level = new ArrayList<>();
        List<Integer> zigZagNodes = new ArrayList<>();

        while (!s.isEmpty()) {

            TreeNode<Integer> t = s.pop();
            zigZagNodes.add(t.getData());

            if (ltr) {

                if (t.getRight() != null) {
                    intS.push(t.getRight());
                }

                if (t.getLeft() != null) {
                    intS.push(t.getLeft());
                }

            } else {

                if (t.getLeft() != null) {
                    intS.push(t.getLeft());
                }

                if (t.getRight() != null) {
                    intS.push(t.getRight());
                }

            }

            if (s.isEmpty()) {

                ltr = !ltr;
                level.add(zigZagNodes);
                zigZagNodes = new ArrayList<>();
                s.addAll(intS);
                intS.clear();
            }

        }

        //output
        System.out.println("Output: " + level);
    }

    private void minAndMaxInBST_Helper(TreeNode<Integer> root, List<Integer> l) {

        if (root == null) {
            return;
        }

        //inorder traversal
        minAndMaxInBST_Helper(root.getLeft(), l);
        if (root != null) {
            l.add(root.getData());
        }
        minAndMaxInBST_Helper(root.getRight(), l);
    }

    public void minAndMaxInBST(TreeNode<Integer> root) {
        List<Integer> inOrder = new ArrayList<>();
        minAndMaxInBST_Helper(root, inOrder);

        System.out.println("Min & Max in BST: " + inOrder.get(0) + " " + inOrder.get(inOrder.size() - 1));

    }

    TreeNode treeToDoublyLinkedList_Prev;
    TreeNode treeToDoublyLinkedList_HeadOfDLL;

    private void treeToDoublyLinkedList_Helper(TreeNode root) {
        if (root == null) {
            return;
        }

        treeToDoublyLinkedList_Helper(root.getLeft());

        if (treeToDoublyLinkedList_Prev == null) {
            treeToDoublyLinkedList_HeadOfDLL = root;
        } else {
            root.setLeft(treeToDoublyLinkedList_Prev);
            treeToDoublyLinkedList_Prev.setRight(root);
        }

        treeToDoublyLinkedList_Prev = root;

        treeToDoublyLinkedList_Helper(root.getRight());
    }

    private void treeToDoublyLinkedList_Print() {

        while (treeToDoublyLinkedList_HeadOfDLL != null) {

            System.out.print(treeToDoublyLinkedList_HeadOfDLL.getData() + " ");
            treeToDoublyLinkedList_HeadOfDLL = treeToDoublyLinkedList_HeadOfDLL.getRight();

        }
        System.out.println();
    }

    public void treeToDoublyLinkedList(TreeNode root) {
        treeToDoublyLinkedList_Helper(root);
        treeToDoublyLinkedList_Print();
        //just resetting
        treeToDoublyLinkedList_Prev = null;
        treeToDoublyLinkedList_HeadOfDLL = null;
    }

    private void checkIfAllLeafNodeOfTreeAtSameLevel_Helper(TreeNode root, int level, Set<Integer> levels) {

        if (root == null) {
            return;
        }

        //leaf
        if (root.getLeft() == null && root.getRight() == null) {
            levels.add(level);
        }

        checkIfAllLeafNodeOfTreeAtSameLevel_Helper(root.getLeft(), level + 1, levels);
        checkIfAllLeafNodeOfTreeAtSameLevel_Helper(root.getRight(), level + 1, levels);

    }

    public void checkIfAllLeafNodeOfTreeAtSameLevel(TreeNode root) {
        Set<Integer> levels = new HashSet<>();
        checkIfAllLeafNodeOfTreeAtSameLevel_Helper(root, 0, levels);

        System.out.println("Leaf at same level: " + (levels.size() == 1));

    }

    TreeNode<Integer> isTreeBST_Prev;

    private boolean isTreeBST_Helper(TreeNode<Integer> root) {

        if (root == null) {
            return true;
        }

        isTreeBST_Helper(root.getLeft());
        if (isTreeBST_Prev != null && isTreeBST_Prev.getData() > root.getData()) {
            return false;
        }

        isTreeBST_Prev = root;
        return isTreeBST_Helper(root.getRight());
    }

    public void isTreeBST(TreeNode<Integer> root) {

        System.out.println("Tree is BST: " + isTreeBST_Helper(root));
        //just resetting
        isTreeBST_Prev = null;
    }

    private void kThLargestNodeInBST_Helper(TreeNode<Integer> root, int K, PriorityQueue<Integer> minHeap) {

        if (root == null) {
            return;
        }

        minHeap.add(root.getData());
        if (minHeap.size() > K) {
            minHeap.poll();
        }

        kThLargestNodeInBST_Helper(root.getLeft(), K, minHeap);
        kThLargestNodeInBST_Helper(root.getRight(), K, minHeap);

    }

    public void kTHLargestNodeInBST(TreeNode<Integer> root, int K) {
        //actual
        //inorder of BST is sorted nodes list
        inOrderTraversal_Iterative(root);

        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        kThLargestNodeInBST_Helper(root, K, minHeap);

        System.out.println(K + " largest node from BST: " + minHeap.poll());
    }

    private void kThSmallestNodeInBST_Helper(TreeNode<Integer> root, int K, PriorityQueue<Integer> maxHeap) {

        if (root == null) {
            return;
        }

        maxHeap.add(root.getData());
        if (maxHeap.size() > K) {
            maxHeap.poll();
        }

        kThLargestNodeInBST_Helper(root.getLeft(), K, maxHeap);
        kThLargestNodeInBST_Helper(root.getRight(), K, maxHeap);

    }

    public void kTHSmallestNodeInBST(TreeNode<Integer> root, int K) {
        //actual
        //inorder of BST is sorted nodes list
        inOrderTraversal_Iterative(root);

        //maxHeap
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(
                (o1, o2) -> o2.compareTo(o1)
        );
        kThSmallestNodeInBST_Helper(root, K, maxHeap);

        System.out.println(K + " smallest node from BST: " + maxHeap.poll());
    }

    class Height {

        int height = 0;
    }

    private boolean isTreeHeightBalanced_Helper(TreeNode root, Height h) {

        //this approach calculates height and check height balanced at the same time
        if (root == null) {
            h.height = -1;
            return true;
        }

        Height lh = new Height();
        Height rh = new Height();

        boolean isLeftBal = isTreeHeightBalanced_Helper(root.getLeft(), lh);
        boolean isRightBal = isTreeHeightBalanced_Helper(root.getRight(), rh);

        //calculate the height for the current node
        h.height = Math.max(lh.height, rh.height) + 1;

        //checking the cond if height balanced
        //if diff b/w left subtree or right sub tree is greater than 1 it's
        //not balanced
        if (Math.abs(lh.height - rh.height) > 1) {
            return false;
        }

        //if the above cond doesn't fulfil
        //it should check if any of the left or right sub tree both are balanced or not
        return isLeftBal && isRightBal;

    }

    public void isTreeHeightBalanced(TreeNode root) {
        Height h = new Height();
        System.out.println("Is tree heght  balanced: " + isTreeHeightBalanced_Helper(root, h));
    }

    public boolean checkTwoTreeAreMirror(TreeNode<Integer> root1, TreeNode<Integer> root2) {
        if (root1 == null && root2 == null) {
            return true;
        }

        if (root1 == null || root2 == null) {
            return false;
        }

        return root1.getData() == root2.getData()
                && checkTwoTreeAreMirror(root1.getLeft(), root2.getRight())
                && checkTwoTreeAreMirror(root1.getRight(), root2.getLeft());
    }

    private int convertTreeToSumTree_Sum(TreeNode<Integer> root) {

        if (root == null) {
            return 0;
        }

        int lSum = convertTreeToSumTree_Sum(root.getLeft());
        int rSum = convertTreeToSumTree_Sum(root.getRight());

        return lSum + rSum + root.getData();

    }

    public void convertTreeToSumTree(TreeNode<Integer> root) {

        //actual
        BinaryTree<Integer> bt = new BinaryTree<>(root);
        bt.treeBFS();

        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            TreeNode<Integer> t = q.poll();

            if (t.getLeft() != null) {
                q.add(t.getLeft());
            }

            if (t.getRight() != null) {
                q.add(t.getRight());
            }

            //leaf
            if (t.getLeft() == null && t.getRight() == null) {
                t.setData(0);
                continue;
            }

            // - t.getData() just don't include the value of that node itself
            t.setData(convertTreeToSumTree_Sum(t) - t.getData());

        }

        //output
        System.out.println();
        bt = new BinaryTree<>(root);
        bt.treeBFS();
        System.out.println();

    }

    private int convertTreeToSumTree_Recursion_Helper(TreeNode<Integer> root) {

        if (root == null) {
            return 0;
        }

        int data = root.getData();

        int lSum = convertTreeToSumTree_Recursion_Helper(root.getLeft());
        int rSum = convertTreeToSumTree_Recursion_Helper(root.getRight());

        //leaf
        if (root.getLeft() == null && root.getRight() == null) {
            root.setData(0);
            return data;
        } else {
            root.setData(lSum + rSum);
            return lSum + rSum + data;
        }

    }

    public void convertTreeToSumTree_Recursion(TreeNode<Integer> root) {

        //OPTIMISED
        //actual
        BinaryTree<Integer> bt = new BinaryTree<>(root);
        bt.treeBFS();

        convertTreeToSumTree_Recursion_Helper(root);

        //output
        System.out.println();
        bt = new BinaryTree<>(root);
        bt.treeBFS();
        System.out.println();

    }

    List<Integer> printKSumPathAnyNodeTopToDown_PathList;

    private void printKSumPathAnyNodeTopToDown_Helper(TreeNode<Integer> root, int K) {

        if (root == null) {
            return;
        }

        printKSumPathAnyNodeTopToDown_PathList.add(root.getData());

        printKSumPathAnyNodeTopToDown_Helper(root.getLeft(), K);
        printKSumPathAnyNodeTopToDown_Helper(root.getRight(), K);

        int pathSum = 0;
        for (int i = printKSumPathAnyNodeTopToDown_PathList.size() - 1; i >= 0; i--) {

            pathSum += printKSumPathAnyNodeTopToDown_PathList.get(i);
            if (pathSum == K) {
                //print actual nodes data
                for (int j = i; j < printKSumPathAnyNodeTopToDown_PathList.size(); j++) {
                    System.out.print(printKSumPathAnyNodeTopToDown_PathList.get(j) + " ");
                }
                System.out.println();
            }

        }

        //remove current node
        printKSumPathAnyNodeTopToDown_PathList.remove(printKSumPathAnyNodeTopToDown_PathList.size() - 1);

    }

    public void printKSumPathAnyNodeTopToDown(TreeNode<Integer> root, int K) {
        printKSumPathAnyNodeTopToDown_PathList = new ArrayList<>();
        printKSumPathAnyNodeTopToDown_Helper(root, K);
    }
    
    private TreeNode<Integer> lowestCommonAncestorOfTree_Helper(TreeNode<Integer> root, int N1, int N2){
        
        if(root == null){
            return null;
        }
        
        if(N1 == root.getData() || N2 == root.getData()){
            return root;
        }
        
        TreeNode<Integer> leftNode = lowestCommonAncestorOfTree_Helper(root.getLeft(), N1, N2);
        TreeNode<Integer> rightNode = lowestCommonAncestorOfTree_Helper(root.getRight(), N1, N2);
        
        if(leftNode != null && rightNode != null){
            return root;
        }
        
        return leftNode == null ? rightNode : leftNode;
        
    }
    
    public void lowestCommonAncestorOfTree(TreeNode<Integer> root, int N1, int N2){
        System.out.println("Lowest common ancestor of "+N1+" "+N2+": "+lowestCommonAncestorOfTree_Helper(root, N1, N2));
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
        System.out.println("Middle eleement of the stack: " + middleElementInStack_Element);
        //just reseting
        middleElementInStack_Element = Integer.MIN_VALUE;
    }

    public void nextSmallerElementInRightInArray(int[] a) {

        Stack<Integer> s = new Stack<>();
        List<Integer> result = new ArrayList<>();
        for (int i = a.length - 1; i >= 0; i--) {

            while (!s.isEmpty() && s.peek() > a[i]) {
                s.pop();
            }

            if (s.isEmpty()) {
                result.add(-1);
            } else {
                result.add(s.peek());
            }
            s.push(a[i]);
        }

        Collections.reverse(result);

        //output
        System.out.println("result: " + result);

    }

    private void reserveStack_Recursion_Insert(Stack<Integer> stack, int element) {

        if (stack.isEmpty()) {
            stack.push(element);
            return;
        }

        int popped = stack.pop();
        reserveStack_Recursion_Insert(stack, element);
        stack.push(popped);
    }

    private void reserveStack_Recursion(Stack<Integer> stack) {

        if (stack.isEmpty()) {
            return;
        }

        int popped = stack.pop();
        reserveStack_Recursion(stack);
        reserveStack_Recursion_Insert(stack, popped);

    }

    public void reverseStack(Stack<Integer> stack) {
        System.out.println("actual: " + stack);
        reserveStack_Recursion(stack);
        System.out.println("output: " + stack);
    }

    public void minCostOfRope(int[] a) {

        //GREEDY ALGO
        //HEAP based approach
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int x : a) {
            minHeap.add(x);
        }

        //calculations
        int cost = 0;
        while (minHeap.size() >= 2) {

            int rope1 = minHeap.poll();
            int rope2 = minHeap.poll();

            cost += rope1 + rope2;
            int newRope = rope1 + rope2;
            minHeap.add(newRope);

        }

        //output
        System.out.println("Min cost to combine all rpes into one rope: " + cost);

    }

    public void majorityElement_1(int[] a) {

        //.............T: O(N)
        //.............S: O(Unique ele in a)
        int maj = a.length / 2;

        Map<Integer, Integer> map = new HashMap<>();
        for (int x : a) {
            map.put(x, map.getOrDefault(x, 0) + 1);
        }

        for (Map.Entry<Integer, Integer> e : map.entrySet()) {
            if (e.getValue() > maj) {
                System.out.println("Majority element: " + e.getKey());
                return;
            }
        }

        System.out.println("Majority element: -1");

    }

    public void majorityElement_2(int[] a) {

        //Moore’s Voting Algorithm
        //https://www.geeksforgeeks.org/majority-element/
        //..........T: O(N)
        //..........S: O(1)
        //finding candidate
        int maj_index = 0, count = 1;
        int i;
        for (i = 1; i < a.length; i++) {
            if (a[maj_index] == a[i]) {
                count++;
            } else {
                count--;
            }
            if (count == 0) {
                maj_index = i;
                count = 1;
            }
        }
        int cand = a[maj_index];

        //validating the cand 
        count = 0;
        for (i = 0; i < a.length; i++) {
            if (a[i] == cand) {
                count++;
            }
        }
        if (count > a.length / 2) {
            System.out.println("Majority element: " + cand);
        } else {
            System.out.println("Majority element: -1");
        }

    }

    public void mergeTwoSortedArraysWithoutExtraSpace(int[] arr1, int[] arr2, int m, int n) {

        //https://www.geeksforgeeks.org/merge-two-sorted-arrays-o1-extra-space/
        // Iterate through all elements of ar2[] starting from 
        // the last element 
        for (int i = n - 1; i >= 0; i--) {
            /* Find the smallest element greater than ar2[i]. Move all 
             elements one position ahead till the smallest greater 
             element is not found */
            int j, last = arr1[m - 1];
            for (j = m - 2; j >= 0 && arr1[j] > arr2[i]; j--) {
                arr1[j + 1] = arr1[j];
            }

            // If there was a greater element 
            if (j != m - 2 || last > arr2[i]) {
                arr1[j + 1] = arr2[i];
                arr2[i] = last;
            }
        }

        //output
        for (int x : arr1) {
            System.out.print(x + " ");
        }
        System.out.println();
        for (int x : arr2) {
            System.out.print(x + " ");
        }

        System.out.println();

    }

    private int[] KMP_PatternMatching_Algorithm_LPSArray(String pattern, int size) {

        int[] lps = new int[size];
        lps[0] = 0; //always 0th index is 0
        int i = 1;
        int len = 0;
        while (i < size) {

            if (pattern.charAt(i) == pattern.charAt(len)) {
                len++;
                lps[i] = len;
                i++;
            } else {
                // char doesn't match
                // This is tricky. Consider the example. 
                // AAACAAAA and i = 7. The idea is similar 
                // to search step. 
                if (len != 0) {
                    len = lps[len - 1];

                    // Also, note that we do not increment 
                    // i here 
                } else {
                    lps[i] = len;
                    i++;
                }
            }

        }

        return lps;

    }

    public void KMP_PatternMatching_Algorithm(String text, String pattern) {

        //geeksforgeeks.org/kmp-algorithm-for-pattern-searching/
        int M = text.length();
        int N = pattern.length();

        //create LPS array for pattern
        int[] lps = KMP_PatternMatching_Algorithm_LPSArray(pattern, N);

        //text and pattern matching
        int i = 0; // index for text
        int j = 0; // index for pattern
        while (i < M) {

            if (text.charAt(i) == pattern.charAt(j)) {
                i++;
                j++;
            }

            //j reached the length of pattern
            if (j == N) {
                System.out.println("Pattern matched at: " + (i - j));
                j = lps[j - 1];
            } else if (i < M && text.charAt(i) != pattern.charAt(j)) {
                // Do not match lps[0..lps[j-1]] characters, 
                // they will match anyway 
                if (j != 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }

        }

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
//        obj.romanStringToDecimal("XI");
//        obj.romanStringToDecimal("IX");
//        obj.romanStringToDecimal("IV");
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
//        System.out.println("Convert tree to doubly linked list");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.treeToDoublyLinkedList(root1);
//        root1 = new TreeNode<>(20);
//        root1.setLeft(new TreeNode(8));
//        root1.getLeft().setLeft(new TreeNode(5));
//        root1.getLeft().setRight(new TreeNode(3));
//        root1.getLeft().getRight().setLeft(new TreeNode(10));
//        root1.setRight(new TreeNode(22));
//        root1.getRight().setLeft(new TreeNode(4));
//        root1.getRight().setRight(new TreeNode(25));
//        root1.getRight().getLeft().setRight(new TreeNode(14));
//        obj.treeToDoublyLinkedList(root1);
        //......................................................................
//        Row: 199
//        System.out.println("Check if all the leaf nodes of tree are at same level");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.checkIfAllLeafNodeOfTreeAtSameLevel(root1);
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode(2));
//        root1.setRight(new TreeNode(3));
//        obj.checkIfAllLeafNodeOfTreeAtSameLevel(root1);
        //......................................................................
//        Row: 216
//        System.out.println("Min & max in the BST");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.minAndMaxInBST(root1);
        //......................................................................
//        Row: 218
//        System.out.println("Check if a tree is BST");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.isTreeBST(root1);
//        root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(10)); //BST break cond.
//        root1.getRight().setRight(new TreeNode(9));
//        obj.isTreeBST(root1);
        //......................................................................
//        Row: 225
//        System.out.println("Kth largest node in the BST");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.kTHLargestNodeInBST(root1, 4);
        //......................................................................
//        Row: 226
//        System.out.println("Kth smallest node in the BST");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.kTHSmallestNodeInBST(root1, 4);
        //......................................................................
//        Row: 169
//        System.out.println("Merge K sorted linked lists");
//        Node<Integer> n1 = new Node<>(1);
//        n1.setNext(new Node<>(2));
//        n1.getNext().setNext(new Node<>(3));
//        Node<Integer> n2 = new Node<>(4);
//        n2.setNext(new Node<>(10));
//        n2.getNext().setNext(new Node<>(15));
//        Node<Integer> n3 = new Node<>(3);
//        n3.setNext(new Node<>(9));
//        n3.getNext().setNext(new Node<>(27));
//        int K = 3;
//        Node<Integer>[] nodes = new Node[K];
//        nodes[0] = n1;
//        nodes[1] = n2;
//        nodes[2] = n3;
//        obj.mergeKSortedLinkedList(nodes);
        //......................................................................
//        Row: 173
//        System.out.println("Print the Kth node from the end of a linked list 3 approaches");
//        //https://www.geeksforgeeks.org/nth-node-from-the-end-of-a-linked-list/
//        Node<Integer> n1 = new Node<>(1);
//        n1.setNext(new Node<>(2));
//        n1.getNext().setNext(new Node<>(3));
//        n1.getNext().getNext().setNext(new Node<>(5));
//        n1.getNext().getNext().getNext().setNext(new Node<>(9));
//        n1.getNext().getNext().getNext().getNext().setNext(new Node<>(15));
//        obj.kThNodeFromEndOfLinkedList_1(n1, 3);
//        obj.kThNodeFromEndOfLinkedList_2(n1, 3);
//        obj.kThNodeFromEndOfLinkedList_3(n1, 3); //OPTIMISED O(N)
        //......................................................................
//        Row: 190
//        System.out.println("Check if a tree is height balanced or not");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.isTreeHeightBalanced(root1);
//        root1 = new TreeNode<>(1); //SKEWED TREE
//        root1.setLeft(new TreeNode(10));
//        root1.getLeft().setLeft(new TreeNode(15));
//        obj.isTreeHeightBalanced(root1);
        //......................................................................
//        Row: 201
//        System.out.println("Check if 2 trees are mirror or not");
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(3));
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.setRight(new TreeNode<>(2));
//        System.out.println("2 tree are mirror: "+obj.checkTwoTreeAreMirror(root1, root2));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(3));
//        root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(2)); //SAME 
//        root2.setRight(new TreeNode<>(3)); //SAME
//        System.out.println("2 tree are mirror: "+obj.checkTwoTreeAreMirror(root1, root2));
        //......................................................................
//        Row: 333
//        System.out.println("Next smaller element to right in array");
//        obj.nextSmallerElementInRightInArray(new int[]{4, 8, 5, 2, 25});
        //......................................................................
//        Row: 309
//        System.out.println("Reverse a stack using recursion");
//        Stack<Integer> stack = new Stack<>();
//        stack.addAll(Arrays.asList(1, 2, 3, 4, 5));
//        obj.reverseStack(stack);
        //......................................................................
//        Row: 7
//        System.out.println("Min & max in array");
//        obj.minMaxInArray_1(new int[]{1000, 11, 445, 1, 330, 3000});
//        obj.minMaxInArray_2(new int[]{1000, 11, 445, 1, 330, 3000});
        //......................................................................
//        Row: 8
//        System.out.println("Kth smallest and largest element in array");
//        obj.kThSmallestElementInArray(new int[]{7, 10, 4, 3, 20, 15}, 3);
//        obj.kThLargestElementInArray(new int[]{7, 10, 4, 3, 20, 15}, 3);
        //......................................................................
//        Row: 9
//        System.out.println("Sort the array containing elements 0, 1, 2");
//        obj.sortArrayOf012_1(new int[]{0, 1, 1, 0, 1, 2, 1, 2, 0, 0, 0, 1});
//        obj.sortArrayOf012_2(new int[]{0, 1, 1, 0, 1, 2, 1, 2, 0, 0, 0, 1}); //DUTCH NATIONAL FLAG ALGO
        //......................................................................
//        Row: 51
//        System.out.println("Rotate a matrix 90 degrees");
//        int[][] mat = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//        obj.rotateMatrixClockWise90Deg(mat);
        //......................................................................
//        Row: 62
//        System.out.println("Count and say");
//        obj.countAndSay(1);
//        obj.countAndSay(2);
//        obj.countAndSay(3);
//        obj.countAndSay(10);
        //......................................................................
//        Row: 93
//        System.out.println("Remove consecutive duplicate char in string");
//        obj.removeConsecutiveDuplicateInString("aababbccd");
//        obj.removeConsecutiveDuplicateInString("aaabbbcccbbbbaaaa");
//        obj.removeConsecutiveDuplicateInString("xyzpqrs");
//        obj.removeConsecutiveDuplicateInString("abcppqrspplmn");
//        obj.removeConsecutiveDuplicateInString("abcdlllllmmmmm");
        //......................................................................
//        Row: 108
//        System.out.println("Majority Element");
//        obj.majorityElement_1(new int[] { 1, 3, 3, 1, 2 });
//        obj.majorityElement_1(new int[] { 1, 3, 3, 3, 2 });
//        obj.majorityElement_2(new int[] { 1, 3, 3, 1, 2 }); //MOORE'S VOTING ALGO
//        obj.majorityElement_2(new int[] { 1, 3, 3, 3, 2 }); //MOORE'S VOTING ALGO
        //......................................................................
//        Row: 195
//        System.out.println("Convert tree to its sun tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.convertTreeToSumTree(root1); //EXTRA QUEUE SPACE IS USED
//        //reset root
//        root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.convertTreeToSumTree_Recursion(root1); //NO EXTRA QUEUE SPACE IS USED - OPTIMISED
        //......................................................................
//        Row: 206
//        System.out.println("K sum path from any node top to down");
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode(3));
//        root1.getLeft().setLeft(new TreeNode(2));
//        root1.getLeft().setRight(new TreeNode(1));
//        root1.getLeft().getRight().setLeft(new TreeNode(1));
//        root1.setRight(new TreeNode(-1));
//        root1.getRight().setLeft(new TreeNode(4));
//        root1.getRight().getLeft().setLeft(new TreeNode(1));
//        root1.getRight().getLeft().setRight(new TreeNode(2));
//        root1.getRight().setRight(new TreeNode(5));
//        root1.getRight().getRight().setRight(new TreeNode(6));
//        obj.printKSumPathAnyNodeTopToDown(root1, 5);
        //......................................................................
//        Row: 349
//        System.out.println("Min cost to combine ropes of diff lengths into one big rope");
//        obj.minCostOfRope(new int[]{4, 3, 2, 6});
        //......................................................................
//        Row: 344
//        System.out.println("Reorganise string");
//        //https://leetcode.com/problems/reorganize-string/
//        System.out.println("Reorganise string output: "+obj.reorganizeString("aab"));
//        System.out.println("Reorganise string output: "+obj.reorganizeString("aaab"));
        //......................................................................
//        Row: 98
//        System.out.println("Print all sentences that can be formed from list/array of words");
//        String[][] arr = {{"you", "we", ""},
//        {"have", "are", ""},
//        {"sleep", "eat", "drink"}};
//        obj.printSentencesFromCollectionOfWords(arr); //GRAPH LIKE DFS
        //......................................................................
//        Row: 74
//        System.out.println("KMP pattern matching algo");
//        String txt = "ABABDABACDABABCABAB";
//        String pat = "ABABCABAB";
//        obj.KMP_PatternMatching_Algorithm(txt, pat);
//        txt = "sangeeangt";
//        pat = "ang";
//        obj.KMP_PatternMatching_Algorithm(txt, pat);
//        obj.longestPrefixAlsoSuffixInString_KMPAlgo("abab");
//        obj.longestPrefixAlsoSuffixInString_KMPAlgo("aaaa");
//        obj.longestPrefixAlsoSuffixInString_KMPAlgo("aabcavefaabca"); //FAIL CASE
//        obj.longestPrefixAlsoSuffixInString_KMPAlgo("abcdef");
        //......................................................................
//        Row: 82
//        System.out.println("Longest common prefix in list of strings");
//        obj.longestCommonPrefix(new String[]{"flower", "flow", "flight"});
//        obj.longestCommonPrefix(new String[]{"dog", "racecar", "car"});
//        obj.longestCommonPrefix(new String[]{"a"});
//        obj.longestCommonPrefix(new String[]{"abc", "abcdef", "abcdlmno"});
        //......................................................................
//        Row: 114
//        System.out.println("Merge 2 sorted arrays without using extra space");
//        int arr1[] = new int[]{1, 5, 9, 10, 15, 20};
//        int arr2[] = new int[]{2, 3, 8, 13};
//        obj.mergeTwoSortedArraysWithoutExtraSpace(arr1, arr2, arr1.length, arr2.length);
        //......................................................................
//        Row: 140
//        System.out.println("Reverse a linked list in K groups");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        LinkedListUtil<Integer> ll = new LinkedListUtil<>(obj.reverseLinkedListInKGroups(node, 2));
//        ll.print();
//        node = new Node<>(3);
//        node.setNext(new Node<>(8));
//        node.getNext().setNext(new Node<>(7));
//        node.getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        ll = new LinkedListUtil<>(obj.reverseLinkedListInKGroups(node, 4));
//        ll.print();
        //......................................................................
//        Row: 207, 220
        System.out.println("Lowest common ancestor of two given node/ node values for binary tree and binary search tree both");
        TreeNode<Integer> root1 = new TreeNode<>(5);
        root1.setLeft(new TreeNode(2));
        root1.getLeft().setLeft(new TreeNode<>(3));
        root1.getLeft().setRight(new TreeNode<>(4));
        obj.lowestCommonAncestorOfTree(root1, 3, 4);
        root1 = new TreeNode<>(5);
        root1.setLeft(new TreeNode(2));
        root1.getLeft().setLeft(new TreeNode<>(3));
        root1.getLeft().setRight(new TreeNode<>(4));
        root1.setRight(new TreeNode<>(6));
        obj.lowestCommonAncestorOfTree(root1, 3, 6);
        //CASE OF BST
        root1 = new TreeNode<>(6);
        root1.setLeft(new TreeNode(2));
        root1.getLeft().setLeft(new TreeNode(0));
        root1.getLeft().setRight(new TreeNode(4));
        root1.getLeft().getRight().setLeft(new TreeNode(3));
        root1.getLeft().getRight().setRight(new TreeNode(5));
        root1.setRight(new TreeNode(8));
        root1.getRight().setLeft(new TreeNode(7));
        root1.getRight().setRight(new TreeNode(9));
        obj.lowestCommonAncestorOfTree(root1, 0, 5);
        root1 = new TreeNode<>(5);
        root1.setLeft(new TreeNode(4));
        root1.getLeft().setLeft(new TreeNode<>(3));
        root1.setRight(new TreeNode(6));
        root1.getRight().setRight(new TreeNode(7));
        root1.getRight().getRight().setRight(new TreeNode(8));
        obj.lowestCommonAncestorOfTree(root1, 7, 8);
    }

}
