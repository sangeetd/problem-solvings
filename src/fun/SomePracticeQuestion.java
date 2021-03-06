/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import com.sun.xml.internal.bind.v2.runtime.Coordinator;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Scanner;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import javafx.util.Pair;

/**
 *
 * @author RAVI
 */
public class SomePracticeQuestion {

    public static int[] countDistinctOnwWindowK(int[] arr, int k) {

        //https://www.geeksforgeeks.org/count-distinct-elements-in-every-window-of-size-k/
        int n = arr.length;
        Set<Integer> s = new HashSet<>();
        List<Integer> l = new ArrayList<>();
        for (int i = 0; i < n - k + 1; i++) {
            for (int j = i; j <= k - 1 + i; j++) {
                s.add(arr[j]);
            }

            l.add(s.size());
            s.clear();

        }

        int[] result = new int[l.size()];
        for (int i = 0; i < l.size(); i++) {
            result[i] = l.get(i);
        }

        return result;

    }

    public static List<String> subsetOfString(String str) {

        List<String> subset = new ArrayList<>();
        for (int i = 0; i < str.length(); i++) {
            for (int j = i; j < str.length(); j++) {
                subset.add(str.substring(i, j + 1));
            }
        }
        return subset;
    }

    public static List<Integer> subsetOfInteger(Integer n) {
        List<String> s = subsetOfString(String.valueOf(n));
        List<Integer> i = s.stream()
                .map(x -> Integer.parseInt(x))
                .collect(Collectors.toList());

        return i;
    }

    public static boolean isSubsequence(String str, String seq, int strLen, int seqLen) {

        if (strLen == 0 && seqLen != 0) {
            return false;
        }

        if (seqLen == 0) {
            return true;
        }

        if (str.charAt(strLen - 1) == seq.charAt(seqLen - 1)) {
            return isSubsequence(str, seq, strLen - 1, seqLen - 1);
        }

        return isSubsequence(str, seq, strLen - 1, seqLen);
    }

    public static int factorial(int n) {
        if (n == 0 || n == 1) {
            return 1;
        }

        return n * factorial(n - 1);

    }

    public static int arrangeSetOfNumberInKGroup(int[] set, int group) {

        int n = set.length;
        int r = group;

        //nCr = n!/r!*(n-r)!
        int result = factorial(n) / (factorial(r) * factorial(n - r));
        return result;
    }

    public static boolean stringCompare(String S, String T) {

        for (int i = 0; i < S.length(); i++) {

            if (!S.equals("") && S.charAt(0) == '#') {
                S = S.replaceFirst("#", "");
                i = 0;
                continue;
            }

            if (S.charAt(i) == '#') {
                String ss = S.substring(i - 1, i + 1);
                S = S.replaceAll(ss, "");
                i = 0;
            }

        }

        for (int i = 0; i < T.length(); i++) {

            if (!T.equals("") && T.charAt(0) == '#') {
                T = T.replaceFirst("#", "");
                i = 0;
                continue;
            }

            if (T.charAt(i) == '#') {
                String ss = T.substring(i - 1, i + 1);
                T = T.replaceAll(ss, "");
                i = 0;
            }

        }

        return S.equals(T);
    }

    public static int recursionCheck(int n) {

        System.out.println(n);
        if (n == 0 || n == 1) {

            return 1;
        }

        return n * recursionCheck(n - 1);

    }

    public static int simpleBinarySearch(int[] a, int key) {

        int f = 0;
        int l = a.length - 1;

        while (f < l) {

            //this mid calculation works normal
            //int mid = (f+l)/2;
            //this mid calculation can handle normal as well as large length array 10^9
            //without creating issues
            int mid = f + ((l - f) / 2);

            if (a[mid] == key) {
                return mid;
            } else if (key < a[mid]) {
                l = mid - 1;
            } else {
                f = mid + 1;
            }

        }

        return -1;

    }

    public static boolean isStringPallindrome(String str) {

//        //...........brute force.............
//        for (int i = 0; i < str.length() / 2; i++) {
//            
//            if (str.charAt(i) != str.charAt(str.length() - 1 - i)) {
//                System.out.println("not pallindrome");
//                break;
//            }
//            
//        }
//        //reverse a string without STL
//        String s = str;
//        int start=0;
//        int end=s.length() - 1;
//        char[] ch = s.toCharArray();
//        while(start < end){
//            
//            char temp = ch[start];
//            ch[start] = ch[end];
//            ch[end] = temp;
//            
//            start++;
//            end--;
//            
//        }
//        
//        System.out.println(String.valueOf(ch));
//        //...........brute force.............
        StringBuilder sb = new StringBuilder(str);
        if (str.equals(sb.reverse().toString())) {
            return true;
        }
        sb = null;
        return false;
    }

    public static String longestPallindromicSubsetFromAString(String str) {

        List<String> l = subsetOfString(str);
        String longestPallindromicString = "";
        for (String s : l) {
            if (s.length() > 1) {
                if (isStringPallindrome(s) && s.length() > longestPallindromicString.length()) {
                    longestPallindromicString = s;
                }
            }
        }
        return longestPallindromicString;
    }

    public static int longestPanllindromicSubsequenceFormAString(String str, int f, int l) {

        if (f > l) {
            //f>l means half of the string has traversed
            //so return 0
            return 0;
        }

        if (f == l) {
            //f==l means 1 single char in string pointed by f and l both
            //and 1 single char is in itself a pallindrome so 1
            return 1;
        }

        int c1 = 0;
        if (str.charAt(f) == str.charAt(l)) {
            //both char at loc f and l are same
            //it means let say char be E and E. EE as subseq is pallindrome of len = 2
            //so when such char matches len of 2 subseq is already found
            //thats 2 is added to below recursion
            c1 = 2 + longestPanllindromicSubsequenceFormAString(str, f + 1, l - 1);
            return c1;
        }

        //probablity when no char matches at f and l loc
        //1. next subseq can be found from f--(l-1) string part
        //2. mext sunseq found from (f+1)--l string part
        int c2 = 0 + longestPanllindromicSubsequenceFormAString(str, f, l - 1);
        int c3 = 0 + longestPanllindromicSubsequenceFormAString(str, f + 1, l);

        //max of all the count is returned as longest subseq is req
        return Math.max(Math.max(c1, c2), c3);
    }

    public static int[] sumOfArrayAsNumberForm1(int[] a, int[] b) {
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        for (int i = 0; i < a.length; i++) {
            sb1.append(a[i]);
        }

        for (int i = 0; i < b.length; i++) {
            sb2.append(b[i]);
        }

        long num1 = Long.parseLong(sb1.toString());
        long num2 = Long.parseLong(sb2.toString());

        long sum = num1 + num2;

        String sumToStr = String.valueOf(sum);

        int[] sumToArrForm = new int[sumToStr.length()];
        for (int j = 0; j < sumToStr.length(); j++) {
            sumToArrForm[j] = Character.getNumericValue(sumToStr.charAt(j));
        }

        return sumToArrForm;

    }

    public static int[] sumOfArrayAsNumberForm2(int[] a, int[] b) {

        //considering a as num is >= b as num
        //ex: a+b=548+62
        //start from both array's last index until they reach their 0 index
        int[] sum;
        int n = 0;

        //whose arrays is longest n will be that len+1
        //bcoz sum[n] can take that much only
        //ex: a+b=900+900= when i=j=0 = 9+9 at max can give 18
        //total =  1800 = 4 digit that is a.len or b.len + 1
        n = (a.length >= b.length) ? a.length + 1 : b.length + 1;

        sum = new int[n];
        int i = a.length - 1;
        int j = b.length - 1;
        int k = n - 1;
        int carry = 0;
        while (true) {

            if (i < 0) {
                sum[k] = carry;
                break;
            }

            int x = a[i];
            int y = 0;

            if (j >= 0) {
                y = b[j];
            }

            //548+62
            //n=(a.len+1) k = n-1 = 3 (0 -- 3)
            //carry at start of loop is 0
            //itr 1: x+y+carry=8+2+0=10 -> 1 and 0 is broken as carry and sum[3] 
            //itr 2: x+y+carry=4+6+1=11 -> 1 and 1 as carry and sum[2]
            //itr 3: since b[j] is consumes and j < 0: y = 0 and x=a[i]=5 as i>=0
            //x+y+carry=5+0+1=6 -> 0 and 6 as carry and sum[1]
            //itr 4: now i < 0 -> sum[0]=carry and break;
            int s = x + y + carry;
            carry = s / 10;
            sum[k] = s % 10;

            i--;
            j--;
            k--;

        }

        return sum;
    }

    public static void addBinaryStrings(String a, String b) {

        StringBuilder sb = new StringBuilder();
        int i = a.length() - 1;
        int j = b.length() - 1;
        int carry = 0;

        while (i >= 0 || j >= 0) {

            int sum = carry;

            if (i >= 0) {
                sum += a.charAt(i--) - '0';
            }
            if (j >= 0) {
                sum += b.charAt(j--) - '0';
            }

            //in binary addtion 1+1 = 10 carry 1 | 10%2 = 0 and 10/2 = 1
            //1+0 = 1  or 0+1 = 1 carry = 0 | 1%2 = 1 and 1/2 = 0
            sb.insert(0, sum % 2);
            carry = sum / 2;

        }

        if (carry > 0) {
            sb.insert(0, 1);
        }

        System.out.println("Binary string result " + sb.toString());

    }

    public static void addTwoNumbersAsStrings(String a, String b) {

        StringBuilder sb = new StringBuilder();
        int i = a.length() - 1;
        int j = b.length() - 1;
        int carry = 0;

        while (i >= 0 || j >= 0) {

            int sum = carry;

            if (i >= 0) {
                sum += a.charAt(i--) - '0';
            }
            if (j >= 0) {
                sum += b.charAt(j--) - '0';
            }

            sb.insert(0, sum % 10);
            carry = sum / 10;

        }

        if (carry > 0) {
            sb.insert(0, carry);
        }

        System.out.println("Sum string result " + sb.toString());

    }

    public static void addDigits(int num) {

        /*
        
         Explanation:
        
         https://leetcode.com/problems/add-digits/solution/
        
         Approach 1: Mathematical: Digital Root
        
         */
        if (num == 0) {
            System.out.println(0);
            return;
        }
        if (num % 9 == 0) {
            System.out.println(9);
            return;
        }

        System.out.println(num % 9);

    }

    public static int fibbonacci(int n) {

        if (n <= 0 || n == 1) {
            return 0;
        } else if (n == 2) {
            return 1;
        }

        return fibbonacci(n - 1) + fibbonacci(n - 2);

    }

    public static int numberFactor(int n) {

        if (n <= 0 || n == 1 || n == 2) {
            return 1;
        }
        if (n == 3) {
            return 2;
        }

        return numberFactor(n - 1) + numberFactor(n - 3) + numberFactor(n - 4);

    }

    public static void numberToWordsConverter(long num) {

        Map<Long, String> numVocab = new HashMap<>();
        numVocab.put(null, "");
        numVocab.put(0l, "zero");//single digit
        numVocab.put(1l, "one");
        numVocab.put(2l, "two");
        numVocab.put(3l, "three");
        numVocab.put(4l, "four");
        numVocab.put(5l, "five");
        numVocab.put(6l, "six");
        numVocab.put(7l, "seven");
        numVocab.put(8l, "eight");
        numVocab.put(9l, "nine");//single digit
        numVocab.put(10l, "ten");//2 gigit
        numVocab.put(11l, "eleven");
        numVocab.put(12l, "twelve");
        numVocab.put(13l, "thirteen");
        numVocab.put(14l, "fourteen");
        numVocab.put(15l, "fifteen");
        numVocab.put(16l, "sixteen");
        numVocab.put(17l, "seventeen");
        numVocab.put(18l, "eighteen");
        numVocab.put(19l, "ninteen");//upto 19
        numVocab.put(20l, "twenty");//2 digit from 20
        numVocab.put(30l, "thirty");
        numVocab.put(40l, "fourty");
        numVocab.put(50l, "fifty");
        numVocab.put(60l, "sixty");
        numVocab.put(70l, "seventy");
        numVocab.put(80l, "eighty");
        numVocab.put(90l, "ninty");// upto 90
        numVocab.put(100l, "hundred");
        numVocab.put(1000l, "thousand");

        String strNum = String.valueOf(num);

        if (strNum.length() == 1) {
            System.out.println(numVocab.get(num));
            return;
        }

        Long[] n = new Long[strNum.length() + strNum.length() - 1];
        int pow = strNum.length() - 1;
        int k = 0;

        for (int i = 0; i < strNum.length(); i++) {
            long tenPow = (long) Math.pow(10, pow);
            long intChar = Character.getNumericValue(strNum.charAt(i));

            n[k] = intChar;
            if (tenPow > 1) {
                n[k + 1] = tenPow;
            }

            k += 2;
            pow--;

        }

        StringBuilder sb = new StringBuilder();
        for (int j = 0; j < n.length - 3; j++) {
            //System.out.println(n[j]);
            sb.append(numVocab.get(n[j]));
            sb.append(" ");
        }

        //process for last 2 digit
        int p = n.length;
        int lastTwoDigit = (int) ((n[p - 3] * n[p - 2]) + n[p - 1]);
        long x = n[p - 3] * n[p - 2];
        if (lastTwoDigit > 20) {
            sb.append(numVocab.get(x));
            sb.append(" ");
            sb.append(numVocab.get(n[p - 1]));
        } else {
            sb.append(numVocab.get((long) lastTwoDigit == 0 ? null : (long) lastTwoDigit));
        }

        System.out.println(sb.toString());

    }

    public static void medianRecurrenceInAStream(int[] stream) {

        int n = stream.length;
        int[] medianKeeper = new int[n];
        medianKeeper[0] = (int) Math.floor((double) stream[0] / 1.0);
        for (int i = 1; i < n; i++) {

            medianKeeper[i] = (int) Math.floor((medianKeeper[i - 1] + (double) stream[i]) / 2.0);
        }

        for (int j : medianKeeper) {
            System.out.println(j);
        }

    }

    private static void swap(int[] a, int x, int y) {
        //swap posi and posj
        int temp = a[x];
        a[x] = a[y];
        a[y] = temp;
    }

    private static void printNextPemutation(int[] a) {
        for (int m : a) {
            System.out.print(m + " ");
        }
        System.out.println();
    }

    public static int[] nextPermutation(int[] a) {

        //if array is in desc order then not possible
        int max = 0;
        boolean isSorted = true;
        for (int i = 0; i < a.length; i++) {
            max = a[i];
            for (int j = i + 1; j < a.length; j++) {
                if (a[j] > max) {
                    max = a[j];
                    isSorted = false;
                }
            }

            if (!isSorted) {
                break;
            }

        }

        if (isSorted) {
            //if the array is already sorted
            System.out.println("Not Possible");
            printNextPemutation(a);
            return a;
        }

        int posi = -1;
        int posj = -1;
        for (int i = a.length - 1; i >= 0; i--) {
            for (int j = a.length - 1; j >= 0; j--) {

                if (a[i] > a[j]) {
                    posi = i;
                    posj = j;
                    break;
                }

            }

            if (posi != -1 && posj != -1) {
                break;
            }

        }

        //swap posi and posj
        swap(a, posi, posj);

        //sort from posj to a.len
        int min;
        int posl = -1;
        int posk = -1;
        for (int k = posj + 1; k < a.length; k++) {
            min = a[k];
            for (int l = k + 1; l < a.length; l++) {
                if (a[l] < min) {
                    min = a[l];
                    posk = k;
                    posl = l;
                }
            }
            //swap 
            if (posk != -1 && posl != -1) {
                swap(a, posk, posl);
            }

        }

        printNextPemutation(a);

        return a;
    }

    private static void reverse(int[] nums, int start) {
        int i = start, j = nums.length - 1;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    public static void nextPermutation_2(int[] a) {
        int i = a.length - 2;
        while (i >= 0 && a[i + 1] <= a[i]) {
            i--;
        }
        if (i >= 0) {
            int j = a.length - 1;
            while (j >= 0 && a[j] <= a[i]) {
                j--;
            }
            swap(a, i, j);
        }
        reverse(a, i + 1);

        printNextPemutation(a);
    }

    public static void noOfPlatformNeeded(int[] arrival, int[] departutre) {

        //greedy algo based
        //sort inc order both depart. and arrival
        Arrays.sort(arrival);
        Arrays.sort(departutre);

        printNextPemutation(arrival);
        printNextPemutation(departutre);

        int minPlatformNeeded = 1;
        int result = 0;
        int i = 0;
        int j = 0;

        while (i < arrival.length && j < departutre.length) {

            //System.out.println("Event: "+arrival[i]+" "+departutre[j]+" platform req: "+result);
            if (arrival[i] > departutre[j]) {
                //single same platform can handle
                result--;
                j++;
            } else if (arrival[i] <= departutre[j]) {
                result++;
                i++;
            }

            if (minPlatformNeeded < result) {
                minPlatformNeeded = result;
            }

        }

        System.out.println("min platform needed " + minPlatformNeeded);

    }

    public static void longestSubsequenceWithAltVowels(String str) {

        int vowFlag = 0;
        int consFlag = 0;
        char ch = '0';
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < str.length(); i++) {

            ch = str.charAt(i);

            if ((ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u')
                    && vowFlag == 0) {

                sb.append(ch);
                vowFlag = 1;
                consFlag = 0;
            } else if (!(ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u')
                    && consFlag == 0) {
                sb.append(ch);
                vowFlag = 0;
                consFlag = 1;

            }
        }

        System.out.println("longest subsequence with alternating vowel : " + sb.toString());

    }

    public static void addWithoutPlus(int a, int b) {

        while (b != 0) {

            int carry = a & b;
            System.out.println("carry " + carry);

            a = a ^ b;
            System.out.println("a " + a);

            b = carry << 1;
            System.out.println("b " + b);

        }

        System.out.println("ans " + a);

    }

    public static void petrolCycle(int[] petrolUnit, int[] distanceUnit) {

        boolean isMarked = false;
        int startCycleIndex = -1;
        int n = petrolUnit.length;
        int i = 0;
        int j = 0;
        int k = 0;
        long petrolBal = 0;

        while (k < n) {

            long petrolTaken = petrolBal + petrolUnit[i];

            if (petrolTaken < distanceUnit[j]) {

                petrolBal = (petrolTaken - distanceUnit[j] > 0) ? petrolTaken - distanceUnit[j] : 0;
                isMarked = false;
                startCycleIndex = -1;
            } else if (petrolTaken >= distanceUnit[j]) {

                petrolBal = (petrolTaken - distanceUnit[j] > 0) ? petrolTaken - distanceUnit[j] : 0;
                if (petrolTaken - distanceUnit[j] > 0 && isMarked != true) {
                    isMarked = true;
                    startCycleIndex = i;
                }

            }

            i++;
            j++;
            k++;

        }

        //from startCycleIndex = n and n = startCycleIndex should form cycle
        boolean confirmed = false;
        long checkPetrolbal = petrolBal;
        i = j = 0;
        while (i <= startCycleIndex) {

            long petrolTaken = checkPetrolbal + petrolUnit[i];

            if (petrolTaken < distanceUnit[j]) {

                checkPetrolbal = (petrolTaken - distanceUnit[j] > 0) ? petrolTaken - distanceUnit[j] : 0;
                confirmed = false;
                break;
            } else if (petrolTaken >= distanceUnit[j]) {

                checkPetrolbal = (petrolTaken - distanceUnit[j] > 0) ? petrolTaken - distanceUnit[j] : 0;
                confirmed = true;

            }

            i++;
            j++;

        }

        if (startCycleIndex == -1 || confirmed == false) {
            System.out.println("No cycle possible");
        } else {
            System.out.println("from: " + (startCycleIndex + 1));
        }

    }

    public static void majorityElement(int[] a) {

        int thanThis = (int) Math.floor((double) a.length / 2.0);
        Map<Integer, Integer> map = new HashMap<>();

        for (int m : a) {
            map.put(m, map.getOrDefault(m, 0) + 1);
        }

        map.entrySet().stream().forEach(e -> {
            if (e.getValue() > thanThis) {
                System.out.print(e.getKey() + " ");
            }

        });
        System.out.println();

    }

    public static void longestPrefixAlsoSuffixInString(String str) {

        int n = str.length();
        StringBuilder sb = new StringBuilder();

        for (int i = 1; i < n; i++) {

            sb.append(str.substring(0, i));
            if (str.substring(i, n).contains(sb.toString())) {
                //continue to check more
                sb.setLength(0);
            } else {
                break;
            }

        }
        String result = sb.toString().substring(0, sb.toString().length() - 1);
        if (result.equals("")) {
            System.out.println("not possible");
        } else {
            System.out.println("possible " + result + " " + result.length());
        }

    }

    public static void nextLargestNoToRight(int[] a) {

        int n = a.length;
        Stack<Integer> s = new Stack<>();
        List<Integer> l = new ArrayList<>();

        for (int i = n - 1; i >= 0; i--) {

            while (!s.isEmpty() && s.peek() < a[i]) {
                s.pop();
            }

            if (s.isEmpty()) {
                l.add(-1);
            } else {
                l.add(s.peek());
            }

            s.push(a[i]);

        }

        Collections.reverse(l);

        for (int m : l) {
            System.out.print(m + " ");
        }

        System.out.println();
    }

    public static void nextLargestNoToLeft(int[] a) {

        int n = a.length;
        Stack<Integer> s = new Stack<>();
        List<Integer> l = new ArrayList<>();

        for (int i = 0; i < n; i++) {

            while (!s.isEmpty() && s.peek() < a[i]) {
                s.pop();
            }

            if (s.isEmpty()) {
                l.add(-1);
            } else {
                l.add(s.peek());
            }

            s.push(a[i]);

        }

        for (int m : l) {
            System.out.print(m + " ");
        }

        System.out.println();
    }

    public static void nextSmallestNoToRight(int[] a) {

        int n = a.length;
        Stack<Integer> s = new Stack<>();
        List<Integer> l = new ArrayList<>();

        for (int i = n - 1; i >= 0; i--) {

            while (!s.isEmpty() && s.peek() > a[i]) {
                s.pop();
            }

            if (s.isEmpty()) {
                l.add(-1);
            } else {
                l.add(s.peek());
            }

            s.push(a[i]);

        }

        Collections.reverse(l);

        for (int m : l) {
            System.out.print(m + " ");
        }

        System.out.println();

    }

    public static void nextSmallestNoToLeft(int[] a) {

        int n = a.length;
        Stack<Integer> s = new Stack<>();
        List<Integer> l = new ArrayList<>();

        for (int i = 0; i < n; i++) {

            while (!s.isEmpty() && s.peek() > a[i]) {
                s.pop();
            }

            if (s.isEmpty()) {
                l.add(-1);
            } else {
                l.add(s.peek());
            }

            s.push(a[i]);

        }

        for (int m : l) {
            System.out.print(m + " ");
        }

        System.out.println();

    }

    private static void propagate(int[][] a,
            int x, int y,
            int[][] neighbours,
            boolean[][] visited) {

        int r = a.length;
        int c = a[0].length;
        //System.out.println("R "+r+" C "+c);
        for (int i = 0; i < neighbours.length; i++) {

            int x1 = x + neighbours[i][0];
            int y1 = y + neighbours[i][1];

            //x1 and y1 are neighbour co-ordinate for x,y
            //they should be in boundry of a[][]
            if ((x1 >= 0 && x1 < r) && (y1 >= 0 && y1 < c)) {

                //any new neighbour 
                //1. not visited previously
                //2. also have 1 (we are considering 1 as an island)
                //from x,y  to all neighbour propagte to them
                if (visited[x1][y1] != true && a[x1][y1] == 1) {
//                    System.out.println(x1+" "+" "+y1+" "+a[x1][y1]);
                    visited[x1][y1] = true;
                    propagate(a, x1, y1, neighbours, visited);
                }

            }

        }

    }

    public static void noOfIslands(int[][] a) {

        int[][] neighbour = {
            {-1, -1},
            {-1, 0},
            {-1, 1},
            {0, -1},
            {0, 1},
            {1, -1},
            {1, 0},
            {1, 1}
        };

        /*
        
         for any x in row and y in col
         surrounding 8 neighbours are below 
         Up,Down,Left,Right,Up-Left,Up-Right,Down-Left,Down-Right
        
         ((x-1),(y-1))  ((x-1),(y)) ((x-1),(y+1))
        
         ((x),(y-1))    ((x),(y))    ((x),(y+1))
        
         ((x+1),(y-1))   ((x+1),(y))  ((x+1),(y+1))
        
         */
        long count = 0;
        int row = a.length;
        int col = a[0].length;

        boolean[][] visited = new boolean[row][col];

        for (int x = 0; x < row; x++) {

            for (int y = 0; y < col; y++) {

                if (a[x][y] == 1 && visited[x][y] != true) {

                    visited[x][y] = true;
                    count++;
                    //System.out.println("----"+x+" "+" "+y+" "+a[x][y]);
                    propagate(a, x, y, neighbour, visited);

                }

            }

        }

        System.out.println("no of islands " + count);

    }

    public static void largestSubarrayWithEqual0and1(int[] a) {

        //O(n)
        for (int i = 0; i < a.length; i++) {
            a[i] = (a[i] == 0) ? -1 : 1;
        }

        HashMap<Integer, Integer> map = new HashMap<>();
        int sum = 0;
        int maxLen = 0;
        int endIndex = -1;
        for (int i = 0; i < a.length; i++) {
            sum += a[i];

            if (sum == 0) {
                maxLen = i + 1;
                endIndex = i;
            }

            if (map.containsKey(sum + a.length)) {

                if (maxLen < i - map.get(sum + a.length)) {

                    maxLen = i - map.get(sum + a.length);
                    endIndex = i;
                }

            } else {
                map.put(sum + a.length, i);
            }

        }

//        int end = endIndex - maxLen + 1; 
//        System.out.println(end + " to " + endIndex); 
        System.out.println(" max  " + maxLen);

    }

    public static void kSortedListAsSingle(List<List<Integer>> kSortedList) {

        List<Integer> newList = new ArrayList<>();

        for (List<Integer> l : kSortedList) {
            newList.addAll(l);
        }

        Collections.sort(newList);

        newList.stream().forEach(x -> System.out.print(x + " "));

    }

    private static int PrecedenceOrder(char ch) {
        switch (ch) {
            case '+':
            case '-':
                return 1;

            case '*':
            case '/':
                return 2;

            case '^':
                return 3;
        }
        return -1;
    }

    public static void expressionInfixToPostfix(String expr) {

        StringBuilder sb = new StringBuilder();
        Stack<Character> st = new Stack<>();
        for (int i = 0; i < expr.length(); i++) {

            char ch = expr.charAt(i);

            if (Character.isLetterOrDigit(ch)) {
                sb.append(ch);
            } else if (ch == '(') {
                st.push(ch);
                //System.out.println(st.toString());
            } else if (ch == ')') {

                while (!st.isEmpty() && st.peek() != '(') {
                    sb.append(st.pop());
                }

                if (!st.isEmpty() && st.peek() != '(') {
                    break;
                } else {
                    st.pop();
                }

            } else {
                while (!st.isEmpty() && PrecedenceOrder(ch) <= PrecedenceOrder(st.peek())) {
                    if (st.peek() == '(') {
                        break;
                    }
                    sb.append(st.pop());
                }
                st.push(ch);
//                System.out.println(st.toString());
            }

        }
        while (!st.isEmpty()) {
            if (st.peek() == '(') {
                break;
            }
            sb.append(st.pop());
        }

        System.out.println("postfix: " + sb.toString());

    }

    public static void countSubarrayWithEvenSumOfElements(int[] a) {

        int sum = 0;
        int count = 0;
        for (int i = 0; i < a.length; i++) {
            sum = 0;
            for (int j = i; j < a.length; j++) {

                sum += a[j];
                if (sum % 2 == 0) {
                    //even
                    count++;
                }

            }
        }

        System.out.println("Brute force solution O(n^2) " + count);

        //..............................O(n)....................
        count = 0;
        sum = 0;
        int[] temp = {1, 0};

        for (int i = 0; i < a.length; i++) {

            //this eqtn keep sum as 0 or 1 
            //that we are using as temp[] indexes
            sum = ((sum + a[i]) % 2 + 2) % 2;
            System.out.println("sum... " + sum);

            temp[sum]++;

        }

        count = count + (temp[0] * (temp[0] - 1) / 2);
        count = count + (temp[1] * (temp[1] - 1) / 2);

        System.out.println("O(n) solution count " + count);

    }

    public static void m0sFlipToFindMaxLength1s(int[] a, int m) {

        int wL = 0, wR = 0;
        int zeroCount = 0;
        int bestWin = 0;
        int bestL = 0;
        int maxLengthOf1s = -1;
        while (wR < a.length) {

            //System.out.println("wL "+wL+" wR "+wR+" seroCount "+zeroCount+" bestWin "+bestWin+" bestL "+bestL);
            if (zeroCount <= m) {

                if (a[wR] == 0) {
                    zeroCount++;
                }
                wR++;

            }

            if (zeroCount > m) {

                if (a[wL] == 0) {
                    zeroCount--;
                }
                wL++;

            }

            if ((wR - wL > bestWin) && zeroCount <= m) {
                bestWin = wR - wL;
                bestL = wL;
            }

            //System.out.println("--wL "+wL+" wR "+wR+" seroCount "+zeroCount+" bestWin "+bestWin+" bestL "+bestL);
        }

        maxLengthOf1s = bestWin;

        System.out.println("max length containing 1s in seq " + maxLengthOf1s);
        for (int i = 0; i < bestWin; i++) {

            if (a[bestL + i] == 0) {
                System.out.println((bestL + i) + " ");
            }
        }

    }

    public static void startToEndFromDictionary(Set<String> dictionary, String start, String end) {

        Set<String> chainOfStringOpertion = new LinkedHashSet<>();
        chainOfStringOpertion.add(start);

        //use  BFS based approach
        Queue<String> q = new LinkedList<>();
        q.add(start);

        int stepToFormTarget = 0;
        boolean isDone;

        while (!q.isEmpty()) {

            isDone = false;
            //System.out.println(q.peek());
            char[] word = q.poll().toCharArray();

            for (int i = 0; i < word.length; i++) {

                char chOriginal = word[i];

                for (char c = 'a'; c <= 'z'; c++) {

                    //we will try to replace char of word[i] with a to z
                    //and check the new word so formed is there in the dictionary or not
                    word[i] = c;

                    //if the word formed is in this attempt matches the end string return 
                    String newWordByReplacingOneChar = String.valueOf(word);

                    if (newWordByReplacingOneChar.equals(end)) {
                        System.out.println(newWordByReplacingOneChar);
                        chainOfStringOpertion.add(newWordByReplacingOneChar);
                        stepToFormTarget++;
                        isDone = true;
                        break;
                    }

                    //if dictionary contains the new word
                    //add this new word to queue
                    //and remove the word from dictionary as we don't want to process
                    //that same again
                    if (dictionary.contains(newWordByReplacingOneChar)) {
                        chainOfStringOpertion.add(newWordByReplacingOneChar);
                        q.add(newWordByReplacingOneChar);
                        dictionary.remove(newWordByReplacingOneChar);
                        stepToFormTarget++;
                        isDone = true;
                        break;
                    }

                }

                //if i-th char doesn't make any change
                //then put choriginal back to i-th location
                //and let i change for the word[]
                word[i] = chOriginal;
                if (isDone) {
                    break;
                }
            }

        }

        //both are equal
        System.out.println("Steps " + (stepToFormTarget + 1));
        System.out.println(chainOfStringOpertion.toString() + "  size " + chainOfStringOpertion.size());

    }

    public static void firstNonRepeatingCharInString(String str) {

        LinkedHashMap<Character, Integer[]> m = new LinkedHashMap<>();

        //for find all char frequency and its first index i where it occurs
        for (int i = 0; i < str.length(); i++) {
            char ch = str.charAt(i);
            if (m.containsKey(ch)) {
                //we are not updating index i if wefound again
                Integer[] t = m.get(ch);
                t[0] = t[0] + 1;
                m.put(ch, t);

            } else {
                //first index i when char ch occurred
                //we are not updating index i if wefound again
                //Integer[] [0]=count,[1]=index
                m.put(ch, new Integer[]{1, i});
            }

        }

        //we will  find char in string whoose count is 1 only 
        //and index less than Max_value
        //if there are 2 char whoose freq is 1 we will choose cahr with lesser
        //index value 
        int index = Integer.MAX_VALUE;
        for (char ch : str.toCharArray()) {

            if (m.get(ch)[0] == 1 && index > m.get(ch)[1]) {
                index = m.get(ch)[1];
            }

        }

        if (index == Integer.MAX_VALUE) {
            System.out.println("result -1");
            return;
        }

        System.out.println("result " + str.charAt(index));

    }

    public static void rainWaterTrapping(int[] towerHeight) {

        int waterCapacity = 0;

        int n = towerHeight.length;

        int left = towerHeight[0];
        int right = towerHeight[n - 1];

        int i = 1;
        int j = n - 2;

        int k = 0;

        while ((i < n) && (j >= 0)) {

            waterCapacity += Math.min(left, right) - towerHeight[k];

            //System.out.println(".. l "+left+" r "+right+" t "+towerHeight[k]+" w "+waterCapacity);
            if (Math.max(left, towerHeight[i]) == towerHeight[i]) {
                left = towerHeight[i];
            }

            if (Math.max(right, towerHeight[j]) == towerHeight[j]) {
                right = towerHeight[j];
            }

            //System.out.println("-- l "+left+" r "+right);
            i++;
            j--;
            k++;

        }

        System.out.println("Water capacity " + waterCapacity);

    }
    
    public static void rainWaterTrappingUsingStack(int[] height){
        
        //https://leetcode.com/problems/trapping-rain-water/solution/
        
        //..................T: O(N)
        //..................S: O(N)
        
        int ans = 0;
        int current = 0;
        int N = height.length;
        Stack<Integer> s = new Stack<>();
        while(current < N){
            
            while(!s.isEmpty() && height[current] > height[s.peek()]){
                
                int top = s.pop();
                if(s.isEmpty()){
                    break;
                }
                
                int distance = current - s.peek() - 1;
                int boundedHeight = Math.min(height[current], height[s.peek()]) - height[top];
                ans += distance * boundedHeight;
                
            }
            
            s.push(current++);
        }
        
        //output
        System.out.println("Rain water trapping using stack: "+ans);
        
    }
    
    public static void rainWaterTrappingUsingTwoPointers(int[] height){
        
        //https://leetcode.com/problems/trapping-rain-water/solution/
        
        //OPTIMISED than stack
        //..................T: O(N)
        //..................S: O(1)
        
        int left = 0;
        int right = height.length - 1;
        int ans = 0;
        int leftMax = 0;
        int rightMax = 0;
        while(left < right){
            if (height[left] < height[right]) {
                
                if(height[left] >= leftMax){
                    leftMax = height[left];
                }else {
                    ans += (leftMax - height[left]);
                }
                
                ++left;
            }
            else {
                
                if(height[right] >= rightMax){
                    rightMax = height[right];
                }else {
                    ans += (rightMax - height[right]);
                }
                
                --right;
            }
        }
        
        //output
        System.out.println("Rain water trapping using tow pointers: "+ans);
        
    }

    private static int findMax(int[] a, int i, int j) {

        int max = a[i];
        for (int x = i + 1; x <= j; x++) {
            if (Math.max(max, a[x]) == a[x]) {
                max = a[x];
            }
        }
        return max;
    }

    public static void maxElementInKWindow_ON2(int[] a, int k) {

        int n = a.length;

        for (int i = 0; i <= n - k; i++) {

            System.out.print(findMax(a, i, i + k - 1) + "  ");

        }

        System.out.println();

    }

    public static void maxElementInKWindow_ON(int[] a, int k) {

        /*
        
         to solve this question in O(N) we will use Dqueue
        
         https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-size-k/
        
         */
        int n = a.length;
        Deque<Integer> dq = new LinkedList<>();
        int i = 0;
        for (; i < k; i++) {

            while (!dq.isEmpty() && a[i] >= a[dq.peekLast()]) {
                dq.removeLast();
            }

            dq.addLast(i);

        }

        for (; i < n; i++) {

            System.out.print(a[dq.peek()] + " ");

            while (!dq.isEmpty() && dq.peek() <= i - k) {
                dq.removeFirst();
            }

            while (!dq.isEmpty() && a[i] >= a[dq.peekLast()]) {
                dq.removeLast();
            }

            dq.addLast(i);

        }

        System.out.print(a[dq.peek()] + " ");
        System.out.println();
    }

    public static void bitonicArray(int[] a) {

        //.......O(n)..............
        int max = a[0];
        int bitonicIndex = -1;
        for (int i = 1; i < a.length; i++) {

            if (Math.max(max, a[i]) == a[i]) {
                max = a[i];
                bitonicIndex = i;
            }

        }

        System.out.println("Element " + max + " index " + bitonicIndex);

    }

    private static int findBitonicPoint(int[] a, int l, int r) {

        int mid = (r + l) / 2;

        if (a[mid] > a[mid - 1] && a[mid] > a[mid + 1]) {
            //System.out.println(mid);
            return mid;
        } else if (a[mid] > a[mid - 1] && a[mid] < a[mid + 1]) {
            //stricly inc area
            return findBitonicPoint(a, mid + 1, r);
        } else if (a[mid] < a[mid - 1] && a[mid] > a[mid + 1]) {
            return findBitonicPoint(a, l, mid - 1);
        }
        return mid;
    }

    private static int bitonicBinarySearch(int[] a, int l, int r, int key) {

        while (l < r) {

            int mid = (r + l) / 2;

            if (a[mid] == key) {
                return mid;
            } else if (a[mid] < key) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }

        }

        return -1;

    }

    public static void bitonicArrayBinarySearch(int[] a, int l, int r, int findKey) {

        int bitonicIndex = findBitonicPoint(a, l, r);
        int keyIndex = -1;

        if (findKey > a[bitonicIndex]) {
            System.out.println("Not found");
        } else if (findKey == a[bitonicIndex]) {
            System.out.println("found " + bitonicIndex);
        } else if (findKey < a[bitonicIndex]) {
            //then we have to find key in both left and right half of bitonic index
            keyIndex = bitonicBinarySearch(a, l, bitonicIndex - 1, findKey);
            if (keyIndex != -1) {
                System.out.println("found " + keyIndex);
                return;
            }
            keyIndex = bitonicBinarySearch(a, bitonicIndex + 1, r, findKey);
            System.out.println("found " + keyIndex);
        }

    }

    private static void bogglerUtil(final Set<String> dictionary,
            int x, int y,
            boolean[][] visited,
            int[][] neighbour,
            char[][] a,
            final StringBuilder sb,
            final Set<String> result) {

        visited[x][y] = true;

        sb.append(a[x][y]);

        if (dictionary.contains(sb.toString())) {
            result.add(sb.toString());
            dictionary.remove(sb.toString());
            return;
        }

        for (int i = 0; i < neighbour.length; i++) {

            int x1 = x + neighbour[i][0];
            int y1 = y + neighbour[i][1];

            if ((x1 >= 0 && x1 < a.length) && (y1 >= 0 && y1 < a[0].length)) {

                if (visited[x1][y1] != true) {
                    visited[x1][y1] = true;
                    bogglerUtil(dictionary, x1, y1, visited, neighbour, a, sb, result);
                }

            }

        }

        sb.deleteCharAt(sb.length() - 1);
        visited[x][y] = false;

    }

    public static void boggleSolver(Set<String> dictionary, char[][] charArray) {

        int[][] neighbour = {
            {-1, -1},
            {-1, 0},
            {-1, 1},
            {0, -1},
            {0, 1},
            {1, -1},
            {1, 0},
            {1, 1}
        };

        int row = charArray.length;
        int col = charArray[0].length;

        boolean[][] visited = new boolean[row][col];
        StringBuilder sb = new StringBuilder();
        Set<String> resultSet = new HashSet<>();

        for (int x = 0; x < row; x++) {
            for (int y = 0; y < col; y++) {

                if (visited[x][y] != true) {
                    bogglerUtil(dictionary, x, y, visited, neighbour, charArray, sb, resultSet);
                }
                sb.setLength(0);
            }
        }

        System.out.println("set : " + resultSet.toString());

    }

    public static void boggleSolver_Trie(Set<String> dictionary, char[][] charArray) {

        Trie trie = new Trie();

        for (String s : dictionary) {
            trie.insert(s);
        }

        int[][] neighbour = {
            {-1, -1},
            {-1, 0},
            {-1, 1},
            {0, -1},
            {0, 1},
            {1, -1},
            {1, 0},
            {1, 1}
        };

        int row = charArray.length;
        int col = charArray[0].length;

        Set<String> found = new HashSet<>();
        StringBuilder sb = new StringBuilder();
        for (int x = 0; x < row; x++) {
            for (int y = 0; y < col; y++) {

                //restricting autosuggest upto 2char that
                //are the 8 neighbour of given charArray[x][y]
                for (int z = 0; z < neighbour.length; z++) {

                    int x1 = x + neighbour[z][0];
                    int y1 = y + neighbour[z][1];

                    if ((x1 >= 0 && x1 < row) && (y1 >= 0 && y1 < col)) {
                        sb.append(charArray[x][y]);
                        sb.append(charArray[x1][y1]);

                        found.addAll(trie.autoSuggestQuery(sb.toString()));
                    }

                    sb.setLength(0);
                }

            }
        }

        System.out.println("set : " + found.toString());

    }

    public static void maxProductOfTriplet_ON3(int[] a) {

        //.....................O(n^3)......................
        int n = a.length;

        if (n < 3) {
            System.out.println("triplet product not possble");
            return;
        }

        int max = Integer.MIN_VALUE;
        for (int i = 0; i < n - 2; i++) {
            for (int j = i + 1; j < n - 1; j++) {
                for (int k = j + 1; k < n; k++) {
                    max = Math.max(max, a[i] * a[j] * a[k]);
                }
            }
        }
        System.out.println("max product " + max);
    }

    public static void maxProductOfTriplet_ONLogN(int[] a) {

        //................O(n log n).........................
        //use sorting like quick and merge which is of logn time
        //here for sake of simplicity..
        Arrays.sort(a);
        //asc order
        int n = a.length;

        if (n < 3) {
            System.out.println("triplet product not possble");
            return;
        }

        int max = Math.max(a[0] * a[1] * a[n - 1], a[n - 1] * a[n - 2] * a[n - 3]);
        System.out.println("max product " + max);

    }

    public static void maxProductOfTriplet_ON(int[] a) {

        //......................O(n)....................
        int n = a.length;

        if (n < 3) {
            System.out.println("triplet product not possble");
            return;
        }

        int maxA = Integer.MIN_VALUE, maxB = Integer.MIN_VALUE, maxC = Integer.MIN_VALUE;
        int minA = Integer.MAX_VALUE, minB = Integer.MAX_VALUE;

        for (int i = 0; i < n; i++) {

            //finding all max
            if (a[i] > maxA) {

                maxC = maxB;
                maxB = maxA;
                maxA = a[i];

            } else if (a[i] > maxB) {
                maxC = maxB;
                maxB = a[i];
            } else if (a[i] > maxC) {
                maxC = a[i];
            }

            //find 2 min
            if (a[i] < minA) {
                minB = minA;
                minA = a[i];
            } else if (a[i] < minB) {
                minB = a[i];
            }

        }

        int max = Math.max(minA * minB * maxA, maxA * maxB * maxC);
        System.out.println("max product " + max);

    }

    public static void coinChangeProblem(int[] coin, int amount) {

        long[] ways = new long[amount + 1];
        ways[0] = 1;
        for (int i = 0; i < coin.length; i++) {
            for (int j = 0; j < ways.length; j++) {
                System.out.println(".." + ways[j]);
                if (j >= coin[i]) {
                    ways[j] += ways[(int) (j - coin[i])];
                }
                System.out.println("--" + ways[j]);
            }
        }

        for (long m : ways) {
            System.out.print(m + "  ");
        }

        System.out.println("ways: " + ways[amount]);

    }

    public static void constructTheArray(int n, int k, int x) {

        //n the array size
        //first ele 1 and last is x
        //rest element will be in 1->k range
        int[] a = new int[n];
        int[] b = new int[n];

        a[0] = 0;//ends with x
        b[0] = 1; // not ends with x

        for (int i = 1; i < n; i++) {
            a[i] = b[i - 1];
            b[i] = a[i - 1] * (k - 1) + b[i - 1] * (k - 2);
        }

        System.out.println(a[n - 1]);

    }

    public static void sumOfNumberAsLinkedList(LinkedListUtil<Integer> a, LinkedListUtil<Integer> b) {

        long aLen = a.length();
        long bLen = b.length();

        LinkedListUtil<Integer> a_;
        LinkedListUtil<Integer> b_;

        if (aLen >= bLen) {
            a_ = a.reverse();
            b_ = b.reverse();
        } else {
            a_ = b.reverse();
            b_ = a.reverse();

            aLen = aLen + bLen;
            bLen = aLen - bLen;
            aLen = aLen - bLen;

        }

        LinkedListUtil<Integer> result = new LinkedListUtil<>();
        long i = 1, j = 1;
        int carry = 0;
        while (true) {

            if (i > aLen) {
                if (carry != 0) {
                    result.append(carry);
                }

                break;
            }

            int x = a_.get((int) i);
            int y = 0;
            if (j <= bLen) {
                y = b_.get((int) i);
            }

            int sum = x + y + carry;

            result.append(sum % 10);

            carry = sum / 10;

            i++;
            j++;
        }

        result.reverse().print();
    }

    public static void removeCharToLeftPallindrome(String str) {

        Map<Character, Integer> m = new HashMap<>();
        for (int i = 0; i < str.length(); i++) {

            char c = str.charAt(i);
            if (m.containsKey(c)) {

                int counter = m.get(c);
                m.put(c, counter + 1);

            } else {
                m.put(c, 1);
            }

        }

        int totalOddChar = 0;
        for (Map.Entry<Character, Integer> e : m.entrySet()) {

            if (e.getValue() % 2 == 1) {
                totalOddChar++;
            }

        }

        System.out.println("total no of char to remove to arrange left char as pallindrome " + (totalOddChar - 1));

    }

    private static int firstIndexOfOneBinarySearch(int[] a, int f, int l) {

        int mid = (f + l) / 2;
        int midPrev = mid - 1 < 0 ? 0 : mid - 1;

        //if first occurence of 1 is at 0 index
        //so midPrev  = mid-1 goes negtive
        if (mid == midPrev) {
            return mid;
        }

        //if there is no 1 in the array
        //then after searching mid+1 will go beyond array len
        //then bound it to array's length
        if (mid + 1 > a.length) {
            return a.length;
        }

        if (a[mid] == 1 && a[midPrev] != 1) {
            return mid;
        } else if (a[mid] == 1 && a[midPrev] == 1) {
            return firstIndexOfOneBinarySearch(a, f, mid - 1);
        } else {
            return firstIndexOfOneBinarySearch(a, mid + 1, l);
        }

    }

    public static void count1sInNonDecrBinaryArray_OLongN(int[] a) {
        int firstOccurOfOne = firstIndexOfOneBinarySearch(a, 0, a.length);
        int sumOfOne = a.length - firstOccurOfOne;

        System.out.println("Sum of 1 in binary non dec arrray in O(LogN) " + sumOfOne);
    }

    private static int lastIndexOfOneBinarySearch(int[] a, int f, int l) {

        //In non incr binary array
        //if 0 index is not 1 then later index can't have 1 as element
        if (a[0] == 0) {
            return -1;
        }

        int mid = (f + l) / 2;
        int midNext = mid + 1 > a.length ? a.length - 1 : mid + 1;

        //if first occurence of 1 is at 0 index
        //so midPrev  = mid-1 goes negtive
        if (mid == midNext) {
            return mid;
        }

        if (midNext == a.length) {
            return f;
        }

        if (a[mid] == 1 && a[midNext] != 1) {
            return mid;
        } else if (a[mid] == 1 && a[midNext] == 1) {
            //right
            return lastIndexOfOneBinarySearch(a, mid + 1, l);
        } else {
            //left
            return lastIndexOfOneBinarySearch(a, f, mid - 1);
        }

    }

    public static void count1sInNonIncrBinaryArray_OLongN(int[] a) {
        int lastOccurOfOne = lastIndexOfOneBinarySearch(a, 0, a.length);
        int sumOfOne = lastOccurOfOne + 1;

        System.out.println("Sum of 1 in binary non incr arrray in O(LogN) " + sumOfOne);
    }

    public static void nextFirstSmallerElement_ON2(int[] a) {

        //brute force O(N^2) approach....................
        boolean found = false;
        List<Integer> l = new ArrayList<>();
        for (int i = 0; i < a.length; i++) {
            for (int j = i + 1; j < a.length; j++) {

                if (a[j] <= a[i]) {
                    l.add(a[j]);
                    found = true;
                    break;
                }

            }

            if (!found) {
                l.add(-1);
            }

            found = false;

        }

        System.out.println("Next first smaller element O(n^2) " + l.toString());

    }

    public static void nextFirstSmallerElement_ON(int[] a) {

        Stack<Integer> s = new Stack<>();
        List<Integer> l = new ArrayList<>();

        for (int i = a.length - 1; i >= 0; i--) {

            while (!s.isEmpty() && s.peek() >= a[i]) {
                s.pop();
            }

            if (s.isEmpty()) {
                l.add(-1);
            } else {
                l.add(s.peek());
            }

            s.push(a[i]);

        }

        Collections.reverse(l);
        System.out.println("Next first smaller element O(n) " + l.toString());

    }

    public static void nStairWaysProblem_DP(int n) {

        int[] ways = new int[n + 1];
        //you can either take 1 step or 2 step at a time
        //ways[] index represent as n stairs
        //so reach stair 0, ways[0] = 1 i.e you dont take any step
        //reach stair 1, ways[1] = 1 i.e, you can only take 1 step at a time to reach stair 1
        //but to reach stari 2, ways[2] there can be 2 ways :
        //  1 + 1 step == stair 2
        //  0 + 2 step == stair 2
        ways[0] = 1;
        ways[1] = 1;
        for (int i = 2; i <= n; i++) {

            ways[i] = ways[i - 1] + ways[i - 2];

        }

        System.out.println("Ways required " + ways[n]);

    }

    private static void rotAllAdjacent(int[][] basket,
            int x, int y,
            boolean[][] visited,
            int row, int col) {

        //all aadjacent coordinate
        //check new coordinates are in bounds
        //check new coordinates are not previously visited
        //maake the adjacent rot and mark them visited
        int x1 = -1;
        int y1 = -1;

        //left coordinate to x,y = x, y-1
        x1 = x;
        y1 = y - 1;
        if ((x1 >= 0 && x1 < row) && (y1 >= 0 && y1 < col) && visited[x1][y1] != true && basket[x1][y1] != 0) {
            visited[x1][y1] = true; //maark them visited
            basket[x1][y1] = 2; //make them rot
            rotAllAdjacent(basket, x1, y1, visited, row, col);
        }

        //right coordinate to x,y = x, y+1
        x1 = x;
        y1 = y + 1;
        if ((x1 >= 0 && x1 < row) && (y1 >= 0 && y1 < col) && visited[x1][y1] != true && basket[x1][y1] != 0) {
            visited[x1][y1] = true; //maark them visited
            basket[x1][y1] = 2; //make them rot
            rotAllAdjacent(basket, x1, y1, visited, row, col);
        }

        //top coordinate to x,y = x-1, y
        x1 = x - 1;
        y1 = y;
        if ((x1 >= 0 && x1 < row) && (y1 >= 0 && y1 < col) && visited[x1][y1] != true && basket[x1][y1] != 0) {
            visited[x1][y1] = true; //maark them visited
            basket[x1][y1] = 2; //make them rot
            rotAllAdjacent(basket, x1, y1, visited, row, col);
        }

        //bottom coordinate to x,y = x+1, y
        x1 = x + 1;
        y1 = y;
        if ((x1 >= 0 && x1 < row) && (y1 >= 0 && y1 < col) && visited[x1][y1] != true && basket[x1][y1] != 0) {
            visited[x1][y1] = true; //maark them visited
            basket[x1][y1] = 2; //make them rot
            rotAllAdjacent(basket, x1, y1, visited, row, col);
        }

    }

    public static void rottenOranges(int[][] basket) {

        int rottenTime = 0;
        int row = basket.length;
        int col = basket[0].length;

        boolean[][] visited = new boolean[row][col];

        for (int x = 0; x < row; x++) {
            for (int y = 0; y < col; y++) {
                if (visited[x][y] != true && basket[x][y] == 2) {
                    //rotten oranges == 2
                    visited[x][y] = true;
                    rottenTime++;
                    rotAllAdjacent(basket, x, y, visited, row, col);
                }
            }
        }

        //check if any one is left unrotten(1)
        for (int x = 0; x < row; x++) {
            for (int y = 0; y < col; y++) {
                if (basket[x][y] == 1) {
                    //rotten oranges == 2
                    rottenTime = -1;
                }
            }
        }

        System.out.println("rotten time " + rottenTime);

    }

    public static void kthSmallestElementInArrayStream(int[] a, int k) {

        //.......................O(N*LogN)......................................
        //brute force approach can be taken as sort the array in incr order
        //and in that sorted array k-1th index will be our kth smallest element
        //for sorting you can use merge or quick sort with time O(LogN)
        //for sorting N element in array with above sorting method it will take O(N*LogN) time
        //.......................O(N*LogN)......................................
        //lets take ex: 7 10 4 3 20 15
        //k = 3 so the 3rd smallest is 7 (as 1st smallest is 3 and 2nd is 4)
        //so every time input of array stream changes maintain the kth smallest element
        //for this we will use heap(priority queue) 
        //for kth smallest use maxHeap up to size k
        //for kth largest use minHeap up to size k
        //using time complexity is O(N*LogK) because we are restricting maxHeaps sise only to K
        //so internal heapify will work on k element at any point of time
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                //-1 will reverse the minHeap to maxHeap
                return -1 * (o1.intValue() - o2.intValue());
            }

        });

        for (int m : a) {

            maxHeap.add(m);

            if (maxHeap.size() > k) {
                //always remove the top of maxHeap
                maxHeap.poll();
            }

        }

        //after a new stream value(m) is inserted if it is greater than maxHeap.peek() 
        //it will be added and heapify will work
        //so that the max value remains at top.
        System.out.println("Kth smallest element: " + maxHeap.peek());
    }

    public static void kthLargestElementInArrayStream(int[] a, int k) {

        //.....................O(N*LogN)........................................
        //brute force approach is similar to kth smallest element 
        //arrange the array in decr sorted order using merge or quick sort(O(LogN))
        //sorting N element is O(N*LogN)
        //from the starting k-1th index of array will be the kth largest element.
        //.....................O(N*LogN)........................................
        //lets take ex: 7 10 4 3 20 15
        //k = 3 so the 3rd largest is 10 (as 1st largest is 20 and 2nd is 15)
        //so every time input of array stream changes maintain the kth largest element
        //for this we will use heap(priority queue) 
        //for kth smallest use minHeap up to size k
        //for kth largest use minHeap up to size k
        //using time complexity is O(N*LogK) because we are restricting maxHeaps sise only to K
        //so internal heapify will work on k element at any point of time
        //by default Priority queue mintained as minHeaps
        //with the use of comaparator change its sorting behaviour as maxHeap
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        for (int m : a) {

            minHeap.add(m);
            if (minHeap.size() > k) {
                minHeap.poll();
            }

        }

        System.out.println("Kth Largest element : " + minHeap.peek());

    }

    public static void kSortedArray(int[] a, int k) {

        //sort the array in the range of k only
        //at any i iteration we are restricted to range k only i+k
        //for this we can use minHeap to the elements in i to i+k
        //minHeap will maintain the min element at top.
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        int n = a.length;

        for (int i = 0; i <= k; i++) {
            minHeap.add(a[i]);
        }

        int indexer = 0;
        for (int j = k + 1; j < n; j++) {

            //System.out.println(a[j]+" "+minHeap.peek());
            if (minHeap.peek() <= a[j]) {
                a[indexer++] = minHeap.peek();
                minHeap.poll();
            }

            minHeap.add(a[j]);
        }
        System.out.println("" + minHeap.size());
        while (!minHeap.isEmpty()) {
            a[indexer++] = minHeap.poll();
        }

        for (int m : a) {
            System.out.print(m + " ");
        }

        System.out.println("");
    }

    public static Node<Integer> mergeSortedLinkedListAsc_Recursive(Node<Integer> a, Node<Integer> b) {

        if (a == null) {
            return b;
        }

        if (b == null) {
            return a;
        }

        if (a.getData() <= b.getData()) {
            //System.out.println("-- " + a.getData() + " " + b.getData());
            a.setNext(mergeSortedLinkedListAsc_Recursive(a.getNext(), b));
            return a;
        } else {
            //System.out.println(".. " + a.getData() + " " + b.getData());
            b.setNext(mergeSortedLinkedListAsc_Recursive(a, b.getNext()));
            return b;
        }

    }

    //this static head ref will hold the last max node return by the below if base conditions
    static Node<Integer> headNew;

    public static Node<Integer> mergeSortedLinkedListDesc_Recursive(Node<Integer> a, Node<Integer> b) {

        if (a == null) {
            headNew = b;
            return b;
        }

        if (b == null) {
            headNew = a;
            return a;
        }

        if (a.getData() <= b.getData()) {
            //System.out.println("-- " + a.getData() + " " + b.getData());
            b = mergeSortedLinkedListDesc_Recursive(a.getNext(), b);
            b.setNext(a);
            a.setNext(null);
            return a;
        } else {
            //System.out.println(".. " + a.getData() + " " + b.getData());
            a = mergeSortedLinkedListDesc_Recursive(a, b.getNext());
            a.setNext(b);
            b.setNext(null);
            return b;
        }

    }

    private static void kthSmallestElementInTreeDeepTraverse(PriorityQueue<TreeNode<Integer>> maxHeap, TreeNode t, int k) {

        if (t == null) {
            return;
        }

        maxHeap.add(t);

        //this line here will ensure that 
        //maxHeap size dont go beyond K
        if (maxHeap.size() > k) {
            maxHeap.poll();
        }

        kthSmallestElementInTreeDeepTraverse(maxHeap, t.getLeft(), k);
        kthSmallestElementInTreeDeepTraverse(maxHeap, t.getRight(), k);

    }

    public static void kthSmallestElementInTree(TreeNode<Integer> root, int k) {

        PriorityQueue<TreeNode<Integer>> maxHeap = new PriorityQueue<>(new Comparator<TreeNode<Integer>>() {

            @Override
            public int compare(TreeNode<Integer> o1, TreeNode<Integer> o2) {

                return -1 * (o1.getData() - o2.getData());

            }

        });

        kthSmallestElementInTreeDeepTraverse(maxHeap, root, k);

        System.out.println("Kth smallest element in tree " + maxHeap.peek().getData());

    }

    private static void kthLargestElementInTreeDeepTraverse(PriorityQueue<TreeNode<Integer>> minHeap, TreeNode t, int k) {

        if (t == null) {
            return;
        }

        minHeap.add(t);

        //this line here will ensure that 
        //minHeap size dont go beyond K
        if (minHeap.size() > k) {
            minHeap.poll();
        }

        kthLargestElementInTreeDeepTraverse(minHeap, t.getLeft(), k);
        kthLargestElementInTreeDeepTraverse(minHeap, t.getRight(), k);

    }

    public static void kthLargestElementInTree(TreeNode<Integer> root, int k) {

        PriorityQueue<TreeNode<Integer>> minHeap = new PriorityQueue<>(new Comparator<TreeNode<Integer>>() {

            @Override
            public int compare(TreeNode<Integer> o1, TreeNode<Integer> o2) {

                return (o1.getData() - o2.getData());

            }

        });

        kthSmallestElementInTreeDeepTraverse(minHeap, root, k);

        System.out.println("Kth largest element in tree " + minHeap.peek().getData());

    }

    private static TreeNode<Integer> linkedListToBST_Recursion(int n) {

        if (n <= 0) {
            return null;
        }
        //System.out.println(n+".."+h.getData());
        TreeNode<Integer> left = linkedListToBST_Recursion(n / 2);

        //System.out.println(n+"|r|"+h.getData());
        TreeNode<Integer> root = new TreeNode<>(h.getData());

        root.setLeft(left);

        h = h.getNext();

        //System.out.println(n+"--"+h.getData());
        root.setRight(linkedListToBST_Recursion(n - n / 2 - 1));

        return root;

    }

    //this static head (h) ref is required for above recursion
    //to happen because in line : h = h.getNext(); h will hold
    //updated h ref for every fun() call stack in the JVM while in recursion
    static Node<Integer> h;

    public static void linkedListToBST(Node<Integer> head) {
        //storing local head as static h ref
        h = head;

        LinkedListUtil<Integer> ll = new LinkedListUtil<>(head);
        int n = (int) ll.length();
        System.out.println("size of ll " + n);
        ll.print();

        TreeNode<Integer> rootRec = linkedListToBST_Recursion(n);
        BinaryTree<Integer> root = new BinaryTree<>(rootRec);
        root.treeBFS();

    }

    public static void rowWiseStringZigZag(String str, int nRow) {

        String[] arr = new String[str.length()];
        Arrays.fill(arr, "");
        int iRow = 0;
        boolean goingDown = true;

        for (int i = 0; i < str.length(); i++) {

            arr[iRow] += str.charAt(i);

            if (iRow == nRow - 1) {
                goingDown = false;
            } else if (iRow == 0) {
                goingDown = true;
            }

            if (goingDown) {
                iRow++;
            } else {
                iRow--;
            }

        }

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] != null) {
                System.out.print(arr[i]);
            }
        }

        System.out.println();

    }

    public static void kMostFrequentlyOccuringWordFromStream(List<String> streams, int k) {

        //count the occurence of each word in the streams
        HashMap<String, Long> wordsCount = new HashMap<>();
        for (String s : streams) {
            if (wordsCount.containsKey(s)) {
                wordsCount.put(s, wordsCount.get(s) + 1);
            } else {
                wordsCount.put(s, 1l);
            }
        }

        //for k most frequent words we will use min heap 
        //it will keep on top the least occuring words
        //we will maintain the heap size upto k only
        //inner comparator ensures that minHeap is sorted in such a way that
        //least occuring words come on top but if 2 or more words have same word count
        //then such words are arranged in alphabetical order
        PriorityQueue<String> minHeap = new PriorityQueue<>(
                (s1, s2) -> {
                    return (wordsCount.get(s1).equals(wordsCount.get(s2))) ? s2.compareTo(s1) : (int) (wordsCount.get(s1) - wordsCount.get(s2));
                }
        );

        wordsCount.entrySet().stream().forEach(e -> {

            minHeap.add(e.getKey());
            if (minHeap.size() > k) {
                minHeap.poll();
            }

        });

        //till now we have managed to find to K occuring word
        //but in this minHeap also peek() word will be least occuring in K most occuring
        //and last word of heap will  have max ocuurence of all the K most occuring word
        //like (peek)c++ -> 1, 
        //python -> 2, 
        //java -> 3, 
        //hello -> 5
        //we will be polling out from top and adding to the result
        //in last result list will have word order in asc order (1, 2, 3, 5)
        //but we need to pass K most occuring word so just reverse it
        List<String> result = new ArrayList<>();
        while (!minHeap.isEmpty()) {
            result.add(minHeap.poll());
        }

        Collections.reverse(result);
        //printing k most frequently occuring words
        result.stream().forEach(System.out::println);

    }

    public static void sortLinkedListThatIsSortedIAsAbsoluteValue(Node<Integer> head) {

        Node<Integer> prev = head;
        Node<Integer> curr = head.getNext();

        while (curr != null) {

            if (curr.getData() < prev.getData()) {
                //System.out.println(curr.getData()+"|"+prev.getData());
                prev.setNext(curr.getNext());

                curr.setNext(head);
                head = curr;

                curr = prev;

            }

            prev = curr;
            curr = curr.getNext();
        }

        //print updated head
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(head);
        ll.print();

    }

    private static Node<Integer> merge(Node<Integer> l1, Node<Integer> l2) {
        Node<Integer> l = new Node<>(0), p = l;

        while (l1 != null && l2 != null) {
            if (l1.getData() < l2.getData()) {
                p.setNext(l1);
                l1 = l1.getNext();
            } else {
                p.setNext(l2);
                l2 = l2.getNext();
            }
            p = p.getNext();
        }

        if (l1 != null) {
            p.setNext(l1);
        }

        if (l2 != null) {
            p.setNext(l2);
        }

        return l.getNext();
    }

    public static Node<Integer> sortLinkedListUsingMergeSort(Node<Integer> head) {

        /*
        
         inplace sorting 
         time complexity O(NLogN)
        
         */
        if (head == null || head.getNext() == null) {
            return head;
        }

        // step 1. cut the list to two halves
        Node<Integer> prev = null, slow = head, fast = head;

        while (fast != null && fast.getNext() != null) {
            prev = slow;
            slow = slow.getNext();
            fast = fast.getNext().getNext();
        }

        prev.setNext(null);

        // step 2. sort each half
        Node<Integer> l1 = sortLinkedListUsingMergeSort(head);
        Node<Integer> l2 = sortLinkedListUsingMergeSort(slow);

        // step 3. merge l1 and l2
        return merge(l1, l2);

    }

    public static void validAnagrams_1(String a, String b) {

        /*
         another approach of O(NLogN)
         sort both the string using their char array and then sort the char array
         using merge/quick sort O(LogN)
         then compare both the char arrays if they are equal or not
         if all the char in arrays matches it will be anagrams
        
         ex: "tea" and "eat"
         -> toCharArray = ['t', 'e', 'a'] and ['e', 'a', 't']
         -> sort(Arrays.sort() can be used) = ['a', 'e', 't'] and ['a', 'e', 't']
         -> String.valueOf = "aet" and "aet"
         -> "aet".equals("aet") true = anagrams | false = not anagrams
        
         */
        if (a.equals("") || b.equals("")) {
            System.out.println("Not anagrams");
        }

        int aAsciiSum = 0;
        int bAsciiSum = 0;

        //if both string are anagrams then sum of ascii of its char 
        //should be equal
        //lets say char = 'a' acsii value is 97
        //by writing charAt(i) - 'a' it means we are limiting 
        //alphabets range from a -> z = 97 -> 123 = 0 -> 25
        //if we add the direct ascii values 97 -> 123 for strings a and b
        //it could be large if length of a and b are huge
        //so limiting the char ascii range 
        for (int i = 0; i < a.length(); i++) {
            aAsciiSum += a.charAt(i) - 'a';
        }

        for (int i = 0; i < b.length(); i++) {
            bAsciiSum += b.charAt(i) - 'a';
        }

        if (aAsciiSum == bAsciiSum) {
            System.out.println("anagrams");
        } else {
            System.out.println("Not anagrams");
        }

    }

    public static void validAnagrams_2(String a, String b) {

        if (a.length() != b.length()) {
            System.out.println("Not anagrams");
            return;
        }

        int[] alpha = new int[26];

        for (int i = 0; i < a.length(); i++) {

            alpha[a.charAt(i) - 'a']++;
            alpha[b.charAt(i) - 'a']--;

        }

        boolean anagram = true;
        for (int i : alpha) {
            if (i != 0) {
                anagram = false;
            }
        }

        if (anagram) {
            System.out.println("anagrams");
        } else {
            System.out.println("Not anagrams");
        }

    }

    public static void groupAnagrams(String[] strs) {

        List<List<String>> l = new ArrayList<>();
        Map<String, List<String>> m = new HashMap<>();

        for (String s : strs) {

            char[] ch = s.toCharArray();
            Arrays.sort(ch);
            String sortedString = String.valueOf(ch);

            if (m.containsKey(sortedString)) {
                m.get(sortedString).add(s);
            } else {
                List<String> k = new ArrayList<>();
                l.add(k);
                k.add(s);
                m.put(sortedString, k);
            }

        }

        System.out.println("Grouped all anagrams " + l);

    }

    public static void findAnagrams(String s, String p) {

        //this solution was taking about 2sec of runtime on leetcode 
        /*
        
         k-->i substring of len == n
         subtring sort  == p.sort 
         mark index
         k=i
        
         */
        List<Integer> result = new ArrayList<>();
        int m = s.length();
        int n = p.length();

        int i = 0;
        int k = 0;

        char[] ch = p.toCharArray();
        Arrays.sort(ch);
        String sortedP = String.valueOf(ch);

        while (i <= m - n) {

            String ss = s.substring(k, i + n);
            char[] ch_ = ss.toCharArray();
            Arrays.sort(ch_);
            String sortedSS = String.valueOf(ch_);

            if (sortedSS.equals(sortedP)) {
                result.add(k);
            }

            i++;
            k = i;

        }
        System.out.println("indexes of anagaram of p " + result);

    }

    public static void inPlaceReverseLinkedList(Node head) {

        /*
        
         ............Space O(N)................................
         maintain a stack 
         traverse the linked list and put element in the stack
         inthis the last element at stack will be the last element of our Linkedlist
         now traverse the stack until  it is empty and create node 
        
         ............Space O(1)................................
         this solution is inplace reversing
         means we are not using heap memory of ext datastructure to hold our N node
         we are just changin the next pointers of the List
        
         */
        Node prev = null;
        Node curr = head;
        Node next = null;

        while (curr != null) {

            next = curr.getNext();
            curr.setNext(prev);
            prev = curr;
            curr = next;

        }

        head = prev;

        LinkedListUtil ll = new LinkedListUtil(head);
        ll.print();

    }

    public static void reverseLinkedList_2(Node head, int m, int n) {

        Node mth = null;
        Node nth = null;
        int indexer = 1;
        Node temp = head;
        while (m < n) {

            if (indexer == m) {
                mth = temp;
            }

            if (indexer == n) {
                nth = temp;
            }

            if (mth != null && nth != null) {

                int val = (int) mth.getData();
                mth.setData((int) nth.getData());
                nth.setData(val);
                temp = mth;
                mth = nth = null;

                m++;
                n--;
                indexer = m - 1;

            }

            indexer++;
            temp = temp.getNext();

        }

        LinkedListUtil ll = new LinkedListUtil(head);
        ll.print();

    }

    private static void reorderList_1(Node head) {

        //143. Reorder List leetcode/ interviewbit
        /*
        
         optimized O(N^2)
         space O(1)
        
         */
        if (head == null) {
            LinkedListUtil ll = new LinkedListUtil(head);
            ll.print();
            return;
        }

        if (head.getNext() == null || head.getNext().getNext() == null) {
            LinkedListUtil ll = new LinkedListUtil(head);
            ll.print();
            return;
        }

        Node lPrev = null;
        Node last = head;
        Node curr = head;
        Node next = null;

        while (curr != null) {

            next = curr.getNext();
            while (last.getNext() != null) {
                lPrev = last;
                last = last.getNext();
            }

            if (curr.getNext() == last || curr == last) {
                break;
            }

            lPrev.setNext(last.getNext());
            last.setNext(next);
            curr.setNext(last);
            curr = next;
            last = curr;

        }

        LinkedListUtil ll = new LinkedListUtil(head);
        ll.print();

    }

    private static void reorderList_2(Node head) {

        /*
        
         optimized O(N)
         space O(N)
        
         */
        if (head == null) {
            LinkedListUtil ll = new LinkedListUtil(head);
            ll.print();
            return;
        }

        if (head.getNext() == null || head.getNext().getNext() == null) {
            LinkedListUtil ll = new LinkedListUtil(head);
            ll.print();
            return;
        }

        Stack<Node> st = new Stack<>();
        Node temp = head;
        while (temp != null) {

            st.push(temp);
            temp = temp.getNext();

        }

        Node curr = head;
        Node next = head.getNext();

        while (curr != null || curr.getNext() != null) {

            Node last = st.pop();

            if (curr.getNext() == last || curr == last) {
                break;
            }

            st.peek().setNext(last.getNext());

            last.setNext(next);
            curr.setNext(last);
            curr = next;
            next = curr.getNext();

        }

        LinkedListUtil ll = new LinkedListUtil(head);
        ll.print();

    }

    public static Node<Integer> rotateListUptoK(Node<Integer> head, int k) {
        int i = 0;
        Node<Integer> temp = head;
        Node<Integer> prev = null;
        while (i < k) {

            while (temp.getNext() != null) {
                prev = temp;
                temp = temp.getNext();
            }

            prev.setNext(null);
            temp.setNext(head);
            head = temp;
            temp = head;
            i++;

        }

        return head;
    }

    public static Node<Integer> rotateListUptoK_Efficient(Node<Integer> head, int k) {

        Node<Integer> fast = head, slow = head;

        int len = 1;

        while (fast.getNext() != null) {
            len++;
            fast = fast.getNext();
        }

//        System.out.println(len + " " + fast.getData());
        //Get the i-k%i th node
        for (int j = len - k % len; j > 1; j--) {
            slow = slow.getNext();
//            System.out.println(j + " " + slow.getData());
        }

//        System.out.println(len + " " + slow.getData());
        fast.setNext(head); //Do the rotation
        head = slow.getNext();
        slow.setNext(null);

        return head;
    }

    public static void verticalTraversal(TreeNode root) {

        if (root == null) {
            return;
        }

        Queue<Pair<TreeNode, Integer>> q = new LinkedList<>();
        q.add(new Pair<>(root, 0));

        Map<Integer, List<Integer>> m = new TreeMap<>();

        while (!q.isEmpty()) {

            Pair<TreeNode, Integer> t = q.poll();

            int vLevel = t.getValue();
            TreeNode n = t.getKey();

            if (m.containsKey(vLevel)) {
                List<Integer> existingList = m.get(vLevel);
                existingList.add((Integer) n.getData());
                m.put(vLevel, existingList);
            } else {
                List<Integer> firstList = new ArrayList<Integer>();
                firstList.add((Integer) n.getData());
                m.put(vLevel, firstList);
            }

            if (n.getLeft() != null) {
                q.add(new Pair<>(n.getLeft(), vLevel - 1));
            }

            if (n.getRight() != null) {
                q.add(new Pair<>(n.getRight(), vLevel + 1));
            }

        }

        List<List<Integer>> result = new ArrayList();
        for (Map.Entry<Integer, List<Integer>> e : m.entrySet()) {
            // Collections.sort(e.getValue());
            result.add(e.getValue());
        }

        System.out.println(result);
    }

    private static int transformToSumTree_SumSubTree(TreeNode<Integer> root) {

        if (root == null) {
            return 0;
        }

        int someSum = transformToSumTree_SumSubTree(root.getLeft());
        someSum += root.getData();
        someSum += transformToSumTree_SumSubTree(root.getRight());

        return someSum;

    }

    public static void transformToSumTree(TreeNode<Integer> root) {

        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);
        while (!q.isEmpty()) {

            TreeNode<Integer> t = q.poll();

            int sumLeftTree = transformToSumTree_SumSubTree(t.getLeft());
            int sumRightTree = transformToSumTree_SumSubTree(t.getRight());

            t.setData(sumLeftTree + sumRightTree);

            if (t.getLeft() != null) {
                q.add(t.getLeft());
            }

            if (t.getRight() != null) {
                q.add(t.getRight());
            }

        }

        BinaryTree<Integer> bt = new BinaryTree<>(root);
        bt.treeBFS();

    }

    public static Node<Integer> reverseLinkedListInKGroups_1(Node<Integer> head, int k) {

        //https://www.geeksforgeeks.org/reverse-a-list-in-groups-of-given-size/
        Node<Integer> current = head;
        Node<Integer> next = null;
        Node<Integer> prev = null;

        int count = 0;

        /* Reverse first k nodes of linked list */
        while (count < k && current != null) {
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
            head.setNext(reverseLinkedListInKGroups_1(next, k));
        }

        // prev is now head of input list 
        return prev;
    }

    public static Node<Integer> reverseLinkedListInKGroups_2(Node<Integer> head, int k) {

        //https://leetcode.com/problems/reverse-nodes-in-k-group/
        Node<Integer> node = head;
        int count = 0;
        while (count < k) {

            if (node == null) {
                return head;
            }

            node = node.getNext();
            count++;

        }

        Node<Integer> pre = reverseLinkedListInKGroups_2(node, k);

        while (count > 0) {

            Node<Integer> next = head.getNext();
            head.setNext(pre);
            pre = head;
            head = next;
            count--;

        }

        return pre;

    }

    public static int stiklerThief(int[] houses, int n) {

        if (n <= 0) {
            return 0;
        }

        return Math.max(houses[n - 1] + stiklerThief(houses, n - 2), stiklerThief(houses, n - 1));

    }

    public static void kThDistinctElementInArray(int[] a, int k) {

        class OccureneceRecord {

            int index;
            int occr;
            int element;

            public OccureneceRecord(int index, int occr, int element) {
                this.index = index;
                this.occr = occr;
                this.element = element;
            }

        }

        HashMap<Integer, OccureneceRecord> map = new HashMap<>();
        for (int i = 0; i < a.length; i++) {

            if (map.containsKey(a[i])) {
                map.get(a[i]).occr++;
            } else {
                map.put(a[i], new OccureneceRecord(i, 1, a[i]));
            }

        }

        PriorityQueue<OccureneceRecord> maxHeap = new PriorityQueue<>(
                (o1, o2) -> o2.index - o1.index
        );

        map.entrySet().stream().forEach(e -> {

            if (e.getValue().occr == 1) {
                maxHeap.add(e.getValue());
                if (maxHeap.size() > k) {
                    maxHeap.poll();
                }
            }

        });

        if (maxHeap.size() == 0) {
            System.out.println("K th distinct element in the array is: " + (-1));
            return;
        }

        System.out.println("K th distinct element in the array is: " + maxHeap.peek().element);

    }

    private static void printBinaryTreeAsDoublyLL(TreeNode head) {
        if (head == null) {
            throw new RuntimeException("LinkedList is already empty");
        }

        TreeNode temp = head;
        TreeNode tempPrev = null;
//        System.out.println("end to head");
        while (temp != null) {

//            System.out.print(temp.getData()+" ");
            tempPrev = temp;
            temp = temp.getLeft();

        }

        System.out.println("head to end");
        while (tempPrev != null) {

            System.out.print(tempPrev.getData() + " ");
            tempPrev = tempPrev.getRight();

        }

        System.out.println();

    }

    public static TreeNode convertBinaryTreeToDoublyLL(TreeNode node, TreeNode parent) {

        if (node == null) {
            return null;
        }

        node.setLeft(convertBinaryTreeToDoublyLL(node.getLeft(), node));
        if (node.getLeft() != null) {
            node.getLeft().setRight(node);
        }

        node.setRight(convertBinaryTreeToDoublyLL(node.getRight(), node));
        if (node.getRight() != null) {
            node.getRight().setLeft(node);
        }

        if (parent != null && parent.getLeft() == node) {
            //move to extreme right
            while (node.getRight() != null) {
                node = node.getRight();
            }
        } else if (parent != null && parent.getRight() == node) {
            //move to extreme right
            while (node.getLeft() != null) {
                node = node.getLeft();
            }
        }

        return node;
    }

    //req some static variables for this second appraoch
    static TreeNode prev;
    static TreeNode headOfDLL;

    public static void convertBinaryTreeToDoublyLL(TreeNode node) {

        //much straight forward
        if (node == null) {
            return;
        }

        convertBinaryTreeToDoublyLL(node.getLeft());

        if (prev == null) {
            headOfDLL = node;
        } else {
            node.setLeft(prev);
            prev.setRight(node);
        }

        prev = node;

        convertBinaryTreeToDoublyLL(node.getRight());

    }

    static int maxPathSum = Integer.MIN_VALUE;

    public static int maxPathSumOfBinaryTree(TreeNode<Integer> node) {

        if (node == null) {
            return 0;
        }
//        System.out.println(node.data);
        int left = Math.max(0, maxPathSumOfBinaryTree(node.getLeft()));
        int right = Math.max(0, maxPathSumOfBinaryTree(node.getRight()));
        maxPathSum = Math.max(maxPathSum, left + right + node.getData());
        return Math.max(left, right) + node.getData();
    }

    private static boolean deepPruning(TreeNode<Integer> node) {

        if (node == null) {
            return false;
        }

        boolean presenceOfOneInLeft = deepPruning(node.getLeft());
        if (!presenceOfOneInLeft) {
            node.setLeft(null);
        }

        boolean presenceOfOneInRight = deepPruning(node.getRight());
        if (!presenceOfOneInRight) {
            node.setRight(null);
        }

        if (node.getData() == 1) {
            return true;
        }

        return presenceOfOneInLeft || presenceOfOneInRight;

    }

    public static TreeNode treePrune(TreeNode<Integer> root) {

        if (!deepPruning(root)) {
            return null;
        }

        return root;
    }

    private static TreeNode lcaSolver(TreeNode<Integer> root, TreeNode<Integer> p, TreeNode<Integer> q) {

        if (root == null) {
            return null;
        }

        if (root.getData() == p.getData() || root.getData() == q.getData()) {
            return root;
        }

        TreeNode leftLca = lcaSolver(root.getLeft(), p, q);
        TreeNode rightLca = lcaSolver(root.getRight(), p, q);

        if (leftLca != null && rightLca != null) {
            return root;
        }

        return leftLca != null ? leftLca : rightLca;

    }

    public static TreeNode lowestCommonAncestor(TreeNode<Integer> root, TreeNode<Integer> p, TreeNode<Integer> q) {

        /*
        
         this method works for BST and non bst
        
         */
        return lcaSolver(root, p, q);

    }

    private static int depth(TreeNode<Integer> node) {

        if (node == null) {
            return 0;
        }

        return Math.max(depth(node.getLeft()), depth(node.getRight())) + 1;

    }

    public static TreeNode<Integer> lcaDeepestLeaves(TreeNode root) {

        if (root == null) {
            return null;
        }

        int leftDepth = depth(root.getLeft());
        int rightDepth = depth(root.getRight());

        //if a certain root its left subtree depth matches with riight subtree depth
        //then that root is lca for its deepest leaf node
        if (leftDepth == rightDepth) {
            return root;
        }

        //if leftDepth is higher then lca for a deepest leaf node is inside somewhere
        //in the left subtree
        //else it is in the right subtree
        return leftDepth > rightDepth ? lcaDeepestLeaves(root.getLeft()) : lcaDeepestLeaves(root.getRight());

    }

    public static TreeNode<Integer> lowestCommonAncestorInBinarySearchTree_Recursion(TreeNode<Integer> node, int n1, int n2) {

        /*
        
         this method only works for BST
        
         */
        //recursion maintains function call stack.
        if (node == null) {
            return null;
        }

        if (node.getData() > n1 && node.getData() > n2) {
            return lowestCommonAncestorInBinarySearchTree_Recursion(node.getLeft(), n1, n2);
        }

        if (node.getData() < n1 && node.getData() < n2) {
            return lowestCommonAncestorInBinarySearchTree_Recursion(node.getRight(), n1, n2);
        }

        return node;
    }

    public static TreeNode<Integer> lowestCommonAncestorInBinarySearchTree_Iterative(TreeNode<Integer> node, int n1, int n2) {

        /*
        
         this method only works for BST
        
         */
        //efficient than recursion
        while (node != null) {

            if (node.getData() > n1 && node.getData() > n2) {
                node = node.getLeft();
            } else if (node.getData() < n1 && node.getData() < n2) {
                node = node.getRight();
            } else {
                break;
            }

        }

        return node;

    }

    //this static class is req for the below algo 
    //for isBinaryTreeHeightBalnced()
    static class Height {

        int height = 0;
    }

    public static boolean isBinaryTreeHeightBalanced(TreeNode node, Height height) {

        if (node == null) {
            height.height = 0;
            return true;
        }

        Height leftHeight = new Height();
        Height rightHeight = new Height();

        boolean l = isBinaryTreeHeightBalanced(node.getLeft(), leftHeight);
        boolean r = isBinaryTreeHeightBalanced(node.getRight(), rightHeight);

        height.height = Math.max(leftHeight.height, rightHeight.height) + 1;

        if (Math.abs(leftHeight.height - rightHeight.height) > 1) {
            return false;
        }

        return l && r;

    }

    public static int diameterOfTree(TreeNode node, Height height) {

        if (node == null) {
            //this will set the height of the node as we passed the height ref in fun() call
            //itself
            height.height = 0;
            //this return statement will give out the diameter of the same node
            return 0;
        }

        Height leftHeight = new Height();
        Height rightHeight = new Height();

        int lD = diameterOfTree(node.getLeft(), leftHeight);
        int rD = diameterOfTree(node.getRight(), rightHeight);

        //choosing the max height from the leftSubtree or rightSubTree  and adding
        // 1 for the current node itself 
        height.height = Math.max(leftHeight.height, rightHeight.height) + 1;

        return Math.max(leftHeight.height + rightHeight.height + 1, Math.max(lD, rD));

    }

    static class Result {

        int result = Integer.MIN_VALUE;
    }

    public static int maxPathSumOfTreeAnyNodeToAnyNode(TreeNode node, Result result) {

        if (node == null) {
            //this return statement will give out the sum of the null node
            return 0;
        }

        //get the sum for left subtree and right subtree
        int lPathSum = maxPathSumOfTreeAnyNodeToAnyNode(node.getLeft(), result);
        int rPathSum = maxPathSumOfTreeAnyNodeToAnyNode(node.getRight(), result);

        //a. either leftPathSum or rightPathSum is greater and the current node will include that
        //b. if leftPathSum and rightPathSum is -ve and our current node is +ve then by default 
        //our current node is greater than all
        //choose max(a, b)
        //temp is a max path can be formed either through 
        //a. leftPath including that node or 
        //b. rightPath including that node
        //z = max(leftPath, rightPath) + including node
        //c. if z (leftPath and rightPath is negative) then path will be through the node only.
        int temp = Math.max(Math.max(lPathSum, rPathSum) + (Integer) node.getData(), (Integer) node.getData());
        //if z or c is not forming max Path then inner subtree will be max path.
        int ans = Math.max(temp, lPathSum + rPathSum + (Integer) node.getData());
        //final max Path will be stored in result
        // max path through leftPath + node
        //max path through rightPath + node
        //max path through the node only (not including both leftPath and rightPath)
        //max path through inner sub tree leftPath + node + rightPath
        result.result = Math.max(Math.max(temp, ans), result.result);
        return temp;

    }

    public static int maxPathSumOfTreeLeafNodeToLeafNode(TreeNode node, Result res) {
        // Base cases 
        if (node == null) {
            return 0;
        }
        if (node.getLeft() == null && node.getRight() == null) {
            res.result = Math.max(res.result, (Integer) node.getData());
            return (Integer) node.getData();
        }

        // Find maximum sum in left and right subtree. Also 
        // find maximum root to leaf sums in left and right 
        // subtrees and store them in ls and rs 
        int ls = maxPathSumOfTreeLeafNodeToLeafNode(node.getLeft(), res);
        int rs = maxPathSumOfTreeLeafNodeToLeafNode(node.getRight(), res);

        // If both left and right children exist 
        if (node.getLeft() != null && node.getRight() != null) {

            // Update result if needed 
            res.result = Math.max(res.result, ls + rs + (Integer) node.getData());

            // Return maxium possible value for root being 
            // on one side 
            return Math.max(ls, rs) + (Integer) node.getData();
        }

        // If any of the two children is empty, return 
        // root sum for root being on one side 
        return (node.getLeft() == null) ? rs + (Integer) node.getData()
                : ls + (Integer) node.getData();

    }

    public static boolean isSubTree(TreeNode<Integer> main, TreeNode<Integer> sub) {

        //empty trees are also subtrees
        if (main == null && sub == null) {
            return true;
        }

        //if one tree is null and other one is not
        //there is no way we can say one is subtree of another
        if (main == null || sub == null) {
            return false;
        }

        //if sub tree's first node matches with the main tree's first node
        //then we need to proove sub.left also matches main.left and sub.right matches main.right
        //and so on till sub trees end
        //if they will match it will return true
        //if sub.left doesn't match with main.left OR sub.right doesn't match with main.right
        //we will not move further in this if block...
        if (main.getData() == sub.getData()
                && isSubTree(main.getLeft(), sub.getLeft())
                && isSubTree(main.getRight(), sub.getRight())) {
            return true;
        }

        //if above if-block is not able to find sub inside main
        //then there could be chances sub can be found in main tree's left subtree
        //OR sub can be found in main tree's right subtree.
        return isSubTree(main.getLeft(), sub) || isSubTree(main.getRight(), sub);

    }

    private static boolean isTreeSymmetricRecurionHelper(TreeNode<Integer> n1, TreeNode<Integer> n2) {

        //.................O(N)
        if (n1 == null && n2 == null) {
            return true;
        }

        if (n1 == null || n2 == null) {
            return false;
        }

        return (n1.getData() == n2.getData()
                && isTreeSymmetricRecurionHelper(n1.getLeft(), n2.getRight())
                && isTreeSymmetricRecurionHelper(n1.getRight(), n2.getLeft()));

    }

    public static boolean isTreeSymmetric_Recursion(TreeNode<Integer> root) {
        return isTreeSymmetricRecurionHelper(root, root);
    }

    public static boolean isTreeSymmetric_Iteraative(TreeNode<Integer> root) {

        //.................O(N)
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        q.add(root);

        while (!q.isEmpty()) {

            TreeNode n1 = q.poll();
            TreeNode n2 = q.poll();

            if (n1 == null && n2 == null) {
                return true;
            }

            if (n1 == null || n2 == null) {
                return false;
            }

            if (n1.getData() != n2.getData()) {
                return false;
            }

            q.add(n1.getLeft());
            q.add(n2.getRight());

            q.add(n1.getRight());
            q.add(n2.getLeft());

        }

        return false;

    }

    static TreeNode<Integer> prevForIsTreeBST;

    public static boolean isTreeBST(TreeNode<Integer> node) {

        if (node == null) {
            return true;
        }

        boolean leftIsBST = isTreeBST(node.getLeft());
        if (leftIsBST == false) {
            return false;
        }

        if (prevForIsTreeBST != null && node.getData() < prevForIsTreeBST.getData()) {
            return false;
        }

        prevForIsTreeBST = node;

        return isTreeBST(node.getRight());

    }

    public static boolean canFinish(int numCourses, int[][] prerequisites) {

        //this is based on Kahn's algorithm
        //topological sort BFS 
        int[][] matrix = new int[numCourses][numCourses]; // i -> j
        int[] indegree = new int[numCourses];

        for (int i = 0; i < prerequisites.length; i++) {
            int ready = prerequisites[i][0];
            int pre = prerequisites[i][1];
            if (matrix[pre][ready] == 0) {
                indegree[ready]++; //duplicate case
            }
            matrix[pre][ready] = 1;
        }

        int count = 0;
        Queue<Integer> queue = new LinkedList();
        for (int i = 0; i < indegree.length; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int course = queue.poll();
            count++;
            for (int i = 0; i < numCourses; i++) {
                if (matrix[course][i] != 0) {
                    if (--indegree[i] == 0) {
                        queue.offer(i);
                    }
                }
            }
        }
        return count == numCourses;
    }

    public static void removeMiddleElementFromStack(Stack<Integer> s, int stackSize, int curr) {

        /*
         challenge is to not use any other data structure to do this
         */
        //using recursion
        if (s.isEmpty() || curr == stackSize) {
            return;
        }

        int popped = s.pop();

        removeMiddleElementFromStack(s, stackSize, curr + 1);

        if (curr != Math.floor(stackSize / 2) + 1) {
            s.push(popped);
        }

    }

    static Stack<Integer> stackReverse = new Stack<>();

    private static void insertInStack(int ele) {

        //if stack is empty push that element
        if (stackReverse.isEmpty()) {
            stackReverse.push(ele);
        } else {
            //if stack is not empty
            //pop all the elements again until stack is empty
            //push the ele in empty stack
            int x_ = stackReverse.pop();
            insertInStack(ele);
            //then push in all the poppped element
            stackReverse.push(x_);
        }

    }

    public static void reverseStackRecursion() {

        if (stackReverse.size() > 0) {
            //pop out all element from stack
            //and in each recursive call x holds popped element
            int x = stackReverse.pop();
            //go in recursion to pop
            reverseStackRecursion();
            //push all element back
            insertInStack(x);
        }

    }

    public static int theMinimumCost_HackerEarth(int N, int M, int cost) {

        if (N > M) {
            return -1;
        }

        if (N == M) {
            return 0;
        }

        int c = 0;
        for (int d = 2; d < N; d = d + 2) {

            if (N % d == 0) {
                c = N / d;
                c += theMinimumCost_HackerEarth(N + d, M, c);
            }

        }

        if (c == 0) {
            return -1;
        }

        return c;

    }

    public static void inPlace90AntiClockRotateMatrix(int[][] mat) {
        int N = mat.length;
        for (int x = 0; x < N / 2; x++) {
            for (int y = x; y < N - 1 - x; y++) {

                int temp = mat[x][y];
                mat[x][y] = mat[y][N - 1 - x];
                mat[y][N - 1 - x] = mat[N - 1 - x][N - 1 - y];
                mat[N - 1 - x][N - 1 - y] = mat[N - 1 - y][x];
                mat[N - 1 - y][x] = temp;

                //enable this if you want to only rorate outer edges of 
                //matrix
//                if(x != 0 && x != N - 1 && y != 0 && y != N - 1){
//                    continue;
//                }
                //90degree clockwwise code...
//                int temp = mat[y][N - 1 - x];
//                mat[y][N - 1 - x] = mat[x][y];
//                mat[x][y] = mat[N - 1 - y][x];
//                mat[N - 1 - y][x] = mat[N - 1 - x][N - 1 - y];
//                mat[N - 1 - x][N - 1 - y] = temp;
            }
        }

        //printing
        for (int x = 0; x < N; x++) {
            System.out.println();
            for (int y = 0; y < N; y++) {
                System.out.print(mat[x][y] + "\t");
            }
        }

    }

    private static int helper(int N, int K, int[][] dict) {
        if (N == 1) {
            return 0;
        }
        int num = helper(N - 1, K / 2, dict);
        return dict[num][K % 2];
    }

    public static int kthGrammar(int N, int K) {
        int[][] dict = {{0, 1}, {1, 0}};
        return helper(N, K - 1, dict);
    }

    public static int kthGrammar_2(int N, int K) {

        /*
        
         easier explanations:...
        
         K -> 
         len of grammer 2^N-1        N=1 0
         len of grammer 2^N-1        N=2 01
         len of grammer 2^N-1        N=3 0110
         len of grammer 2^N-1        N=4 01101001
        
         for given N and K find the bit in the grammar
         observe N=4 len = 2^N-1 = 8
         in this, first half 0110 is exactly same as N=3 
         and other half 1001 is exactly reverse of all K in N=3 (0110) N=4 other_half(1001)
         so if K=1 to mid of len at given Nth row = K at N-1th row
         ex: N=4 K=1 = N=3 K=1 == 0 upto K = mid = 4
         else if K > mid of len at given Nth row (K=mid+1 to len(Nth row)) = rev(K-mid) at N-1th row
         ex: N=4 K=6 = N=3 K=(K-mid=6-4)=2 = 1 and rev(1) = 0
        
         */
        if (N == 1) {
            return 0;
        }

        int mid = (int) (Math.pow(2, N - 1) / 2);
        if (K <= mid) {
            //first half of Nth row
            return kthGrammar_2(N - 1, K);
        } else {
            //other half of Nth row ans would be rev
            return kthGrammar_2(N - 1, K - mid) == 0 ? 1 : 0;
        }

    }

    public static void connectRopesWithMinCost(int[] ropesLength) {

        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int i : ropesLength) {
            minHeap.add(i);
        }
        int cost = 0;

        //if heap size goes below 2 i.e, 1 that means we have combined all the ropes to make one
        while (minHeap.size() >= 2) {

            //first shorter lengthed rope
            int firstRope = minHeap.poll();

            //second shorter lengthed rope
            int secondRope = minHeap.poll();

            //cost for joining 2 ropes
            cost += firstRope + secondRope;

            //now 2 ropes are combined to form one rope of length (firstRope.len + secondRope.len)
            //add it heap again
            minHeap.add(firstRope + secondRope);

        }

        System.out.println("min cost of joining all the ropes together " + cost);

    }

    public static String intToRoman(int num) {

        Map<Integer, String> map = new HashMap<>();
        map.put(0, "");
        map.put(1, "I");
        map.put(2, "II");
        map.put(3, "III");
        map.put(4, "IV");
        map.put(5, "V");
        map.put(6, "VI");
        map.put(7, "VII");
        map.put(8, "VIII");
        map.put(9, "IX");
        map.put(10, "X");
        map.put(20, "XX");
        map.put(30, "XXX");
        map.put(40, "XL");
        map.put(50, "L");
        map.put(60, "LX");
        map.put(70, "LXX");
        map.put(80, "LXXX");
        map.put(90, "XC");
        map.put(100, "C");
        map.put(200, "CC");
        map.put(300, "CCC");
        map.put(400, "CD");
        map.put(500, "D");
        map.put(600, "DC");
        map.put(700, "DCC");
        map.put(800, "DCCC");
        map.put(900, "CM");
        map.put(1000, "M");
        map.put(2000, "MM");
        map.put(3000, "MMM");

        int mul = 0;
        StringBuilder sb = new StringBuilder();
        while (num != 0) {

            // System.out.println(num +" -- " +(Math.pow(10, mul) * (num%10))+" -- "+map.get((int)(Math.pow(10, mul++) * (num%10))));
            sb.insert(0, map.get((int) (Math.pow(10, mul++) * (num % 10))));

            num /= 10;

        }

        return sb.toString();

    }

    public static int romanToInt(String s) {

        Map<Character, Integer> roman = new HashMap<>();
        roman.put('I', 1);
        roman.put('V', 5);
        roman.put('X', 10);
        roman.put('L', 50);
        roman.put('C', 100);
        roman.put('D', 500);
        roman.put('M', 1000);

        int result = 0;

        for (int i = 0; i < s.length(); i++) {

            if ((i - 1 >= 0) && (roman.get(s.charAt(i - 1)) < roman.get(s.charAt(i)))) {
                result += roman.get(s.charAt(i)) - 2 * roman.get(s.charAt(i - 1));
            } else {
                result += roman.get(s.charAt(i));
            }

        }

        return result;

    }

    public static void slidingWindowArrayWithFixedWindowK_BasicAlgo(int[] a, int K) {

        /*
         time.................O(N)............................
         */
        System.out.println("array size: " + a.length);
        int n = a.length;
        int start = 0;
        int end = 0;
        List<Integer> subArray = new ArrayList<>();
        while (end < n) {

            subArray.add(a[end]);
//            System.out.println(start +" : "+end);
            if (end - start + 1 < K) {
                end++;
            } else if (end - start + 1 == K) {
                //logic changes here as required
//                System.out.println(" K "+(end - start + 1));
                System.out.println("Sub array of given size K : " + subArray);
                //to shift window bracket remove what was first added in sub array i.e element at 0th index
                subArray.remove(0);
//                and incr start bracket
                start++;
                //and incr end bracket
                end++;

            }

        }

    }

    public static void slidingWindowStringWithFixedWindowK_BasicAlgo(String s, int K) {

        /*
         time.................O(N)............................
         */
        System.out.println("string size: " + s.length());
        int n = s.length();
        int start = 0;
        int end = 0;
        List<String> subString = new ArrayList<>();
        while (end < n) {

//            System.out.println(start +" : "+end);
            if (end - start + 1 < K) {
                end++;
            } else if (end - start + 1 == K) {
                subString.add(s.substring(start, end + 1));
                //logic changes here as required
//                System.out.println(" K "+(end - start + 1));
                System.out.println("Sub string of given size K : " + subString);
//                and incr start bracket
                start++;
                //and incr end bracket
                end++;

            }

        }

    }

    public static void slidingWindowMaxSumSubArrayOfWindowSizeK(int[] a, int K) {

        int n = a.length;
        int start = 0;
        int end = 0;
        int maxSum = Integer.MIN_VALUE;
        int sum = 0;
        while (end < n) {

            sum += a[end];
            if (end - start + 1 < K) {
                end++;
            } else if (end - start + 1 == K) {
                maxSum = Math.max(maxSum, sum);
                sum -= a[start];
                start++;
                end++;
            }

        }

        System.out.println("Max sum of sub array with fixed size K : " + maxSum);

    }

    public static void slidingWindowMaximum_ON2(int[] nums, int k) {

        //this is O(N^2) approach can give you TLE
        ArrayList<Integer> result = new ArrayList<>();

        if (k == 1 || nums.length == 1) {
            System.out.print(nums[0]);

        }

        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(
                (a, b) -> b.compareTo(a)
        );
        for (int i = 0; i <= nums.length - k; i++) {

            for (int j = i; j < i + k; j++) {
                // System.out.println(nums[j]);
                maxHeap.add(nums[j]);

            }

            result.add(maxHeap.poll());
            maxHeap.clear();

        }

        result.stream().forEach((l) -> {
            System.out.print(l + " ");
        });

        System.out.println();

    }

    public static void slidingWindowMaximum_ON(int[] nums, int k) {

        //this is O(N) approach
        // max_upto array stores the index 
        // upto which the maximum element is a[i] 
        // i.e. max(a[i], a[i + 1], ... a[max_upto[i]]) = a[i] 
        int n = nums.length;
        int[] max_upto = new int[n];

        // Update max_upto array similar to 
        // finding next greater element 
        Stack<Integer> s = new Stack<>();
        s.push(0);
        for (int i = 1; i < n; i++) {
            while (!s.empty() && nums[s.peek()] < nums[i]) {
                max_upto[s.peek()] = i - 1;
                s.pop();
            }
            s.push(i);
        }

        while (!s.empty()) {
            max_upto[s.peek()] = n - 1;
            s.pop();
        }

        int j = 0;
        for (int i = 0; i <= n - k; i++) {

            // j < i is to check whether the 
            // jth element is outside the window 
            while (j < i || max_upto[j] < i + k - 1) {
                j++;
            }
            System.out.print(nums[j] + " ");
        }
        System.out.println();
    }

    private static void allTreePathRootToLeafWithWithGivenSum_Recursion(TreeNode<Integer> node, int K,
            List<Integer> curr, ArrayList<List<Integer>> result) {

        if (node == null) {
            return;
        }

        curr.add(node.getData());
        if (K - node.getData() == 0 && node.getLeft() == null && node.getRight() == null) {
            result.add(curr);
            return;
        }

        allTreePathRootToLeafWithWithGivenSum_Recursion(node.getLeft(), K - node.getData(),
                new ArrayList<>(curr), result);

        allTreePathRootToLeafWithWithGivenSum_Recursion(node.getRight(), K - node.getData(),
                new ArrayList<>(curr), result);

    }

    public static void allTreePathRootToLeafWithWithGivenSum(TreeNode<Integer> node, int K) {

        ArrayList<List<Integer>> result = new ArrayList<>();
        allTreePathRootToLeafWithWithGivenSum_Recursion(node, K, new ArrayList<>(), result);

        System.out.println(result);

    }

    private static void goDeepInTreeUptoKDistance(TreeNode<Integer> root, int K, Set<Integer> result) {

        if (root == null) {
            return;
        }
        if (K == 0) {
            result.add(root.getData());
            return;
        }
        goDeepInTreeUptoKDistance(root.getLeft(), K - 1, result);
        goDeepInTreeUptoKDistance(root.getRight(), K - 1, result);

    }

    private static int binaryTreeNodeAtKDistanceFromTarget_Recursion(TreeNode<Integer> root,
            int target, int K, Set<Integer> result) {

        if (root == null) {
            return -1;
        }

        if (root.getData() == target) {
            goDeepInTreeUptoKDistance(root, K, result);
            return 0;

        }

        int distLeft = binaryTreeNodeAtKDistanceFromTarget_Recursion(root.getLeft(),
                target, K, result);

        if (distLeft != -1) {

            //if K == 1
            if (distLeft + 1 == K) {

                result.add(root.getData());

            } else {
                goDeepInTreeUptoKDistance(root.getRight(), K - distLeft - 2, result);
            }
            return distLeft + 1;
        }

        int distRight = binaryTreeNodeAtKDistanceFromTarget_Recursion(root.getLeft(),
                target, K, result);

        if (distRight != -1) {

            //if K == 1
            if (distRight + 1 == K) {

                result.add(root.getData());

            } else {
                goDeepInTreeUptoKDistance(root.getLeft(), K - distRight - 2, result);
            }
            return distRight + 1;
        }

        return -1;

    }

    public static void binaryTreeNodeAtKDistanceFromTarget(TreeNode<Integer> root, int target, int K) {

        Set<Integer> result = new HashSet<>();
        binaryTreeNodeAtKDistanceFromTarget_Recursion(root, target, K, result);

        System.out.println(result);

    }

    public static void findXInSortedMatrix(int[][] mat, int X) {

        /*
        
         Complexity Analysis:
         Time Complexity: O(n).
         Only one traversal is needed, i.e, i from 0 to n and j from n-1 to 0 with at most 2*n steps.
         The above approach will also work for m x n matrix (not only for n x n). Complexity would be O(m + n).
         Space Complexity: O(1).
         No extra space is required.
        
         */
        int m = mat.length;
        int i = 0;
        int n = mat[i].length;
        int j = n - 1;
        while (i < m && j >= 0) {

            if (mat[i][j] == X) {
                System.out.println("Found " + i + ":" + j);
                return;
            } else if (X < mat[i][j]) {
                j--;
            } else {
                i++;
            }

        }

        System.out.println("Not Found");

    }

    public static void binaryTreeFlattening(TreeNode root) {

        /*
        
         ArrayDeque in Java provides a way to apply resizable-array in addition to the implementation of 
         the Deque interface. It is also known as Array Double Ended Queue or Array Deck. 
         This is a special kind of array that grows and allows users to add or remove an element 
         from both sides of the queue. Few important features of ArrayDeque are as follows:

         Array deques have no capacity restrictions and they grow as necessary to support usage.
         They are not thread-safe which means that in the absence of external synchronization, ArrayDeque does not 
         support concurrent access by multiple threads.
         Null elements are prohibited in the ArrayDeque.
         ArrayDeque class is likely to be faster than Stack when used as a stack.
         ArrayDeque class is likely to be faster than LinkedList when used as a queue
        
         */
        if (root == null) {
            return;
        }

        Deque<TreeNode> q = new ArrayDeque<>();
        q.addFirst(root);
        while (!q.isEmpty()) {
            TreeNode curr = q.removeFirst(); //pop first from the queue

            if (curr.getRight() != null) {
                q.addFirst(curr.getRight()); //push at first in queue
            }
            if (curr.getLeft() != null) {
                q.addFirst(curr.getLeft()); //push at first in queue
            }
//            System.out.println(q);
            if (!q.isEmpty()) {
//                System.out.println(q.peek()+"--");
                curr.setRight(q.peek());
                curr.setLeft(null);
            }
        }

        BinaryTree<Integer> bt = new BinaryTree<>(root);
        bt.treeBFS();
        System.out.println();

    }

    public static String breakPalindrome(String pallindrome) {

        /*
        
         Given a palindromic string palindrome, replace exactly one character by any lowercase English letter 
         so that the string becomes the lexicographically smallest possible string that isn't a palindrome.

         After doing so, return the final string.  If there is no way to do so, return the empty string.
        
         Example 1:

         Input: palindrome = "abccba"
         Output: "aaccba"
         Example 2:

         Input: palindrome = "a"
         Output: ""
        
         */
        if (pallindrome == null || pallindrome.length() == 0 || pallindrome.length() == 1) {
            return "";
        }

        char[] ch = pallindrome.toCharArray();

        int firstNonA = -1;
        for (int i = 0; i < ch.length; i++) {
            if (ch[i] != 'a') {
                firstNonA = i;
                char org = ch[i];
                ch[i] = (char) (ch[i] + 1);
                String check = String.valueOf(ch);
                if (check.equals(new StringBuilder(check).reverse().toString())) {
                    firstNonA = -1;
                    ch[i] = org;
                    continue;
                }
                break;
            }
        }

        if (firstNonA != -1) {
            ch[firstNonA] = 'a';
        } else {
            ch[ch.length - 1] = 'b';
        }

        return String.valueOf(ch);

    }

    public static String replaceWords(List<String> dictionary, String sentence) {

        /*
        
         n English, we have a concept called root, which can be followed by some other 
         word to form another longer word - let's call this word successor. For example, 
         when the root "an" is followed by the successor word "other", we can form a new word "another".
         Given a dictionary consisting of many roots and a sentence consisting of 
         words separated by spaces, replace all the successors in the sentence with the root 
         forming it. If a successor can be replaced by more than one root, replace it with the root 
         that has the shortest length.
         Return the sentence after the replacement.
        
         Input: dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
         Output: "the cat was rat by the bat"
        
         */
        Set<String> set = dictionary.stream().collect(Collectors.toSet());

        StringBuilder sb = new StringBuilder();
        boolean isDone = false;
        for (String w : sentence.split(" ")) {
            isDone = false;
            for (int i = 1; i < w.length(); i++) {
                String prefix = w.substring(0, i);
                if (set.contains(prefix)) {
                    sb.append(prefix + " ");
                    isDone = true;
                    break;
                }
            }

            if (!isDone) {
                sb.append(w + " ");
            }

        }

        return sb.toString().trim();

    }

    public static void continousSubArrayWhoseSumIsK(int[] a, int K) {

        /*
        
         ...............O(N^2)....................
         brute force approach will be to try all the possible sub arrays that form
         the given sum
         this can be achieved using 2 nested fo loops
        
         loop: i=0 -> i<n -> i++
         sum = a[i]
         loop: j = i+1 -> i<n -> j++
         if: sum == K
         int end = j - 1
         print: "subarray indexes" + i +" to "+ end
         return
                
         if: sum > K
         break
                
         sum += a[j]
        
         the aabove algo is of O(N^2)
         reason for finding end variable is because we adding into the sum variable a[j]
         in the last of loop 2, so if aadding a[j] makes sum == K that can only be checked 
         in next iteration of j where j got already incremented so balancing the last element added was at j-1 index 
       
         */
        //................O(N)...........................................
        int n = a.length;
        int start = 0;
        int sum = a[0];
        for (int i = 1; i <= n; i++) {

            while (sum > K && start < i - 1) {
                sum -= a[start];
                start++;
            }

            if (sum == K) {
                int end = i - 1;
                System.out.println("subarray index " + start + " to " + end);
                return;
            }

            if (i < n) {
                sum += a[i];
            }

        }

        System.out.println("no subarray found");

    }

    public static void continousSubArrayWhoseSumIsK_HandlesNegtiveNum(int[] a, int K) {

        //10, 2, -2, -20, 10 | K = -10
        int n = a.length;
        int start = 0;
        int end = -1;
        int sum = 0;
        Map<Integer, Integer> m = new HashMap<>();
        for (int i = 0; i < n; i++) {
            sum += a[i];
            if (sum - K == 0) {
                start = 0;
                end = i;
                break;
            }

            if (m.containsKey(sum - K)) {
                start = m.get(sum - K) + 1;
                end = 1;
                break;
            }

            m.put(sum, i);

        }

        // if end is -1 : means we have reached end without the sum 
        if (end == -1) {
            System.out.println("No subarray with given sum exists");
        } else {
            System.out.println("Sum found between indexes "
                    + start + " to " + end);
        }
    }

    public static void findOddOccurringNumber_UsingHashing(int[] a) {

        //...................O(N)..............................
        Map<Integer, Integer> map = new HashMap<>();
        for (int x : a) {
            map.put(x, map.getOrDefault(x, 0) + 1);
        }

        for (Map.Entry<Integer, Integer> e : map.entrySet()) {
            if (e.getValue() % 2 == 1) {
                System.out.println("odd occuerence hashing: " + e.getKey());
                return;
            }
        }

        System.out.println("odd occuerence hashing: " + 0);
    }

    public static void findOddOccurringNumber_UsingXOR(int[] a) {
        //..............O(N)..................................
        int xor = 0;
        for (int x : a) {
            xor = xor ^ x;
        }

        //odd occurred element will be in the xor
        System.out.println("odd occuerence xor: " + xor);
    }

    public static boolean isRectangle(int[][] a) {

        int row = a.length;
        int col = a[0].length;
        Map<Integer, Set<Integer>> cap = new HashMap<>();
        for (int x = 0; x < row; x++) {
            for (int y = 0; y < col; y++) {
                for (int k = y + 1; k < col; k++) {

                    if (a[x][y] == 1 && a[x][k] == 1) {
//                        System.out.println("(" + x + "," + y + ")-(" + x + "," + k + ")");

                        if (cap.containsKey(y) && cap.get(y).contains(k)) {
                            return true;
                        }

                        if (cap.containsKey(k) && cap.get(k).contains(y)) {
                            return true;
                        }

                        if (!cap.containsKey(y)) {
                            Set<Integer> s = new HashSet<>();
                            s.add(k);
                            cap.put(y, s);
                        } else {
                            cap.get(y).add(k);
                        }

                        if (!cap.containsKey(k)) {
                            Set<Integer> s = new HashSet<>();
                            s.add(y);
                            cap.put(k, s);
                        } else {
                            cap.get(k).add(y);
                        }

                    }

                }
            }
        }

        return false;

    }

    public static void removeDuplicateFromLinkedList_Iterative(Node<Integer> head) {

        /*
        
         removes duplicate irrespective of linked list is sorted or unsorted
         It kind of maintains insertion order while removing consecutive duplicates
        
         iterative keeps the discontinous occurence of node in the list and 
         removes all dupliacate elements that occured consecutively 
        
         */
        Node<Integer> cur = head;
        Node<Integer> temp = head.getNext();

        while (temp != null) {
            if (cur.getData() != temp.getData()) {
                cur.setNext(temp);
                cur = temp;
            }

            temp = temp.getNext();

        }

        //if duplicates are in the end segments of the list
        cur.setNext(temp);
        System.out.println("remove duplicate using iterative");
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(head);
        ll.print();
    }

    public static void removeDuplicateFromLinkedList_Hashing(Node<Integer> head) {

        /*
        
         removes duplicate irrespective of linked list is sorted or unsorted
        
         hashing keeps the first occuring node in the list and 
         removes all other dupliacate elements that occured anywhere in the list later on 
        
         */
        HashSet<Integer> set = new HashSet<>();
        Node<Integer> cur = head;
        Node<Integer> prev = head;
        while (cur != null) {

            if (set.contains(cur.getData())) {
                prev.setNext(cur.getNext());
            } else {
                set.add(cur.getData());
                prev = cur;
            }

            cur = cur.getNext();

        }

        System.out.println("remove duplicate using hashing");
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(head);
        ll.print();

    }

    public static TreeNode<Integer> removeNodeFromBSTNotInRange(TreeNode<Integer> node, int min, int max) {

        if (node == null) {
            return null;
        }

        node.setLeft(removeNodeFromBSTNotInRange(node.getLeft(), min, max));
        node.setRight(removeNodeFromBSTNotInRange(node.getRight(), min, max));

        if (node.getData() < min) {
            return node.getRight();
        } else if (node.getData() > max) {
            return node.getLeft();
        }

        return node;

    }

    public static void detectLoopInLinkedList_Hashing(Node<Integer> node) {

        /*
        
         time: ......................O(N)..............................
         space: ....................O(N)...............................
        
         */
        HashSet<Node<Integer>> set = new HashSet<>();
        Node temp = node;
        while (temp != null) {

            //if temp node's next is a linked ref to some internal previous node of the same 
            //linked list then this cond forms cycle
            if (set.contains(temp)) {
                System.out.println("Cycle found at " + temp.getData());
                return;
            }

            set.add(temp);

            temp = temp.getNext();

        }

        //no cycle found
        System.out.println("No cycle found");

    }

    public static void detectLoopInLinkedList_Iterative(Node<Integer> node) {

        /*
        
         time: ......................O(N)..............................
         space: ....................O(1)...............................
        
         Also the floyd's warshall algo adaptation to find the the cycle using the concept of 
         slow and fast pointers
        
         */
        Node slow = node;
        Node fast = node.getNext().getNext();
        while (slow != null && fast != null && fast.getNext() != null) {

            //debugg
//            System.out.println("Slow: "+slow.getData()+" fast: "+fast.getData());
            if (slow == fast) {
                break;
            }

            slow = slow.getNext();
            fast = fast.getNext().getNext();

        }
        
        /* If loop exists */
        if (slow == fast) {
            slow = node;
            while (slow != fast.getNext()) {
                slow = slow.getNext();
                fast = fast.getNext();
            }

            /* since fast->next is the looping point */
            System.out.println("Cycle found at " + fast.getNext().getData());
            return;
        }

        //no cycle found
        System.out.println("No cycle found");

    }

    public static void detectAndRemovingLoopInLinkedList_UsingHashing(Node<Integer> node) {

        /*
        
         time: ......................O(N)..............................
         space: ....................O(N)...............................
        
         Using hashing
        
         */
        HashSet<Node<Integer>> set = new HashSet<>();
        Node<Integer> prev = null;
        Node<Integer> temp = node;
        while (temp != null) {

            if (set.contains(temp)) {
                prev.setNext(null);
                break;
            }

            set.add(temp);
            prev = temp;

            temp = temp.getNext();

        }

        System.out.println("Removing cycle using hashing");
        new LinkedListUtil<Integer>(node).print();

    }

    public static void detectAndRemovingLoopInLinkedList_UsingIteration(Node<Integer> node) {

        /*
        
         time: ......................O(N)..............................
         space: ....................O(1)...............................
        
         Using floyd warshal's adaptation by maintaing the 2 pointers
        
         */
        // If list is empty or has only one node 
        // without loop 
        if (node == null || node.getNext() == null) {
            return;
        }

        // Move slow and fast 1 and 2 steps 
        // ahead respectively. 
        Node slow = node;
        Node fast = node.getNext();
//        System.out.println(slow.getData() + "#" + fast.getData());
        // Search for loop using slow and fast pointers 
        while (slow != null && fast != null && fast.getNext() != null) {
            if (slow == fast) {
                break;
            }

            slow = slow.getNext();
            fast = fast.getNext().getNext();
//            System.out.println(slow.getData() + "~" + fast.getData());
        }

        /* If loop exists */
        if (slow == fast) {
            slow = node;
//            System.out.println(slow.getData()+"|"+fast.getData());
            while (slow != fast.getNext()) {
                slow = slow.getNext();
                fast = fast.getNext();
//                System.out.println(slow.getData()+"-"+fast.getData());
            }

            /* since fast->next is the looping point */
            fast.setNext(null); /* remove loop */

        }

        System.out.println("Removing cycle using iterative");
        new LinkedListUtil<Integer>(node).print();

    }

    public static void sortAListInRelativeOrderOfBList(int[] a, int[] b) {

        //treemap is sorted in order of keys
        Map<Integer, Integer> map = new TreeMap<>();
        int[] result = new int[a.length];
        int k = 0;

        //O(N) loop N = a.length
        //getting just the occurence of the element from a list
        for (int x : a) {
            map.put(x, map.getOrDefault(x, 0) + 1);
        }

        //O(M) loop M = b.length
        //relative order of b list
        for (int x : b) {
            //inner loops depends upon the occuernce of each elements
            //x (common to both a list and b list)
            for (int i = 0; i < map.get(x); i++) {
                result[k++] = x;
            }
            //once we process the x in b list for its relative order
            //we will remove it from the map
            //so that we are left with those elements in map that are not in b list
            map.remove(x);
        }

        //those elements not in b list 
        //append them in result in sorted order
        //O(log(N-M)) getting out of the treemap
        for (Map.Entry<Integer, Integer> e : map.entrySet()) {
            int x = e.getKey();
            //the left over elements, they can be occurrring more than one then 
            //in result we have to generate that situations also
            for (int i = 0; i < e.getValue(); i++) {
                result[k++] = x;
            }
        }

        //output
        for (int x : result) {
            System.out.print(x + " ");
        }

        System.out.println();

    }
    
    public static Node removeNthFromEnd(Node head, int n) {
        
        if(head == null || head.getNext() == null && n == 1){
            return null;
        }
        
        Node ref = head;
        Node main = head;
        Node prevMain = null;
        while(n-- != 0){
            ref = ref.getNext();
            if(ref == null){
                ref = head;
            }
            
        }
        
        if(ref == main){
            return head = head.getNext();
        }
        
        while(ref != null){
            prevMain = main;
            main = main.getNext();
            ref = ref.getNext();
        }
        
        prevMain.setNext(main.getNext());
        
//         System.out.println(prevMain.val);
        return head;
        
    }

    public static boolean childrenSumPropertyOfTree(TreeNode<Integer> node) {

        /* left_data is left child data and right_data is for right  
         child data*/
        int leftData = 0, rightData = 0;

        /* If node is NULL or it's a leaf node then 
         return true */
        if (node == null || (node.getLeft() == null && node.getRight() == null)) {
            return true;
        }

        /* If left child is not present then 0 is used 
         as data of left child */
        if (node.getLeft() != null) {
            leftData = (int) node.getLeft().getData();
        }

        /* If right child is not present then 0 is used 
         as data of right child */
        if (node.getRight() != null) {
            rightData = (int) node.getRight().getData();
        }

        /* if the node and both of its children satisfy the 
         property return 1 else 0*/
        if (node.getData() == leftData + rightData
                && childrenSumPropertyOfTree(node.getLeft())
                && childrenSumPropertyOfTree(node.getRight())) {
            return true;
        } else {
            return false;
        }
    }

    public static int buyAndSellStockMaxProfit(int[] prices) {

        if (prices == null || prices.length == 0) {
            return 0;
        }

        int minprice = Integer.MAX_VALUE;
        int maxprofit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minprice) {
                minprice = prices[i];
            } else if (prices[i] - minprice > maxprofit) {
                maxprofit = prices[i] - minprice;
            }
        }
        return maxprofit;

    }

    static class Interval {

        int buy, sell;
    }

    public static void stockBuySellTime(int price[], int n) {
        // Prices must be given for at least two days 
        if (n == 1) {
            return;
        }

        int count = 0;

        // solution array 
        ArrayList<Interval> sol = new ArrayList<>();

        // Traverse through given price array 
        int i = 0;
        while (i < n - 1) {
            // Find Local Minima. Note that the limit is (n-2) as we are 
            // comparing present element to the next element. 
            while ((i < n - 1) && (price[i + 1] <= price[i])) {
                i++;
            }

            // If we reached the end, break as no further solution possible 
            if (i == n - 1) {
                break;
            }

            Interval e = new Interval();
            e.buy = i++;
            // Store the index of minima 

            // Find Local Maxima.  Note that the limit is (n-1) as we are 
            // comparing to previous element 
            while ((i < n) && (price[i] >= price[i - 1])) {
                i++;
            }

            // Store the index of maxima 
            e.sell = i - 1;
            sol.add(e);

            // Increment number of buy/sell 
            count++;
        }

        // print solution 
        if (count == 0) {
            System.out.println("There is no day when buying the stock "
                    + "will make profit");
        } else {
            for (int j = 0; j < count; j++) {
                System.out.println("Buy on day: " + sol.get(j).buy
                        + "        "
                        + "Sell on day : " + sol.get(j).sell);
            }
        }

    }

    public static boolean parenthesisPairChecker(String expr) {

        if (expr == null || expr.equals("")) {
            return false;
        }

        Stack<Character> s = new Stack<>();
        for (char c : expr.toCharArray()) {

            if (c == '(' || c == '{' || c == '[') {
                s.push(c);
                continue;
            }

            if (s.isEmpty()) {
                return false;
            }

            switch (c) {
                case ')':
                    if (s.pop() != '(') {
                        return false;
                    }
                    break;
                case '}':
                    if (s.pop() != '{') {
                        return false;
                    }
                    break;
                case ']':
                    if (s.pop() != '[') {
                        return false;
                    }
                    break;
            }

        }
        return s.isEmpty();
    }

    public static int longestValidParentheses(String s) {

        int len = 0;

        Stack<Integer> st = new Stack<>();
        st.push(-1);
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '(') {
                st.push(i);
            } else {
                st.pop();
                if (st.isEmpty()) {
                    st.push(i);
                } else {
                    len = Math.max(len, i - st.peek());
                }
            }
        }
        return len;
    }

    public static boolean checkValidString(String s) {

        /*
        
         greedy approach
         Intuition

         When checking whether the string is valid, we only cared about the "balance": the number of extra, 
         open left brackets as we parsed through the string. 
         For example, when checking whether '(()())' is valid, we had a balance of 1, 2, 1, 2, 1, 0 as 
         we parse through the string: '(' has 1 left bracket, '((' has 2, '(()' has 1, and so on. 
         This means that after parsing the first i symbols, (which may include asterisks,) we only need to keep 
         track of what the balance could be.

         For example, if we have string '(***)', then as we parse each symbol, 
         the set of possible values for the balance is [1] for '('; 
         [0, 1, 2] for '(*'; [0, 1, 2, 3] for '(**'; [0, 1, 2, 3, 4] for '(***', and [0, 1, 2, 3] for '(***)'.

         Furthermore, we can prove these states always form a contiguous interval. 
         Thus, we only need to know the left and right bounds of this interval. 
         That is, we would keep those intermediate states described above as 
         [lo, hi] = [1, 1], [0, 2], [0, 3], [0, 4], [0, 3].
        
         */
        int lo = 0, hi = 0;
        for (char c : s.toCharArray()) {
            lo += c == '(' ? 1 : -1;
            hi += c != ')' ? 1 : -1;
            if (hi < 0) {
                break;
            }
            lo = Math.max(lo, 0);
        }
        return lo == 0;
    }

    public static String smallestWindowContainingAllCharOfString(String str) {
        int n = str.length();

        // Count all distinct characters. 
        int dist_count = 0;
        Map<Character, Integer> distCountMap = new HashMap<>();
        for (char c : str.toCharArray()) {
            distCountMap.put(c, distCountMap.getOrDefault(c, 0) + 1);
        }

        dist_count = distCountMap.size();

        // Now follow the algorithm discussed in below 
        // post. We basically maintain a window of characters 
        // that contains all characters of given string. 
        int start = 0, start_index = -1;
        int min_len = Integer.MAX_VALUE;

        int count = 0;
        Map<Character, Integer> currCountMap = new HashMap<>();
        for (int j = 0; j < n; j++) {
            // Count occurrence of characters of string
            currCountMap.put(str.charAt(j),
                    currCountMap.getOrDefault(str.charAt(j), 0) + 1);

            if (currCountMap.get(str.charAt(j)) == 1) {
                count++;
            }

            // if all the characters are matched 
            if (count == dist_count) {
                // Try to minimize the window i.e., check if 
                // any character is occurring more no. of times 
                // than its occurrence in pattern, if yes 
                // then remove it from starting and also remove 
                // the useless characters. 
                while (currCountMap.get(str.charAt(start)) > 1) {
                    if (currCountMap.get(str.charAt(start)) > 1) {
                        currCountMap.put(str.charAt(start),
                                currCountMap.get(str.charAt(start)) - 1);
                    }
                    start++;
                }

                // Update window size 
                int len_window = j - start + 1;
                if (min_len > len_window) {
                    min_len = len_window;
                    start_index = start;
                }
            }
        }
        // Return substring starting from start_index 
        // and length min_len 
        return str.substring(start_index, start_index + min_len);
    }

    private static int peekElementHelper(int[] arr, int low, int high, int n) {

        // Find index of middle element 
        // (low + high)/2 
        int mid = low + (high - low) / 2;

        // Compare middle element with its 
        // neighbours (if neighbours exist) 
        if ((mid == 0 || arr[mid - 1] <= arr[mid])
                && (mid == n - 1 || arr[mid + 1] <= arr[mid])) {
            return mid;
        } // If middle element is not peak 
        // and its left neighbor is 
        // greater than it, then left half 
        // must have a peak element 
        else if (mid > 0 && arr[mid - 1] > arr[mid]) {
            return peekElementHelper(arr, low, (mid - 1), n);
        } // If middle element is not peak 
        // and its right neighbor 
        // is greater than it, then right 
        // half must have a peak 
        // element 
        else {
            return peekElementHelper(arr, (mid + 1), high, n);
        }

    }

    public static int peakElement_OLogN(int[] arr, int n) {
        return peekElementHelper(arr, 0, n, n);

    }

    public static void invertABinaryTree(TreeNode<Integer> node) {
        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(node);

        while (!q.isEmpty()) {

            TreeNode<Integer> t = q.poll();

            if (t.getLeft() != null) {
                q.add(t.getLeft());
            }

            if (t.getRight() != null) {
                q.add(t.getRight());
            }

            if (t.getLeft() != null && t.getRight() != null) {

                /*
                
                 below code invert only child of immediate node not full level, order wise
                
                 */
                int temp = (int) t.getRight().getData();
                t.getRight().setData(t.getLeft().getData());
                t.getLeft().setData(temp);

                /*
                 full invert level order wise
                 TreeNode temp = t.getRight();
                 t.setRight(t.getLeft());
                 t.setLeft(temp);
                
                 */
            } else if (t.getLeft() == null && t.getRight() != null) {
                t.setLeft(t.getRight());
                t.setRight(null);
            } else if (t.getLeft() != null && t.getRight() == null) {
                t.setRight(t.getLeft());
                t.setLeft(null);
            }

        }

        BinaryTree<Integer> bt = new BinaryTree<>(node);
        bt.treeBFS();

    }

    public static void countNoOfWords(String s) {

        final int IN = 1;
        final int OUT = 0;

        int state = OUT;
        int wordCount = 0;
        int i = 0;
        while (i < s.length()) {

            if (s.charAt(i) == ' ' || s.charAt(i) == '\t' || s.charAt(i) == '\n') {
                state = OUT;
            } else if (state == OUT) {
                state = IN;
                wordCount++;
            }

            i++;

        }

        System.out.println("actual word count " + wordCount);

    }

    public static void rotateAndDelete() {
        Scanner scan = new Scanner(System.in);
        System.out.println("Enter test case:");
        int t = Integer.parseInt(scan.nextLine());

        List<String> stringList = new ArrayList<>();
        String answer = "";

        while (t-- > 0) {

            System.out.println("Enter size of array:");
            int n = Integer.parseInt(scan.nextLine());
            System.out.println("Enter array (space seprated):");
            String[] sList = scan.nextLine().split(" ");

            stringList.clear();
            stringList.addAll(Arrays.asList(sList));

            int counter = 1;

            while (stringList.size() != 1) {

                Collections.rotate(stringList, 1);

                if (stringList.size() - counter < 0) {
                    counter = stringList.size();
                }

                stringList.remove(stringList.size() - counter);
                counter++;

            }
            answer += stringList.get(0) + "\n";

        }
        System.out.println("Output per test case:");
        System.out.println(answer);

    }

    private static void doInOrder(TreeNode node, List<Integer> inOrder) {

        if (node == null) {
            return;
        }

        doInOrder(node.getLeft(), inOrder);
        inOrder.add((int) node.getData());
        doInOrder(node.getRight(), inOrder);

    }

    public static void bstToGreaterSumTree(TreeNode<Integer> root) {

        /*
         Given the root of a Binary Search Tree (BST), convert it to a 
         Greater Tree such that every key of the original 
         BST is changed to the original key plus sum of all 
         keys greater than the original key in BST.
         */
        List<Integer> inOrder = new ArrayList<>();

        //get inorder of the given BST
        doInOrder(root, inOrder);

        //get the sum of all the node which is greater than the node itself
        //and stored as map
        int sum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inOrder.size(); i++) {
            sum = 0;
            for (int j = i; j < inOrder.size(); j++) {
                sum += inOrder.get(j);
            }

            map.put(inOrder.get(i), sum);
        }

        //do a level order traversal on BST 
        // and for every node find the greater sum from the map
        //and replace the value for that node
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        while (!q.isEmpty()) {

            TreeNode t = q.poll();
            t.setData(map.get(t.getData()));

            if (t.getLeft() != null) {
                q.add(t.getLeft());
            }

            if (t.getRight() != null) {
                q.add(t.getRight());
            }

        }

        System.out.println();
        //print greater sun tree (which may not be a BST after all conversions)
        BinaryTree<Integer> bt = new BinaryTree<>(root);
        bt.treeBFS();

    }

    public static int minDifferenceBetweenTwoPairOfElement_ONLogN(int[] a) {

        /*
         ............Brute force approach O(N^2).....................
         // Initialize difference as infinite 
         int diff = Integer.MAX_VALUE; 
      
         // Find the min diff by comparing difference 
         // of all possible pairs in given array 
         for (int i=0; i<n-1; i++) {
         for (int j=i+1; j<n; j++) {
         if (Math.abs((arr[i] - arr[j]) )< diff){ 
         diff = Math.abs((arr[i] - arr[j]));
         }   
         }
         }
         // Return min diff     
         return diff;
        
         */
        //sort the array
        //O(LogN)
        Arrays.sort(a);

        //window based traversing login O(N)
        int start = 0;
        int end = 0;
        int k = 2; //window is fixed to Two pair of element
        int minDiff = Integer.MAX_VALUE;
        while (end < a.length) {

            if (end - start + 1 < k) {
                end++;
            } else if (end - start + 1 == k) {
                minDiff = Math.min(minDiff, a[end] - a[start]);
                start++;
                end++;
            }

        }

        return minDiff;
    }

    public static void flipAndInvertImage(int[][] A) {

        int row = A.length;
        int col = A[0].length;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < (col + 1) / 2; j++) {

                //horizontal flip
                int temp = A[i][j] == 0 ? 1 : 0; //inverse first element
                A[i][j] = A[i][col - j - 1] == 0 ? 1 : 0; //inverse second element
                A[i][col - j - 1] = temp;
                //element are inversed and swaped at the same time

            }

        }

        //printing
        for (int[] r : A) {
            for (int c : r) {
                System.out.print(c + "\t");
            }
            System.out.println();
        }

    }

    public static void zigZagLinkedList(Node<Integer> node) {

        //actual Linkedlist
        System.out.println("input:");
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(node);
        ll.print();

        boolean flag = true;
        int temp = 0;
        Node<Integer> curr = node;
        while (curr != null && curr.getNext() != null) {

            if (flag) {

                if ((int) curr.getData() > (int) curr.getNext().getData()) {
                    temp = curr.getData();
                    curr.setData((int) curr.getNext().getData());
                    curr.getNext().setData(temp);
                }

            } else {

                if ((int) curr.getData() < (int) curr.getNext().getData()) {
                    temp = curr.getData();
                    curr.setData((int) curr.getNext().getData());
                    curr.getNext().setData(temp);
                }

            }

            curr = curr.getNext();
            flag = !flag;

        }

        //converted node
        System.out.println("output:");
        ll = new LinkedListUtil<>(node);
        ll.print();

    }

    private static void sumOfOnlyLeftChildOfNodeInTree_Helper(TreeNode<Integer> node,
            TreeNode<Integer> parent) {

        if (node == null) {
            return;
        }

//        System.out.println(sumOfOnlyLeftChildOfNodeInTree+"--"+node.getData()+"--"+(parent == null ? "Null" : parent.getData()));
        sumOfOnlyLeftChildOfNodeInTree_Helper(node.getLeft(), node);
        if (parent != null && node == parent.getLeft()) {
            sumOfOnlyLeftChildOfNodeInTree += node.getData();
        }

        sumOfOnlyLeftChildOfNodeInTree_Helper(node.getRight(), node);

    }
    private static int sumOfOnlyLeftChildOfNodeInTree = 0;

    public static void sumOfOnlyLeftChildOfNodeInTree(TreeNode<Integer> node) {
        sumOfOnlyLeftChildOfNodeInTree_Helper(node, null);
        System.out.println("Sum of all the left childs of any node in the tree: " + sumOfOnlyLeftChildOfNodeInTree);
        //resetting static variabe
        sumOfOnlyLeftChildOfNodeInTree = 0;
    }

    private static void sumOfLeftLeafNodesOfTree_Helper(TreeNode<Integer> root, TreeNode<Integer> parent) {

        if (root == null) {
            return;
        }

        sumOfLeftLeafNodesOfTree_Helper(root.getLeft(), root);

        if (root.getLeft() == null && root.getRight() == null
                && parent != null && parent.getLeft() == root) {
            sumOfOnlyLeftChildOfNodeInTree += root.getData();
        }

        sumOfLeftLeafNodesOfTree_Helper(root.getRight(), root);

    }

    public static void sumOfLeftLeafNodesOfTree(TreeNode<Integer> root) {
        sumOfLeftLeafNodesOfTree_Helper(root, null);
        System.out.println("Sum of left-leaf nodes in the tree: " + sumOfOnlyLeftChildOfNodeInTree);
        //resetting static variabe
        sumOfOnlyLeftChildOfNodeInTree = 0;

    }

    public static boolean wordBreak(Set<String> dictionary, String word) {
        int size = word.length();

        // base case 
        if (size == 0) {
            return true;
        }

        //else check for all words 
        for (int i = 1; i <= size; i++) {
            // Now we will first divide the word into two parts , 
            // the prefix will have a length of i and check if it is  
            // present in dictionary ,if yes then we will check for  
            // suffix of length size-i recursively. if both prefix and  
            // suffix are present the word is found in dictionary. 

            if (dictionary.contains(word.substring(0, i))
                    && wordBreak(dictionary, word.substring(i, size))) {
                return true;
            }
        }

        // if all cases failed then return false 
        return false;
    }

    public static int commonPrefix_Helper(String str, String suffix, int len1, int len2) {

        if (len1 >= str.length() || len2 >= suffix.length()) {
            return 0;
        }

        if (str.charAt(len1) == suffix.charAt(len2)) {
            return commonPrefix_Helper(str, suffix, len1 + 1, len2 + 1) + 1;
        } else {
            return 0;
        }

    }

    public static List<Integer> commonPrefix(List<String> inputs) {

        //Hackerrank test for Akash inst.
        /*
    
         Given a string, split the string into two substrings at every possible point. 
         The rightmost substring is a suffix The beginning of the string is the prefix 
         Determine the lengths of the common prefix between each suffix and the original 
         string. Sum and return the lengths of the common prefixes. Return an array where 
         each element i is the sum for string i
        
         ex String s = "abcabcd"
        
         //every suffix in the below suffix column is to be compared with actual String s
         //such that suffix and s should have common perfix in them
         //take this suffix = 'abcd' s = 'abcabcd' common prefix = 'abc' length = 3
         //add all those lengths in which we can find common prefix
        
         //Remove to leave suffix |   Suffix |    Common Prefix |     Length
         //''                         'abcabcd'	'abcabcd'       	7
         //'a'                        'bcabcd'	''                      0
         //'ab'                       'cabcd'	''                      0
         //'abc'                      'abcd'	'abc'                   3
         //'abca'                     'bcd'	''                      0
         //'abcab'                    'cd'	''                      0
         //'abcabc'                   'd'         ''                    0
        
         
         another ex s = "ababaa"
         The suffixes are ['ababaa', 'babaa', 'abaa', 'baa', 'aa', 'a']. 
         The common prefix lengths of each of these suffixes with 
         the original string are [6, 0, 3, 0, 1, 1] respectively, 
         and they sum to 11.
        
         */
        List< Integer> result = new ArrayList<>();
        int sum = 0;
        for (String str : inputs) {

            int len = str.length();
            sum = 0;
            for (int i = 0; i <= len - 1; i++) {

                String suffix = str.substring(i, len);
                // System.out.println(suffix);
                sum += commonPrefix_Helper(str, suffix, 0, 0);

            }

            result.add(sum);

        }

        return result;

    }

    private static TreeNode<Integer> bstFromPreOrderArray_Helper(int[] preorder, int start, int end) {

        if (start > end) {
            return null;
        }

        TreeNode<Integer> root = new TreeNode<>(preorder[start]);

        int i;
        for (i = start + 1; i <= end; i++) {
            if (preorder[i] > root.getData()) {
                //find the index i of the first greater element than the 
                //root.data itself and break.
                break;
            }
        }

        root.setLeft(bstFromPreOrderArray_Helper(preorder, start + 1, i - 1));
        root.setRight(bstFromPreOrderArray_Helper(preorder, i, end));

        return root;

    }

    public static void bstFromPreOrderArray(int[] preorder) {

        /*
        
         In preorder of a BST the first element is always root of tree
         root = preorder[0]
         and the whole preorder can be divided into two parts
         elements less than preorder[0] are left sub tree of root
         elements greater than preorder[0] are right sun tree of root
        
         the sub problem is also repeated in left and right sub tree respect.
        
         */
        TreeNode<Integer> root = bstFromPreOrderArray_Helper(preorder, 0, preorder.length - 1);
        BinaryTree<Integer> bt = new BinaryTree<>(root);
        bt.treeBFS();
        System.out.println();

    }

    private static TreeNode<Integer> bstFromPostOrderArray_Helper(int[] postorder, int start, int end) {

        if (start > end) {
            return null;
        }

        TreeNode<Integer> root = new TreeNode<>(postorder[end]);

        int i;
        for (i = end - 1; i >= start; i--) {
            if (postorder[i] < root.getData()) {
                //find the index i of the last greater element than the 
                //root.data itself and break.
                break;
            }
        }

        root.setLeft(bstFromPostOrderArray_Helper(postorder, start, i));
        root.setRight(bstFromPostOrderArray_Helper(postorder, i + 1, end - 1));

        return root;

    }

    public static void bstFromPostOrderArray(int[] postorder) {

        /*
        
         In postorder of a BST the last element is always root of tree
         root = postorder[end]
         and the whole postorder can be divided into two parts
         elements less than preorder[end] are left sub tree of root
         elements greater than preorder[end] are right sun tree of root
        
         the sub problem is also repeated in left and right sub tree respect.
        
         */
        TreeNode<Integer> root = bstFromPostOrderArray_Helper(postorder, 0, postorder.length - 1);
        BinaryTree<Integer> bt = new BinaryTree<>(root);
        bt.treeBFS();
        System.out.println();

    }

    static class Type {

        int min;
        int max;
        int val;

        Type(int min, int max, int val) {
            this.min = min;
            this.max = max;
            this.val = val;
        }
    }

    private static Type maxDiffBetweenNodeAndAncestor_Helper(TreeNode<Integer> node) {
        if (node == null) {
            return new Type(Integer.MAX_VALUE, Integer.MIN_VALUE, 0);
        }
        Type left = maxDiffBetweenNodeAndAncestor_Helper(node.getLeft());
        Type right = maxDiffBetweenNodeAndAncestor_Helper(node.getRight());
        int min = Math.min(Math.min(left.min, right.min), node.getData());
        int max = Math.max(Math.max(left.max, right.max), node.getData());
        int val = Math.max(Math.max(left.val, right.val),
                Math.max(Math.abs(max - node.getData()), Math.abs(min - node.getData())));
        return new Type(min, max, val);
    }

    public static int maxDiffBetweenNodeAndAncestor(TreeNode<Integer> root) {
        return maxDiffBetweenNodeAndAncestor_Helper(root).val;
    }

    public static void maxAreaOfHistogram(int[] histogram, int n) {

        //find next smaller element to right and next smaller element to left
        //Pair: key = arr element, value = arr index i of element
        Stack<Pair<Integer, Integer>> s = new Stack<>();
        List<Pair<Integer, Integer>> smallerRight = new ArrayList<>();
        List<Pair<Integer, Integer>> smallerLeft = new ArrayList<>();
        //next smaller element to right
        for (int i = n - 1; i >= 0; i--) {

            while (!s.isEmpty() && s.peek().getKey() > histogram[i]) {
                s.pop();
            }

            if (s.isEmpty()) {
                smallerRight.add(new Pair<>(-1, n));
            } else {
                smallerRight.add(new Pair<>(s.peek().getKey(), s.peek().getValue()));
            }

            s.push(new Pair<>(histogram[i], i));

        }

        s.clear();
        //reverse because the loop for smaller in right is run from right most end to left most end
        //data processed in reverse order
        Collections.reverse(smallerRight);

        //next smaller element to left
        for (int i = 0; i < n; i++) {

            while (!s.isEmpty() && s.peek().getKey() > histogram[i]) {
                s.pop();
            }

            if (s.isEmpty()) {
                smallerLeft.add(new Pair<>(-1, -1));
            } else {
                smallerLeft.add(new Pair<>(s.peek().getKey(), s.peek().getValue()));
            }

            s.push(new Pair<>(histogram[i], i));

        }

        //calculate with of histogrm
        int maxArea = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {

            int width = smallerRight.get(i).getValue() - smallerLeft.get(i).getValue() - 1;
            maxArea = Math.max(maxArea, width * histogram[i]);

        }

        System.out.println("Max area of the given histogram " + maxArea);

    }

    public static void pushZerosToEnd(int[] arr) {
        // code here
        //..........................O(N).....................
        int start = 0;
        int end = start + 1;

        while (end < arr.length) {

            if (arr[start] == 0 && arr[end] != 0) {
                int temp = arr[start];
                arr[start] = arr[end];
                arr[end] = temp;
                start++;
            }
            if (arr[start] != 0) {
                start++;
            }

            end++;
        }

        //output
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public static void calculateAngleBetweenHourAndMin(int hour, int min) {

        /*
        
         The idea is to take 12:00 (h = 12, m = 0) as a reference. Following are detailed steps.

         1. Calculate the angle made by hour hand with respect to 12:00 in h hours and m minutes. 
         2. Calculate the angle made by minute hand with respect to 12:00 in h hours and m minutes. 
         3. The difference between the two angles is the angle between the two hands.

         How to calculate the two angles with respect to 12:00? 
         The minute hand moves 360 degrees in 60 minute(or 6 degrees in one minute) and hour hand moves 360 
         degrees in 12 hours(or 0.5 degrees in 1 minute). In h hours and m minutes, 
         the minute hand would move (h*60 + m)*6 and hour hand would move (h*60 + m)*0.5. 
        
         */
        // validate the input
        if (hour < 0 || min < 0 || hour > 12 || min > 60) {
            System.out.println("Wrong input");
        }

        if (hour == 12) {
            hour = 0;
        }
        if (min == 60) {
            min = 0;
            hour += 1;
            if (hour > 12) {
                hour = hour - 12;
            }
        }

        // Calculate the angles moved by hour and minute hands
        // with reference to 12:00
        int hour_angle = (int) (0.5 * (hour * 60 + min));
        int minute_angle = (int) (6 * min);

        // Find the difference between two angles
        int angle = Math.abs(hour_angle - minute_angle);

        // smaller angle of two possible angles
        angle = Math.min(360 - angle, angle);

        System.out.println(angle + " degrees");

    }

    public static int lastOccurence(int arr[], int low, int high, int x, int n){
        if (high >= low) {
            int mid = low + (high - low) / 2;
            if ((mid == n - 1 || x < arr[mid + 1]) && arr[mid] == x)
                return mid;
            else if (x < arr[mid])
                return lastOccurence(arr, low, (mid - 1), x, n);
            else
                return lastOccurence(arr, (mid + 1), high, x, n);
        }
        return -1;
    }
    
    public static int firstOccurence(int arr[], int low, int high, int x, int n){
        if (high >= low) {
            int mid = low + (high - low) / 2;
            if ((mid == 0 || x > arr[mid - 1]) && arr[mid] == x)
                return mid;
            else if (x > arr[mid])
                return firstOccurence(arr, (mid + 1), high, x, n);
            else
                return firstOccurence(arr, low, (mid - 1), x, n);
        }
        return -1;
    }
    
    public static void firstAndLastOccurenceInSortedArray(int[] nums, int target) {
        
        int N = nums.length;
        int f = 0;
        int l = nums.length;
        int[] result = {-1, -1};
        
        result[0] = firstOccurence(nums, f, l-1, target, N);
        result[1] = lastOccurence(nums, f, l-1, target, N);
        
        System.out.println("First and last: "+result[0]+" "+result[1]);
        
    }
    
    public static int longestCommonSubsequence_Recursive(String a, String b, int aLen, int bLen) {

        //least end point when we reach to 0 index of string
        if (aLen == 0 || bLen == 0) {
            return 0;
        } else if (a.charAt(aLen - 1) == b.charAt(bLen - 1)) {
            return 1 + longestCommonSubsequence_Recursive(a, b, aLen - 1, bLen - 1);
        } else {
            return Math.max(longestCommonSubsequence_Recursive(a, b, aLen - 1, bLen),
                    longestCommonSubsequence_Recursive(a, b, aLen, bLen - 1));
        }

    }

    public static int longestCommonSubsequence_DP_Memoization(String a, String b, int aLen, int bLen, int memo[][]) {

        if (aLen == 0 || bLen == 0) {
            memo[aLen][bLen] = 0;
            return memo[aLen][bLen];
        }

        //return if anything is previously memorized 
        //returing from here where something is already memo, saves below down function call 
        //of recursion
        //-1 is chosen becuse in our base cond 0 is our least output
        //so choose somthing less than your least output from base cond
        if (memo[aLen][bLen] != -1) {
            return memo[aLen][bLen];
        }

        if (a.charAt(aLen - 1) == b.charAt(bLen - 1)) {

            memo[aLen][bLen] = 1 + longestCommonSubsequence_DP_Memoization(a, b, aLen - 1, bLen - 1, memo);

            return memo[aLen][bLen];
        } else {
            memo[aLen][bLen] = Math.max(longestCommonSubsequence_DP_Memoization(a, b, aLen - 1, bLen, memo),
                    longestCommonSubsequence_DP_Memoization(a, b, aLen, bLen - 1, memo));
            return memo[aLen][bLen];
        }

    }

    public static int longestCommonSubsequence_DP_TopDown(String a, String b, int aLen, int bLen) {
        //DP top down approach restricts recursive calls and 
        //praise for iterative approach
        //but both use memo matrix to store data

        /*
         Basic recursive algo 
         just use algo to derive calculation for top down aapproach
         consider aLen = x, bLen = y in memo[x][y]
         if (aLen == 0 || bLen == 0) {
            
         here if aLen is 0 whole (x,y), (x+1,y), ... (aLen,y) == 0
         here if bLen is 0 whole (x,y), (x,y+1), ... (x,bLen) == 0
         above means 1st row and 1st col will be init with base cond return i.e, 0
        
         return 0;
         } else if (a.charAt(aLen - 1) == b.charAt(bLen - 1)) {
        
         //when a char matches add 1 and move 1 char ahead in both strings
         //so in memo[x][y] = 1+ memo[x-1][y-1];
         //for x,y -> x-1,y-1 observe it is previous diagonal element in matrix
        
         x-1, y-1    -       -
         -           x,y     -
         -           -       x+1, y+1
        
         return 1 + longestCommonSubsequence_Recursive(a, b, aLen - 1, bLen - 1);
         } else {
        
         //when char doesn't match we go ahead in 1st string and 2nd string both at a time and 
         //take its max
         //memo[x][y] -> max(memo[x-1][y], memo[x][y-1])
         //observer for x,y -> x-1,y is previous row-same col element and
         //x,y -> x,y-1 is same row but previous col element
        
         -           x-1,y       -
         x, y-1      x,y         -
         -           -           -
         return Math.max(longestCommonSubsequence_Recursive(a, b, aLen - 1, bLen),
         longestCommonSubsequence_Recursive(a, b, aLen, bLen - 1));
         }
        
         these are the calculation derived from actual recursive algo
        
         */
        int[][] memo = new int[aLen + 1][bLen + 1];

        //base cond...
        //below for loop will however fill all the cells with 0
        //but that doesnt matter 
        //our 1st row and 1st col is filled with base cond that we needed for our Top down approach
        //this base cond init is necessary to start top down
        for (int[] r : memo) {
            Arrays.fill(r, 0);
        }

        //x = 1, y = 1 as x = 0 , y = 0 is assumed to be our base cond data
        for (int x = 1; x < aLen + 1; x++) {
            for (int y = 1; y < bLen + 1; y++) {

                if (a.charAt(x - 1) == b.charAt(y - 1)) {
                    memo[x][y] = 1 + memo[x - 1][y - 1];
                } else {
                    memo[x][y] = Math.max(memo[x - 1][y], memo[x][y - 1]);
                }

            }
        }

        //our ans will be at aLen,bLen
        return memo[aLen][bLen];

    }

    public static int knapSack01_Recusrion(int[] profit, int[] weight, int n, int W) {

        //for base cond think like
        //if profit array is not given then W will automatically is 0
        //if W the weight capacity of knapsack is 0 that means you cant pick any thing 
        //from profit as your knapsack capacity is 0
        if (n == 0 || W == 0) {
            return 0;
        }

        //now choose those weights that are less than W
        if (weight[n - 1] <= W) {
            //if  weight[x] is less than W then we have 2 choices 
            //either we pick it up or we will leave it and may be choose next
            //a. if we pick it up then profit must be considered and W should
            //be reduced by some weight[x] and we move further
            //b. if dont choose it then its profit value is not applicable and its weight will
            //also not be taken hence W remains as it is
            //now problem says we need max profit at weight <= W
            return Math.max(profit[n - 1] + knapSack01_Recusrion(profit, weight, n - 1, W - weight[n - 1]),
                    knapSack01_Recusrion(profit, weight, n - 1, W));
        } else {
            //suppose in weights[x] is aalready greater than the W then there is no choice 
            //our knapsack can hold it. so we will leave it and move to next n
            return knapSack01_Recusrion(profit, weight, n - 1, W);
        }

    }

    public static int knapSack01_DP_Memoization(int[] profit, int[] weight, int n, int W, int[][] memo) {

        //in dp memoization observe those variable that decreases by the recursive calss
        //here acc to recursion above n, W are changing by recursive call
        //so memo[][] will conatins this changing variable
        //memo[x][y] x = n+1, y=W+1
        //memo[0][0] where whole row x = 0 and whole col y = 0  will be our base cond
        //now our base cond will be the starting values in memo[][]
        //if no profits are given W will be 0
        //if W is 0 we cant hold anything in the knapsack so profit will be 0
        //so memo[0][0] 0th row signifies no profits provided so all x=0 will be init 0
        //0th col signifies W if W = 0 so profit will be 0 so all y=0 will be init 0
        if (n == 0 || W == 0) {
            memo[n][W] = 0;
            return memo[n][W];
        }

        if (memo[n][W] != 0) {
            return memo[n][W];
        }

        if (weight[n - 1] <= W) {
            memo[n][W] = Math.max(profit[n - 1] + knapSack01_DP_Memoization(profit, weight, n - 1, W - weight[n - 1], memo),
                    knapSack01_DP_Memoization(profit, weight, n - 1, W, memo));
            return memo[n][W];
        } else {
            memo[n][W] = knapSack01_DP_Memoization(profit, weight, n - 1, W, memo);
            return memo[n][W];
        }

    }

    public static int knapSack01_DP_TopDown(int[] profit, int[] weight, int n, int W) {

        /*
         //basic recursive algo to derive calculation for 
         //top down
         //memo[x][y] where x = n+1, y=W+1
         //memo[0][0] where x=0 0th row will be base cond
         //y=0 0th col will be base cond
         //x=0 signifies not profit array is given
         //y=0 signifies if W is 0 then you cant hold anything in your knapsack
         if(n == 0 || W == 0){
         return 0;
         }
        
         if(weight[n-1] <= W){
         //x is analogous to n
         //if(weight[x-1] <= y)
         //previous diagonal element
         //y is analogous to W so we will do [y - weight[y-1]]
         //a. profit[x-1] + memo[x-1][y-1] = profit[x-1] + memo[x-1][y - weight[y-1]]
         //b. memo[x-1][y];
         //previous row same col
         //c. memo[n][W]= max(a, b);
         return Math.max(profit[n-1] + knapSack01_Recusrion(profit, weight, n-1, W - weight[n-1]), 
         knapSack01_Recusrion(profit, weight, n-1, W));
         }else {
         //memo[n][W] = memo[x-1][y];
         //previous row same col
         return knapSack01_Recusrion(profit, weight, n-1, W);
         }
        
         */
        int[][] memo = new int[n + 1][W + 1];
        for (int[] r : memo) {
            //base cond
            Arrays.fill(r, 0);
        }

        //x=0,y=0 row and col are base cond to start with
        //so start from x=1,y=1
        for (int x = 1; x < n + 1; x++) {
            for (int y = 1; y < W + 1; y++) {

                if (weight[x - 1] <= y) {
                    memo[x][y] = Math.max(profit[x - 1] + memo[x - 1][y - weight[x - 1]], memo[x - 1][y]);
                } else {
                    memo[x][y] = memo[x - 1][y];
                }

            }
        }

        return memo[n][W];

    }

    public static boolean subsetSum_Recursion(int[] a, int n, int sum) {

        //think like if a size is 0 i.e, no elements given
        //the only subset is possbile in this case is null/empty subset
        //so sum of empty subset is 0
        //another if a size is given but sum to be calculated is 0
        //then in that case also it is possible to return sum as 0 becuase 
        //there always lie an empty subset whoose sum sum is 0
        if (n == 0 && sum == 0) {
            return true;
        }

        //in cases we have given no element but sum is given which is non zero
        //thenin that case you cant form a sum i.e, >0
        //ex a[] = 20, 30, 40; sum = 10
        //ex a[] = []; sum = 10
        if (n == 0 && sum != 0) {
            return false;
        }

        //if the taken element is already greater than the sum just move
        if (a[n - 1] > sum) {
            return subsetSum_Recursion(a, n - 1, sum);
        }
        //else we have 2 choices 
        //a. we can choose last element of our a in the subset that may or may not
        //create that sumTotal = sum
        //b. We can leave that last element and move to next one in this one 
        //we know that we havent chosn the element so it will not form sumTotal = sum
        //in both a and b we have to take that value which is giving true 
        return subsetSum_Recursion(a, n - 1, sum - a[n - 1]) || subsetSum_Recursion(a, n - 1, sum);

    }

    public static boolean subsetSum_DP_Memoization(int[] a, int n, int sum, boolean[][] memo) {

        //varibles that are changing with recusrion calls
        //n and sum
        //so memo[x][y]  = memo[n+1][sum+1]
        //where x=0, y=0 row and col are our base condition
        //visualize at x=0 row y is ranging from 0 -> sum+1
        //x=0 denotes no element is given in a[] 
        //so with no element given only empty subset is possble whoose sum is 0
        //a. x,y == 0,0 is true
        //b.but when x=0 and y >=1 -> sum+1 no subset is possble that can form sumTotal = sum
        //x,y = where y = [1->sum+1] is false
        //c. x = [0->n+1] now a[] size is provided but y=0 
        //that means we any no of element is given we have to make sum = y =0
        //any no element in that always empty subset is possble so sum = y = 0 always possible
        //then y =0 col is always true
        //      0     1       2       3...Sum+1
        //0     T     F       F       F
        //1     T
        //2     T
        //n+1   T
        if (n == 0 && sum == 0) {
            return memo[n][sum] = true;
        }

        //in cases we have given no element but sum is given which is non zero
        //thenin that case you cant form a sum i.e, >=0
        //ex a[] = 20, 30, 40; sum = 10
        //ex a[] = []; sum = 10
        if (n == 0 && sum != 0) {
            return memo[n][sum] = false;
        }

        //if some true value is there return
        if (memo[n][sum]) {
            return memo[n][sum];
        }

        //if the taken element is already greater than the sum just move
        if (a[n - 1] > sum) {
            return memo[n][sum] = subsetSum_Recursion(a, n - 1, sum);
        }
        //else we have 2 choices 
        //a. we can choose last element of our a in the subset that may or may not
        //create that sumTotal = sum
        //b. We can leave that last element and move to next one in this one 
        //we know that we havent chosn the element so it will not form sumTotal = sum
        //in both a and b we have to take that value which is giving true 
        memo[n][sum] = subsetSum_Recursion(a, n - 1, sum - a[n - 1]) || subsetSum_Recursion(a, n - 1, sum);
        return memo[n][sum];
    }

    public static boolean subsetSum_DP_TopDown(int[] a, int n, int sum) {

        /*
        
         //varibles that are changing with recusrion calls
         //n and sum
         //so memo[x][y]  = memo[n+1][sum+1]
         //where x=0, y=0 row and col are our base condition
         //visualize at x=0 row y is ranging from 0 -> sum+1
         //x=0 denotes no element is given in a[] 
         //so with no element given only empty subset is possble whoose sum is 0
         //a. x,y == 0,0 is true
         //b.but when x=0 and y >=1 -> sum+1 no subset is possble that can form sumTotal = sum
         //x,y = where y = [1->sum+1] is false
         //c. x = [0->n+1] now a[] size is provided but y=0 
         //that means we any no of element is given we have to make sum = y =0
         //any no element in that always empty subset is possble so sum = y = 0 always possible
         //then y =0 col is always true
         //      0     1       2       3...Sum+1
         //0     T     F       F       F
         //1     T
         //2     T
         //n+1   T
        
         if(n == 0 || sum == 0){
         return true;
         }
        
         if(n == 0 && sum !=0){
         return false;
         }
        
         //n = x, sum=y
         if(a[x-1]>y)
         if(a[n-1] > sum){
         //memo[x][y] = subsetSum_Recursion(a, n-1, sum) = memo[x-1][y]
         //previous row same col
         return subsetSum_Recursion(a, n-1, sum);
         }
        
         memo[x][y] = memo[x-1][y-a[x-1]] || memo[x-1][y]
         return subsetSum_Recursion(a, n-1, sum - a[n-1]) || subsetSum_Recursion(a, n-1, sum);
        
         */
        //base cond
        boolean[][] memo = new boolean[n + 1][sum + 1];
        for (int x = 0; x < n + 1; x++) {
            for (int y = 0; y < sum + 1; y++) {
                if (x == 0) {
                    memo[x][y] = false;
                }

                if (y == 0) {
                    memo[x][y] = true;
                }
            }
        }

        for (int x = 1; x < n + 1; x++) {
            for (int y = 1; y < sum + 1; y++) {
                if (a[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = memo[x - 1][y - a[x - 1]] || memo[x - 1][y];
                }
            }
        }

        return memo[n][sum];

    }

    public static boolean equalSumSubsetPartition(int[] a) {

        //the problem is to check if the given array can be divided
        //into 2 subset in such a ways tht sum(subset1) = sum(subset2)
        //a[] = 1, 5, 5, 11
        //s1 = 1, 5, 5 | s2 = 11
        //both sum of s1 and s2 are equal = 11
        //observe a sum can be equally divided only if it is even
        //ex sum of a[] = 1+5+5+11 = 22 = 2*11 that is even 
        //if it was odd like 23 = it cant be possible to make equal sum subsets
        //so for sum(a[0->n-1]) == odd return false
        //for even we can use above algo of subsetSum()
        //observe sumTotal = 22 = 2*11
        //sumTotal/2 we can find if it is possble for just one subset then it means the other
        //subset is already be half of it i.e, 11
        int sumTotal = 0;
        for (int i = 0; i < a.length; i++) {
            sumTotal += a[i];
        }

        if (sumTotal % 2 != 0) {
            //odd
            return false;
        } else {
            return subsetSum_Recursion(a, a.length, sumTotal / 2);
        }

    }

    public static int countSubsetSum_Recursion(int[] a, int n, int sum) {

        if (n == 0 && sum == 0) {
            return 1;
        }

        if (n == 0 && sum != 0) {
            return 0;
        }

        if (a[n - 1] > sum) {
            return countSubsetSum_Recursion(a, n - 1, sum);
        }

        return countSubsetSum_Recursion(a, n - 1, sum - a[n - 1]) + countSubsetSum_Recursion(a, n - 1, sum);

    }

    public static int countSubsetSum_DP_Memoization(int[] a, int n, int sum, int[][] memo) {

        if (n == 0 && sum == 0) {
            return memo[n][sum] = 1;
        }

        if (n == 0 && sum != 0) {
            return memo[n][sum] = 0;
        }

        if (memo[n][sum] != 0) {
            return memo[n][sum];
        }

        if (a[n - 1] > sum) {
            return memo[n][sum] = countSubsetSum_DP_Memoization(a, n - 1, sum, memo);
        }

        return memo[n][sum] = countSubsetSum_DP_Memoization(a, n - 1, sum - a[n - 1], memo) + countSubsetSum_DP_Memoization(a, n - 1, sum, memo);

    }

    public static int countSubsetSum_DP_TopDown(int[] a, int n, int sum) {

        //base cond
        int[][] memo = new int[n + 1][sum + 1];
        for (int x = 0; x < n + 1; x++) {
            for (int y = 0; y < sum + 1; y++) {
                if (x == 0) {
                    memo[x][y] = 0;
                }

                if (y == 0) {
                    memo[x][y] = 1;
                }
            }
        }

        //start aahead of base cond
        for (int x = 1; x < n + 1; x++) {
            for (int y = 1; y < sum + 1; y++) {

                if (a[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = memo[x - 1][y - a[x - 1]] + memo[x - 1][y];
                }

            }
        }

        return memo[n][sum];

    }

    public static int minDiffInEqualSubset(int[] a) {

        //ex:  3, 1, 4, 2, 2, 1
        //with the given array a the max sum that can be 
        //possible is the sum of array itself
        int sumArray = 0;
        for (int i = 0; i < a.length; i++) {
            sumArray += a[i];
        }

        //0 ---- sumArray
        int n = a.length;
        //base cond
        boolean[][] memo = new boolean[n + 1][sumArray + 1];
        for (int x = 0; x < n + 1; x++) {
            for (int y = 0; y < sumArray + 1; y++) {
                if (x == 0) {
                    memo[x][y] = false;
                }

                if (y == 0) {
                    memo[x][y] = true;
                }
            }
        }

        //start from ahead of base cond
        //to analyse y = sum = [0 -> sumArray+1] is possible or not
        for (int x = 1; x < n + 1; x++) {
            for (int y = 1; y < sumArray + 1; y++) {
                if (a[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = memo[x - 1][y - a[x - 1]] || memo[x - 1][y];
                }
            }
        }

//        //printing
//        for (int x = 0; x < n + 1; x++) {
//            for (int y = 0; y < sumArray + 1; y++) {
//                System.out.print(memo[x][y]+"\t");
//            }
//            System.out.println();
//        }
        //now consider the last row of the memo[][] i.e, x = nth row
        //that row signifies all the elements in array is taken to form y = sum = [0 -> sumArray+1]
        int min = Integer.MAX_VALUE;
        for (int i = (sumArray + 1) / 2; i >= 0; i--) {
            if (memo[n][i]) {
                min = Math.abs(sumArray - 2 * i);
                break;
            }
        }

        return min;

    }

    public static int subsetSumToAGivenDifference(int[] a, int n, int diff) {

        //for a given array its sumof subset
        //s1 and s2 = diff
        //s1 - s2 = diff
        //also s1+s2 = sumArr
        //both eqn = 2s1= sumarr+diff
        //s1 = (sumarr+diff)/2 i.e, sum of s1
        //if we can aply countSubsetSum algo with arr, n , sum = ((sumarr+diff)/2)
        //we can find the no of s1 can be formed whoose sum would be ((sumarr+diff)/2)
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += a[i];
        }

        int sumS1 = (sum + diff) / 2;

        return countSubsetSum_DP_TopDown(a, n, sumS1);

    }

    public static int waysToFormAGivenTargetSumByAssigningSigns(int[] a, int n, int target) {

        //problem is like, given array is like 
        //a[] = 1, 1, 2, 3
        //target = 1
        //assign sign before element such that by performing oper it should result to target
        //we have to find the no of ways we can assign sign.
        //ex: +1, -1, -2, +3 = 1
        //ex: -1, +1, -2, +3 = 1
        //ex: +1, +1, +2, -3 = 1
        //total 3 ways 
        //observe keep assign similar sign +(1,3) - (1,2) = 1
        //+(1,3) - (1,2) = 1
        //+(3) - (1,1,2) = 1
        //it is actaully subsetSumToAGivenDifference
        return subsetSumToAGivenDifference(a, n, target);

    }

    public static int rodCutting(int[] length, int[] price, int L) {

        //since length,price and L all denotes the rod 
        //length array denotes length segments made out of rod L like 1 len + 7 len = L
        //1, 2, 3...L
        //price array denotes price of per segment of the rod L
        //so sometimes length array is not given then we can make our own array
        //loop 1 -> L
        //since n == L so actaull memo[][] is n*n = n^2 so overall time O(n^2)
        int n = length.length; //which is basically L
        int[][] memo = new int[n + 1][L + 1]; //[n+1][n+1] will also be correct

        //if rod of 0 length is given then no price can be assumed for that i.e, x = 0 row
        //if some rod length is given but we are cuttin just 0 length out of it then price of this 
        //is also not assumes y = 0 col 
        //base cond
        for (int[] r : memo) {
            Arrays.fill(r, 0);
        }

        for (int x = 1; x < n + 1; x++) {
            for (int y = 1; y < L + 1; y++) {

                if (length[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = Math.max(price[x - 1] + memo[x][y - length[x - 1]], memo[x - 1][y]);
                }

            }
        }

        return memo[n][L];

    }

    public static int waysToMakeCoinChange(int[] coins, int n, int K) {

        //K is something whoose chnange is to find using coin[]
        //n is no of unique coins in the coins[]
        //there is unlimited supply  of each amount of coin
        //unbounded knapsack problem
        int[][] memo = new int[n + 1][K + 1];
        //if x=0 row no coin is given then only change of K = 0 is possble by
        //not taking any coins
        //other chnage K from 1 -> K+1 is not possible
        //similarly if some amount of coin is prsent in coins[] but K = y =0 col
        //then change of 0 is possible by not choosing any coins
        //x=0 row ...0
        //y=0 col ...1

        //base cond
        for (int x = 0; x < n + 1; x++) {
            for (int y = 0; y < K + 1; y++) {
                if (x == 0) {
                    memo[x][y] = 0;
                }

                if (y == 0) {
                    memo[x][y] = 1;
                }
            }
        }

        for (int x = 1; x < n + 1; x++) {
            for (int y = 1; y < K + 1; y++) {

                if (coins[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = memo[x][y - coins[x - 1]] + memo[x - 1][y];
                }

            }
        }

        return memo[n][K];

    }

    public static int minNoOfCoinsUsedForChange(int[] coins, int n, int K) {

        //this problem has a vriation in setting base cond that we 
        //usually use in coin change and unbounded knapsack 
        //lets say coins[]= 1,2,3 and K=3
        //when memo[x][y] x=0 row when no coins is given 
        //then think mathemaically to create a change for y=0 -> <K+1
        //we may need some no of coins to make that change
        //like coin[] = empty and K= 3
        //we may require 1,2,3....upto infinite no of coins to make that K change
        //so for x=0 row where we have to make change y = K = 0 -> <K+1
        //we need min of Int_Max supply of coins
        //now y=0 col but x is upto n+1 that means some no of coins is provided
        //but we have to make change for K= y= 0
        //then out of any no of coins prodvided to you need min 0 coins to make change
        //for K =0
        //so y=0 col is 0 where x>=1 -> <n+1
        //here is the variation,
        //we need to set base cond for x=1 row where y >=1 -> K+1
        //lets say coins[]= 3,2,1 
        //if coin[x] = coin[1-1] = 3 one element of coin is given 
        //now y=K=1 then if y/coin[0] => 1%3 == 0 then min coin req is one
        //again y=K=3 coin[0] = 3%3 == 0 then we require only one coin of 3 to make
        //cahnge for K=3
        //y=K=4 coin[0] =3 => 4%3 !=0 then it means min Int_Max no of coin more may be req to make change for K=4
        //for safer side use Int_Max - 1  so that if min no of coin exceeds the mathematical
        //limit it should be handled
        int[][] memo = new int[n + 1][K + 1];

        //base cond
        for (int x = 0; x < n + 1; x++) {
            for (int y = 0; y < K + 1; y++) {

                if (x == 0) {
                    memo[x][y] = Integer.MAX_VALUE - 1;
                }

                if (y == 0) {
                    memo[x][y] = 0;
                }

                //variation in base cond
                if (x == 1 && y >= 1) {
                    if (y % coins[x - 1] == 0) {
                        memo[x][y] = 1;
                    } else {
                        memo[x][y] = Integer.MAX_VALUE - 1;
                    }
                }

            }
        }

        for (int x = 1; x < n + 1; x++) {
            for (int y = 1; y < K + 1; y++) {

                if (coins[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    //unbounded use of coins from coins[]
                    //we have to take min of no of coin used to make K=y chnange
                    //a. we will use the coin in the coin[] if we are using then min 1 coin we are using
                    //for that reason 1 is added
                    //b. we will not choose that coin in the coin[] then we move to next coin in coin[]
                    //but out K hasnot got change form the coin we didn't choose.
                    memo[x][y] = Math.min(1 + memo[x][y - coins[x - 1]], memo[x - 1][y]);
                }

            }
        }

        return memo[n][K];
    }

    public static int longestCommonSubstring(String a, String b, int aLen, int bLen) {

        //longest common substring is variation in longest common subseq
        //a substing is contnious occurence of char in both the string
        //lets say a = "abcfgh" , b = "abdgeh"
        //in this ab is longest common subtring 
        //as the char at 3 index c and d is causing discontinuity
        //as long as char occur are matching in both we will add 1 as the char mtches 
        //and when discontinuity occurence we will consider it as 0 this signifies
        //there may be a substring which is greater after the discontinuity
        //lets say  a = "abcfghxyz" , b = "abdgehxyz"
        //in above ab is sub string with max len 2 the at index 3 discontinuity occur 'c'and'd'
        //so for discontinuity length is 0
        //after that hxyz is sub string that matches and has len greater than ab substring
        //so in this longest common substring is 4(hxyz)
        //base cond 
        //if aLen and bLen is 0 that means empty string is given
        //then there is no longest common substring is possbile
        int[][] memo = new int[aLen + 1][bLen + 1];

        //base cond
        //if both string len is 0 then longest common substring is 0
        for (int[] r : memo) {
            Arrays.fill(r, 0);
        }
        int maxLen = 0;
        for (int x = 1; x < aLen + 1; x++) {
            for (int y = 1; y < bLen + 1; y++) {

                if (a.charAt(x - 1) == b.charAt(y - 1)) {
                    memo[x][y] = 1 + memo[x - 1][y - 1];
                    maxLen = Math.max(maxLen, memo[x][y]);
                } else {
                    //case of discontunuity
                    memo[x][y] = 0;
                }

            }
        }

        return maxLen;

    }

    public static void printLongestCommonSubsequence_DP_TopDown(String a, String b, int aLen, int bLen) {

        int[][] memo = new int[aLen + 1][bLen + 1];

        //base cond
        //if both string are empty then longest common subseq is not possible i.e, 0
        for (int[] r : memo) {
            Arrays.fill(r, 0);
        }

        //lcs topdown as earlier
        StringBuilder sb = new StringBuilder();
        for (int x = 1; x < aLen + 1; x++) {
            for (int y = 1; y < bLen + 1; y++) {

                if (a.charAt(x - 1) == b.charAt(y - 1)) {
                    memo[x][y] = 1 + memo[x - 1][y - 1];
                    sb.append(a.charAt(x - 1));
                } else {
                    memo[x][y] = Math.max(memo[x - 1][y], memo[x][y - 1]);
                }

            }
        }
        System.out.println("printing LCsubseq by performing LCsubseq top down " + sb.toString());

        //traversing memo[][] from x,y = aLen+1, bLen upto
        //x>0, y>0 because x =0, y=0 row and col are our base cond
        int i = aLen;
        int j = bLen;
        sb = new StringBuilder();
        while (i > 0 && j > 0) {

            //if last char of strings a, b matched then 
            //it would be the result of memo[x-1][y-1] + 1
            //i.e, previous row previous col/ diagonally prev element + 1
            //we will take this char
            if (a.charAt(i - 1) == b.charAt(j - 1)) {
                sb.append(a.charAt(i - 1));
                //move to new char 
                i--;
                j--;
            } else {
                //if those char doesn't match we cant add to our sb
                //it would be the result of max of memo[x-1][y], memo[x][y-1]
                //i.e, Max(element at prev row same col, element at prev col same row)
                //whichever element is max we move to that part
                if (memo[i - 1][j] > memo[i][j - 1]) {
                    //element at prev row was max 
                    //move to that row by adjusting i = row
                    i--;
                } else {
                    //element at prev col was max 
                    //move to that col by adjusting j = col
                    j--;
                }

            }

        }

        //since we are traversing from last of char/memo[][] our actual longest common subseq 
        //in sb will be stored as reverse
        System.out.println("printing LCsubseq by processing memo[][] " + sb.reverse().toString());

    }

    public static int shortestCommonSuperSequence_Recusrion(String a, String b, int aLen, int bLen) {

        //a super seq string is somthing from which a is subseq and b is also subseq
        //ex a = "adh" b = "achi" its superseq is "adchi" as a and b both are subseq of this
        //think for base cond
        //if a is empty and b is given the shortest common superseq would be b itself
        //ex a = "" b = "achi" superseq = "achi" 
        //and same is the reason if a is given and b is empty
        if (aLen == 0) {
            return bLen;
        }

        if (bLen == 0) {
            return aLen;
        }

        //if char matches as last index then atleast 1 char that is in common to
        //both string will be in the superseq
        if (a.charAt(aLen - 1) == b.charAt(bLen - 1)) {
            return 1 + shortestCommonSuperSequence_Recusrion(a, b, aLen - 1, bLen - 1);
        } else {
            //if the char doesn't match 
            //a. move one char in a and check with b as whole
            //b. move one char in b and check a as whole
            //since we are concering over shortest we will take min of above cond
            //now that min value must be added to our superseq for that we will add 1
            return 1 + Math.min(shortestCommonSuperSequence_Recusrion(a, b, aLen - 1, bLen),
                    shortestCommonSuperSequence_Recusrion(a, b, aLen, bLen - 1));
        }

    }

    public static int shortestCommonSuperSequence_DP_Memoization(String a, String b, int aLen, int bLen, int[][] memo) {

        if (memo[aLen][bLen] != -1) {
            return memo[aLen][bLen];
        }

        if (aLen == 0) {
            return memo[aLen][bLen] = bLen;
        }

        if (bLen == 0) {
            return memo[aLen][bLen] = aLen;
        }

        if (a.charAt(aLen - 1) == b.charAt(bLen - 1)) {
            return memo[aLen][bLen] = 1 + shortestCommonSuperSequence_DP_Memoization(a, b, aLen - 1, bLen - 1, memo);
        } else {
            return memo[aLen][bLen] = 1 + Math.min(shortestCommonSuperSequence_DP_Memoization(a, b, aLen - 1, bLen, memo),
                    shortestCommonSuperSequence_DP_Memoization(a, b, aLen, bLen - 1, memo));
        }

    }

    public static int shortestCommonSuperSequence_DP_TopDown(String a, String b, int aLen, int bLen) {

        //base cond
        int[][] memo = new int[aLen + 1][bLen + 1];
        for (int x = 0; x < aLen + 1; x++) {
            for (int y = 0; y < bLen + 1; y++) {
                if (x != 0 || y != 0) {
                    memo[x][y] = -1;
                }

                if (x == 0) {
                    memo[x][y] = y;
                }

                if (y == 0) {
                    memo[x][y] = x;
                }

            }
        }

        for (int x = 1; x < aLen + 1; x++) {
            for (int y = 1; y < bLen + 1; y++) {

                if (a.charAt(x - 1) == b.charAt(y - 1)) {
                    memo[x][y] = 1 + memo[x - 1][y - 1];
                } else {
                    memo[x][y] = 1 + Math.min(memo[x - 1][y], memo[x][y - 1]);
                }

            }
        }

        return memo[aLen][bLen];

    }

    public static int shortestCommonSuperSequnce_BasedDirectOnLCS(String a, String b, int aLen, int bLen) {

        //approach based on LCsubsequence
        //a = "AGGTAB" b = "GXTXAYB"
        //lcs of a, b will give GTAB len = 4
        //in worst case super seq could be a concat b = AGGTABGXTXAYB len = 13
        //lcs(a, b) = GTAB len = 4 
        //observer in a concat b  = AGGTABGXTXAYB -> GTAB seq is occuring twice 
        //so remove one occurenec of lcs
        //scs(a, b) = len(a) + len(b) - lcs(a,b)
        //shortest common supersequence =  AGXGTXAYB len = 9
        return a.length() + b.length() - longestCommonSubsequence_DP_TopDown(a, b, aLen, bLen);

    }

    public static void minInsertOrDeleteReqToConvertAToBString(String a, String b) {

        //probelm is to convert string a to b with min no of insert and delete operation
        //lets say a= "heap" b = "pea"
        // if we can delete 2 char from a = "ea" del(h, p)
        //and after that we can insert 1 char to a now a= "pea" ins(p)
        //now a == b
        //now observe  a = "heap" and b = "pea" where ea is LCsubseq in both string
        //if we can convert a to LcSubseq to b we can easily find the no of char to
        //del and to ins
        //LCsubseq = "ea" len = 2
        //a = LCsubseq = 2 del = a.length  - LCsubeq.length
        //LCsubseq = b = 1 ins = b.length - LCsubseq.length
        int aLen = a.length();
        int bLen = b.length();

        int[][] memo = new int[aLen + 1][bLen + 1];

        //base cond
        for (int[] r : memo) {
            Arrays.fill(r, 0);
        }

        for (int x = 1; x < aLen + 1; x++) {
            for (int y = 1; y < bLen + 1; y++) {

                if (a.charAt(x - 1) == b.charAt(y - 1)) {
                    memo[x][y] = 1 + memo[x - 1][y - 1];
                } else {
                    memo[x][y] = Math.max(memo[x - 1][y], memo[x][y - 1]);
                }

            }
        }

        int lcs = memo[aLen][bLen];

        int delReq = aLen - lcs;
        int insReq = bLen - lcs;

        System.out.println("min insertion req : " + insReq + " min deletion req : " + delReq);

    }

    public static int longestPallindromicSubsequence_LCSubseqAsParent(String a) {

        //problem says from a string a = "agbcba" get a subseq 
        //that is also pallindromic and also longest
        //here bcb,abcba are pallindroimic subseq 
        //but abcba is longest
        //how we can apply LCSubseq as parent to this problem
        //a = "agbcba" b = a.reverse = "abcbga"
        //if we can apply LCsubseq on a,b it will be same as lonegest Pallindromic subseq
        return longestCommonSubsequence_DP_TopDown(a, new StringBuilder(a).reverse().toString(),
                a.length(),
                a.length());

    }

    public static int minNoOfDeletionInStringToMakeItPallindrome_LPSBasedApproach(String a) {

        ////https://www.geeksforgeeks.org/amazon-interview-experience-sde-1-feb-2020-exp-1-5-yr/?ref=leftbar-rightbar
        //lets say a = "agbcba"
        //min no of char to be deleted that the left string is pallindromic
        //if we delete a,g,a left string = bcb 
        //in this min no deletion is 3 and length of pallindromic string is 3
        //another delete g left string is abcba
        //min no of deletion is 1 and length of pllindromic string is 5
        //observe if we can find the longestPallindromicSubseq of string
        //then the sub the LPS from string.length will give the min no of deleteion
        //alternate approach O(N)
        //removeCharToLeftPallindrome();
        //.........O(N^2) N= a.length()
        return a.length() - longestPallindromicSubsequence_LCSubseqAsParent(a);

    }

    public static void printShortestCommonSuperSequence(String a, String b, int aLen, int bLen) {

        //while talking about LCSubseq
        //lets say a= "acdefgh" and b="axyezfkg"
        //lcs  = aefg common in both
        //for scs a.len + b.len - lcs.len
        //scs to be acdxyezfkgh = len = 11
        //for printing scs as string we will take lcs string once 
        //and all other string will be included in it 
        //will be performing lcs printing code here
        int[][] memo = new int[aLen + 1][bLen + 1];

        //base cond
        for (int x = 0; x < aLen + 1; x++) {
            for (int y = 0; y < bLen + 1; y++) {
                if (x == 0) {
                    memo[x][y] = 0;
                }
                if (y == 0) {
                    memo[x][y] = 0;
                }
            }
        }

        //LCS top down
        for (int x = 1; x < aLen + 1; x++) {
            for (int y = 1; y < bLen + 1; y++) {

                if (a.charAt(x - 1) == b.charAt(y - 1)) {
                    memo[x][y] = 1 + memo[x - 1][y - 1];
                } else {
                    memo[x][y] = Math.max(memo[x - 1][y], memo[x][y - 1]);
                }

            }
        }

        //traversing memo[][]
        int i = aLen;
        int j = bLen;
        StringBuilder sb = new StringBuilder();
        while (i > 0 && j > 0) {

            if (a.charAt(i - 1) == b.charAt(j - 1)) {
                //if char has matched that means we have come from 
                //diagonally previous element in memo[][]
                //the point of lcs a common char in both string 
                sb.append(a.charAt(i - 1));
                i--;
                j--;
            } else {
                //if char didn't match
                //means we have come from max of either prev row same col (x-1,j)
                //or prev col same row(x, y-1)
                if (memo[i - 1][j] > memo[i][j - 1]) {
                    //a point where prev row same col is greater 
                    //and since i represents aLen we will take a.char(i-1)
                    //in our SCS
                    sb.append(a.charAt(i - 1));
                    i--;
                } else {
                    //similarly here prev col same row is greater
                    //and j represents bLen so here we will take
                    //b.char(j-1)
                    sb.append(b.charAt(j - 1));
                    j--;
                }
            }

        }

        //if in any point of time j reaches to 0 leaving some char in i string
        //we will simply append all i = aLen = a string left char to sb
        while (i > 0) {
            sb.append(a.charAt(i - 1));
            i--;
        }
        //if in any point of time i reaches to 0 leaving some char in j string
        //we will simply append all j = bLen = b string left char to sb
        while (j > 0) {
            sb.append(b.charAt(j - 1));
            j--;
        }

        System.out.println("printing shortest common superseq using LCS printing code variation : " + sb.reverse().toString());

    }

    private static int longestRepeatingSubsequence_Recursion_LCSModification(String a, String b, int aLen, int bLen) {

        if (aLen == 0 || bLen == 0) {
            return 0;
        } else if (a.charAt(aLen - 1) == b.charAt(bLen - 1) && aLen != bLen) {
            return 1 + longestRepeatingSubsequence_Recursion_LCSModification(a, b, aLen - 1, bLen - 1);
        } else {
            return Math.max(longestRepeatingSubsequence_Recursion_LCSModification(a, b, aLen - 1, bLen),
                    longestRepeatingSubsequence_Recursion_LCSModification(a, b, aLen, bLen - 1));
        }

    }

    public static int longestRepeatingSubsequence_Recursion(String a) {

        /*
        
         a longest repeating subseq is a seq that appears more than one in the same string
         ex: "aabebcdd"
         here abd is a seq that is appearing more than one
         i = 0   1   2   3   4   5   6   7
         s = a   a   b   e   b   c   d   d
         j = 0   1   2   3   4   5   6   7
        
         we can make use of LCsubseq logic here 
         lcs adds 1 when the two char in the two string matches 
         here the two strings are going to be the same so when two
         char of these string are found at same location that can form one seq at a time, 
         then for repeating seq the same char should occur more than one and at diff location
         ex: i=j=0 -> a
         i=j=2 -> b
         i=j=7 -> d
         now the same "abd" seq should be repeating more than one and also i!=j location
         the main restriction/variation in basic LCS logic is this 
         (a.char == b.char and i!=j)
         ex: i=j=0 -> a  i=0, j=1 -> a i!=j
         i=j=2 -> b  i=2, j=4 -> b i!=j
         i=j=7 -> d  i=7, j=6 -> d i!=j
        
         */
        return longestRepeatingSubsequence_Recursion_LCSModification(a, a, a.length(), a.length());
    }

    public static int longestRepeatingSubsequence_DP_Memoization(String a, String b, int aLen, int bLen, int[][] memo) {

        /*
         LRS is also a modification of LCS here 
         */
        if (aLen == 0 || bLen == 0) {
            return memo[aLen][bLen] = 0;
        } else if (a.charAt(aLen - 1) == b.charAt(bLen - 1) && aLen != bLen) {
            return memo[aLen][bLen] = 1 + longestRepeatingSubsequence_DP_Memoization(a, b, aLen - 1, bLen - 1, memo);
        } else {
            return memo[aLen][bLen] = Math.max(longestRepeatingSubsequence_DP_Memoization(a, b, aLen - 1, bLen, memo),
                    longestRepeatingSubsequence_DP_Memoization(a, b, aLen, bLen - 1, memo));
        }

    }

    public static int longestRepeatingSubsequence_DP_TopDown(String a, String b, int aLen, int bLen) {

        /*
         LRS is also a modification of LCS here 
         */
        int[][] memo = new int[aLen + 1][bLen + 1];
        //base cond
        for (int[] r : memo) {
            Arrays.fill(r, 0);
        }

        for (int x = 1; x < aLen + 1; x++) {
            for (int y = 1; y < bLen + 1; y++) {

                if (a.charAt(x - 1) == b.charAt(y - 1) && x != y) {
                    memo[x][y] = 1 + memo[x - 1][y - 1];
                } else {
                    memo[x][y] = Math.max(memo[x - 1][y], memo[x][y - 1]);
                }

            }
        }

        return memo[aLen][bLen];

    }

    public static void sequencePatternMatching(String str, String seq) {

        /*
        
         problem is to find wheather a string seq is present in str or not
         seq = "axy" str = "achxgyij"
         in this seq is completely present in the str
        
         observe if 2 string are given s1, s2
         either s1 is completely inside s2 or s2 is completely inside s1
         s1 = "axy" s2 = "achxgyij" or 
         s1 = "abchxgyikj" s2 = "achxgyij"
         in such cases min seq mtching range is 
         0 to min(s1.len, s2.len)
         that means both string didnot matched to min length string is completely inside the other one
        
         also that if any of the given string is inside another that 
         particular string is a subseq of that string
         "axy" in "achxgyij" so basically the LCS of this is axy which is min(seq.len, str.len) == LCS;
        
         here we are asked that if seq is matching/subseq of str or not
        
         */
        int lcs = longestCommonSubsequence_DP_TopDown(str, seq, str.length(), seq.length());
        if (seq.length() == lcs) {
            System.out.println("seq is completely mtched inside str");
        } else {
            System.out.println("seq is not matching inside str");
        }

    }

    public static int minNoOfInsertionInStringToMakeItPallindrome(String a) {

        /*
        
         in our previous method minNoOfDeletionInStringToMakeItPallindrome_LPSBasedApproach()
         we have an example a = "agbcba" if we can delete 1 char (g) then the left over string will be 
         pallindromic
         a = -g = "abcba"
        
         also for string a = "agbcba" the longest pallindromic subseq is abcba 
         using this we got a formula in that method 
         minNoOfDeletion = a.length - LPS.length 
         = "agbcba".length - "abcba".length
         = 1 del(g)
        
         now observe
        
         in this problem statement where we need min no of insertion to make a string pallindromic
         in above we have deleted 'g' from string what if we have inserted one more 'g' to the string
         a= "agbcba" ins(g) a="agbcbga" also pallindromic and min insertion is 1
        
         so basically what happend here is g and g are now pair 
        
         formula here is minNoOfInsertion = minNoOfDeletion
        
         lets take another ex:
         a = "aebcbda" len = 7
         LPS = "abcba" len = 5
        
         minNoOfDeletion = 7-5 = 2 del(e, d)
        
         what if we had extra e and d char in string to make it pallindrome
         if a = "adebcbeda" also pallindrome and now e and d are in pairs (2*e and 2*d)
         what we did min no of char we added in the string is 1 e and 1 d (2 char)
        
         from above  minNoOfInsertion = minNoOfDeletion = 2
        
         */
        return minNoOfDeletionInStringToMakeItPallindrome_LPSBasedApproach(a);

    }

    public static int matrixChainMultiplicationLowCost_Recursion(int[] a, int i, int j) {

        if (i >= j) {
            return 0;
        }

        int min = Integer.MAX_VALUE;

        for (int k = i; k < j; k++) {
            int temp = matrixChainMultiplicationLowCost_Recursion(a, i, k)
                    + matrixChainMultiplicationLowCost_Recursion(a, k + 1, j)
                    + a[i - 1] * a[k] * a[j];

            if (temp < min) {
                min = temp;
            }

        }

        return min;

    }

    public static int matrixChainMultiplicationLowCost_DP_Memoization(int[] a, int i, int j, int[][] memo) {

        if (i >= j) {
            return memo[i][j] = 0;
        }

        if (memo[i][j] != -1) {
            return memo[i][j];
        }

        int min = Integer.MAX_VALUE;
        for (int k = i; k < j; k++) {

            int temp = (memo[i][k] == -1) ? matrixChainMultiplicationLowCost_DP_Memoization(a, i, k, memo) : memo[i][k];
            temp += (memo[k + 1][j] == -1) ? matrixChainMultiplicationLowCost_DP_Memoization(a, k + 1, j, memo) : memo[k + 1][j];
            temp += a[i - 1] * a[k] * a[j];

            if (temp < min) {
                min = temp;
            }

        }

        return memo[i][j] = min;

    }

    public static int pallindromePartitioning_Recursion(String a, int i, int j) {

        /*
        
         pallindrome partioning means a given string 
         "nitik"
         how many min partition can be made sot the partitioned string are pallindrome in itself
         here n | iti | k
         2 partition and partitioned strings are pallinndrome
         let
         i be some left most range 
         j be some right most range
         k will be some intermedite i -> j
         such that we will check all the possibility 
         i -> k and k+1 -> j 
        
         since this recursive fun calls for i, j where logic changes i and j value as k or k+1 respect.
         we will check substrings(i,j) is pallindrome or not if yes not partitioning is needed
        
         if substring(i,j) is not pallindrome we will move next steps
         wee will part the string i -> k and k+1 -> j that cost of this single partition is 1(which is added)
         subproblems will find the min cost of partitioning and will return that
        
         i = 0 j = 6
         k = i = 0
         1st partition i,k and k+1,j k=0 => 0,0 and 1,6 further more partitioning will be done as subproblems
         2nd partition i,k and k+1,j k=1 => 1,1 and 2,6 further more partitioning will be done as subproblems
         ..so on
         
         */
        //if i>j i is left most and j is right most range
        //if on looping and chaning k ranges as k or k+1 respect. i becomes > j that means 
        //string becomes empty and return 0 for that
        //if i == j means a single char is given in string which is pallindrome in itself but we don't need any
        //partitioning for this single char string so return 0
        //base cond i>=j
        if (i >= j) {
            return 0;
        }

        //if string from i,j range is pallindrome return 0
        if (isStringPallindrome(a.substring(i, j))) {
//            System.out.println(a.substring(i,j));
            return 0;
        }

        int min = Integer.MAX_VALUE;

        for (int k = i; k < j; k++) {
//            System.out.println(i+" "+j);
            int temp = pallindromePartitioning_Recursion(a, i, k) + pallindromePartitioning_Recursion(a, k + 1, j) + 1;

            if (temp < min) {
                min = temp;
            }

        }
        return min;
    }

    public static int pallindromePartitioning_DP_Memoization(String a, int i, int j, int[][] memo) {

        if (i >= j || isStringPallindrome(a.substring(i, j))) {
            return memo[i][j] = 0;
        }

        if (memo[i][j] != -1) {
            return memo[i][j];
        }

        int min = Integer.MAX_VALUE;

        for (int k = i; k < j; k++) {

            int temp = (memo[i][k] == -1) ? pallindromePartitioning_DP_Memoization(a, i, k, memo) : memo[i][k];
            temp += (memo[k + 1][j] == -1) ? pallindromePartitioning_DP_Memoization(a, k + 1, j, memo) : memo[k + 1][j];
            temp += 1;

            min = Math.min(min, temp);
        }

        return min;

    }

    public static int booleanParenthersizationThatEqualsMatch(String expr, int i, int j, boolean match) {

        if (i > j) {
            return 0;
        }

        if (i == j) {
            if (match == true) {
                return expr.charAt(i) == 'T' ? 1 : 0;
            } else {
                return expr.charAt(i) == 'F' ? 1 : 0;
            }
        }
        int ans = 0;
        for (int k = i + 1; k < j; k += 2) {

            System.out.println(expr.charAt(k));
            int leftTrue = booleanParenthersizationThatEqualsMatch(expr, i, k - 1, true);
            int leftFalse = booleanParenthersizationThatEqualsMatch(expr, i, k - 1, false);
            int rightTrue = booleanParenthersizationThatEqualsMatch(expr, k + 1, j, true);
            int rightFalse = booleanParenthersizationThatEqualsMatch(expr, k + 1, j, false);
            System.out.println(leftTrue + " " + leftFalse + " " + rightTrue + " " + rightFalse);
            if (expr.charAt(k) == '&') {

                //and based expr
                //result = expr1 & expr2
                if (match == true) {
                    //if we want our result to true
                    //expr1 & expr2 req
                    //T & T only
                    ans += leftTrue * rightTrue;
                } else {
                    //if we want our result to false
                    //expr1 & expr2 req
                    //T & F
                    ans += leftTrue * rightFalse;
                    //F & T
                    ans += leftFalse * rightTrue;
                    //F & F
                    ans += leftFalse * rightFalse;
                }

            } else if (expr.charAt(k) == '|') {

                //or based expr
                //result = expr1 | expr2
                if (match == true) {
                    //if we want our result to true
                    //expr1 | expr2 req
                    //T | T
                    ans += leftTrue * rightTrue;
                    //T | F
                    ans += leftTrue * rightFalse;
                    //F | T
                    ans += leftFalse * rightTrue;
                } else {
                    //if we want our result to false
                    //expr1 | expr2 req
                    //F | F
                    ans += leftFalse * rightFalse;
                }

            } else if (expr.charAt(k) == '^') {

                //xor based expr
                //result = expr1 ^ expr2
                if (match == true) {
                    //if we want our result to true
                    //expr1 ^ expr2 req
                    //T ^ F
                    ans += leftTrue * rightFalse;
                    //F ^ T
                    ans += leftFalse * rightTrue;
                } else {
                    //if we want our result to false
                    //expr1 ^ expr2 req
                    //T ^ T
                    ans += leftTrue * rightTrue;
                    //F ^ F
                    ans += leftFalse * rightFalse;
                }

            }

        }

        return ans;

    }

    public static boolean scrambledString(String a, String b) {

        if (a.length() != b.length()) {
            return false;
        }
        if (a.length() == 0 || a.equals(b)) {
            return true;
        }

        char[] arr1 = a.toCharArray();
        char[] arr2 = b.toCharArray();
        Arrays.sort(arr1);
        Arrays.sort(arr2);
        if (!new String(arr1).equals(new String(arr2))) {
            return false;
        }

//        System.out.println(a + " " + b);
        int n = a.length();

        for (int i = 1; i < n; i++) {
            String s11 = a.substring(0, i);
            String s12 = a.substring(i, a.length());
            String s21 = b.substring(0, i);
            String s22 = b.substring(i, b.length());
            String s23 = b.substring(0, b.length() - i);
            String s24 = b.substring(b.length() - i, b.length());

            System.out.println(s11 + " " + s12 + " " + s21 + " " + s22 + " " + s23 + " " + s24);

            if (scrambledString(s11, s21) && scrambledString(s12, s22)) {
                return true;
            }
            if (scrambledString(s11, s24) && scrambledString(s12, s23)) {
                return true;
            }

        }

        return false;

    }

    public static int eggDroppingTrials(int floor, int eggs) {

        if (floor == 0 || floor == 1) {
            return floor;
        }

        if (eggs == 1) {
            return floor;
        }

        int min = Integer.MAX_VALUE;
        int res = 0;
        for (int k = 1; k <= floor; k++) {

            res = Math.max(eggDroppingTrials(k - 1, eggs - 1), eggDroppingTrials(floor - k, eggs));
            if (res < min) {
                min = res;
            }
        }

        return min + 1;

    }

//    private static int $=10;
    public static void main(String[] args) throws Exception {

//        StringBuilder b = "hacker";
//        b.append(4).deleteCharAt(3).delete(3, b.length() - 1);
//        System.out.println(b);
//        int[] arr = new int[]{1, 2, 1, 3, 4, 2, 3};
//        int k =4;       
//        System.out.println("countDistinctOnwWindowK");
//        int[] result = countDistinctOnwWindowK(arr, k);
//        for(int x: result){
//            System.out.print(x+" ");
//        }
//...............................................................        
//        System.out.println("subset of string");
//        String str = "aaaabbaa";
//        List<String> s = subsetOfString(str);
//        s.stream().forEach(x -> System.out.println(x));
//.................................................................        
//        System.out.println("subset of Integer");
//        int n = 2563;
//        List<Integer> i = subsetOfInteger(n);
//        i.stream().forEach(x -> System.out.println(x));
//...................................................................        
//        System.out.println("seq is subsequence of str");
//        String seq = "gksrek";
//        String str1 = "geeksforgeeks";
//        int m = str1.length();
//        int n1 = seq.length();
//        System.out.println(isSubsequence(str1, seq, m, n1));
//.......................................................................        
//        System.out.println(" Given a set of numbers, you need to find out the number of ways you "
//                + "can divide the set into two groups such that no two groups are left empty.");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-for-sde-1-7/?ref=leftbar-rightbar
//        System.out.println(arrangeSetOfNumberInKGroup(new int[]{2,1,3,6,7,10,0}, 2));
//........................................................................
//        System.out.println(" Given two strings containing a special character ‘#’ which represents a backspace, "
//                + "you need to print true if both the strings will be equal after processing the backspaces");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-for-sde-1-7/?ref=leftbar-rightbar
//        //# means backspace operation string
//        System.out.println(stringCompare("AA##BCAS#", "B#BCA"));
//...........................................................................
//        System.out.println("recursion stack check");
//        recursionCheck(5);
//..............................................................................
//        System.out.println("Simple Binary search");
//        //mid calculation inn effective way
//        System.out.println("key found at: "+simpleBinarySearch(new int[]{1,2,4,5,6,7,8,9,10}, 2));
//..............................................................................
//        System.out.println("Given a string, write a program to find longest length palindrome "
//                + "from that given string");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-153-sde1/?ref=rp
//        //System.out.println(longestPallindromicSubsetFromAString("aaaabbaa"));
//        System.out.println(longestPallindromicSubsetFromAString("ELRMENMET")); //fail case
//.............................................................................
//        System.out.println("Longest pallindromic subsequence from string");
//        String str = "ELRMENMET";
//        //System.out.println(longestPallindromicSubsetFromAString("aaaabbaa"));
//        System.out.println(longestPanllindromicSubsequenceFormAString(str, 0, str.length()-1)); //fail case
//.............................................................................
//        System.out.println("Q1. Given two link list that represents no. "
//                + "write a program to add two given two link list and return new link list that represents sum of no");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-153-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/sum-of-two-numbers-represented-as-arrays/0
//        int[] a = new int[]{5, 4, 8};
//        int[] b = new int[]{ 6, 2};
//        int[] sum = sumOfArrayAsNumberForm1(a, b);
//        for (int i : sum) {
//            System.out.print(i + " ");
//        }
//         System.out.println();
//        sum = sumOfArrayAsNumberForm2(a, b);
//        for (int i : sum) {
//            System.out.print(i + " ");
//        }
//..............................................................................
//        System.out.println("67. Add Binary represented as string");
//        //https://leetcode.com/problems/add-binary/
//        addBinaryStrings("11", "1");
//..............................................................................
//        System.out.println("415. Add Strings");
//        //https://leetcode.com/problems/add-strings/
//        addTwoNumbersAsStrings("548", "162");
//..............................................................................
//        System.out.println("258. Add Digits");
//        //https://leetcode.com/problems/add-digits/
//
//        /*
//        
//         Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
//
//         Example:
//
//         Input: 38
//         Output: 2 
//         Explanation: The process is like: 3 + 8 = 11, 1 + 1 = 2. 
//         Since 2 has only one digit, return it.
//         Follow up:
//         Could you do it without any loop/recursion in O(1) runtime?
//        
//         */
//        addDigits(38);
//.............................................................................
//        System.out.println("fibbonacci by D and Q");
//        System.out.println(fibbonacci(1));
//.............................................................................
//        System.out.println("number factor by D and Q create no of ways to create N using 1,3,4 only");
//        System.out.println(numberFactor(4));
//.............................................................................
//        System.out.println("convert a number into its word form");
//        numberToWordsConverter(9999);
//        numberToWordsConverter(1212);
//        numberToWordsConverter(10);
//        numberToWordsConverter(1);
//        numberToWordsConverter(0);
//        numberToWordsConverter(100);
//        numberToWordsConverter(324);
//        numberToWordsConverter(1000);
//.............................................................................
//        System.out.println("3. Find median in a stream");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-188-for-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/find-median-in-a-stream/0
//        medianRecurrenceInAStream(new int[]{5, 15, 1, 3});
//.............................................................................
//        System.out.println("2. Given a number say 12345, find the immediate next number using "
//                + "the same digits, in this case 12354.");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-262-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/find-next-greater-number-set-digits/
//        //.............O(N^2)
//        nextPermutation(new int[]{5, 3, 4, 9, 7, 6});
//        nextPermutation(new int[]{1, 2, 3, 4});
//        nextPermutation(new int[]{4, 3, 2, 1});
//        nextPermutation(new int[]{1, 2});
//        nextPermutation(new int[]{7, 4, 3, 5, 6});
//        nextPermutation(new int[]{2, 3, 1}); //fail case
//        //..........O(N)
//        //https://leetcode.com/problems/next-permutation/
//        nextPermutation_2(new int[]{5, 3, 4, 9, 7, 6});
//        nextPermutation_2(new int[]{4, 3, 2, 1});
//        nextPermutation_2(new int[]{2, 3, 1}); //working
//.............................................................................
//        System.out.println("1. Given a Hotel and checkin/Checkout time of visitors, find the maximum numbers of "
//                + "rooms required. (different version of Trains/Platform question) similar to Minimum Platforms");
////        //https://www.geeksforgeeks.org/amazon-interview-experience-set-262-for-sde1/?ref=rp
////        //https://practice.geeksforgeeks.org/problems/minimum-platforms/0
//        //https://www.geeksforgeeks.org/minimum-number-platforms-required-railwaybus-station/
//        noOfPlatformNeeded(new int[]{900, 940, 950, 1100, 1500, 1800},
//                new int[]{910, 1200, 1120, 1130, 1900, 2000});
//        noOfPlatformNeeded(new int[]{900, 1100, 1235},
//                new int[]{1000, 1200, 1240});
//.............................................................................
//        System.out.println("Longest subsequence consisting of alternate vowels and consonants");
//        //https://www.geeksforgeeks.org/longest-subsequence-consisting-of-alternate-vowels-and-consonants/?ref=leftbar-rightbar
//        longestSubsequenceWithAltVowels("geeksforgeeks");
//        longestSubsequenceWithAltVowels("sangeet");
//        longestSubsequenceWithAltVowels("elephant");
//.............................................................................
//        System.out.println("addition of 2 no without + operator");
//        addWithoutPlus(5, 7);
//.............................................................................
//        System.out.println("2. Given a general stack, design an advanced DS, such that getMin(),getMax() happens in o(1). "
//                + "Many cross questions on this. about optimizations and all.");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-262-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/design-a-stack-that-supports-getmin-in-o1-time-and-o1-extra-space/
//        SpecialStack s = new SpecialStack();
//        s.push(3);
//        System.out.println("min: "+s.getMin()); 
//        s.push(2);
//        System.out.println("min: "+s.getMin()); 
//        s.push(-1);
//        System.out.println("min: "+s.getMin()); 
//        s.push(1);
//        System.out.println("min: "+s.getMin()); 
//        System.out.println("peek: "+s.peek());
//        System.out.println("pop: "+s.pop());
//        System.out.println("min: "+s.getMin()); 
//        System.out.println("peek: "+s.peek());
//        System.out.println("pop: "+s.pop());
//        System.out.println("min: "+s.getMin()); 
//        System.out.println("peek: "+s.peek());
//        System.out.println("pop: "+s.pop());
//        System.out.println("min: "+s.getMin()); 
//        System.out.println("peek: "+s.peek());
//.............................................................................
//        System.out.println("Gas Station");
//        //https://www.interviewbit.com/problems/gas-station/
//        //https://www.geeksforgeeks.org/find-a-tour-that-visits-all-stations/
//        petrolCycle(new int[]{1, 2}, new int[]{2, 1});
//        petrolCycle(new int[]{2, 1}, new int[]{1, 2});
//        petrolCycle(new int[]{4, 6, 2, 4}, new int[]{6, 5, 3, 5});
//        //special case
//        petrolCycle(new int[]{4, 6, 7, 4}, new int[]{6, 5, 3, 5});//cycle indx 1->3->1
//        petrolCycle(new int[]{2, 6, 7, 4}, new int[]{6, 5, 3, 5});
//        petrolCycle(new int[]{1, 6, 7, 4}, new int[]{6, 5, 3, 5});//not cycle indx 1->3 but ->1 fails
//        //special case
//        petrolCycle(new int[]{1, 2, 3, 4}, new int[]{4, 3, 2, 1});
//..............................................................................
//        System.out.println("Majority element");
//        //https://www.interviewbit.com/problems/majority-element/
//        majorityElement(new int[]{2, 1, 2});
//        majorityElement(new int[]{2, 1, 2, 2, 1, 1});
//..............................................................................
//        System.out.println("Longest prefix which is also suffix");
//        //Longest prefix which is also suffix
//        //https://www.geeksforgeeks.org/longest-prefix-also-suffix/
//        longestPrefixAlsoSuffixInString("aabcdaabc");
//        longestPrefixAlsoSuffixInString("abcab");
//        longestPrefixAlsoSuffixInString("sangeet");
//        longestPrefixAlsoSuffixInString("aaaa");
//        longestPrefixAlsoSuffixInString("blablabla");
//..............................................................................
//        System.out.println("Next largest element to right");
//        //https://www.geeksforgeeks.org/next-greater-element-in-same-order-as-input/
//        nextLargestNoToRight(new int[]{4, 5, 2, 1, 25});
//        nextLargestNoToRight(new int[]{4, 5, 2, 3, 25});
//..............................................................................
//        System.out.println("Next largest element to left");
//        nextLargestNoToLeft(new int[]{4, 5, 2, 1, 25});
//        nextLargestNoToLeft(new int[]{4, 5, 2, 3, 25});
//..............................................................................
//        System.out.println("Next smallest element to right");
//        nextSmallestNoToRight(new int[]{4, 5, 2, 1, 25});
//        nextSmallestNoToRight(new int[]{4, 5, 2, 3, 25});
//..............................................................................
//        System.out.println("Next smallest element to left");
//        nextSmallestNoToLeft(new int[]{4, 5, 2, 1, 25});
//        nextSmallestNoToLeft(new int[]{4, 5, 2, 3, 25});
//..............................................................................
//        System.out.println("Find the number of Islands");
//        //https://www.geeksforgeeks.org/find-the-number-of-islands-set-2-using-disjoint-set/
//        int[][] a = new int[][] {{1, 1, 0, 0, 0}, 
//                                 {0, 1, 0, 0, 1}, 
//                                 {1, 0, 0, 1, 1}, 
//                                 {0, 0, 0, 0, 0}, 
//                                 {1, 0, 1, 0, 1} 
//                                }; 
//        noOfIslands(a);
//..............................................................................
//        System.out.println("Find k-th smallest element in BST (Order Statistics in BST)");
//        //https://www.geeksforgeeks.org/find-k-th-smallest-element-in-bst-order-statistics-in-bst/
//        //to find kth element of Binary search tree use inorder traversal
//        //Left-Root-Right
//        //Left - (count++ until count==kth) - Right
//..............................................................................
//        System.out.println("Given an array 0 and 1 find the largest subarray that caintains equal no of 0 and 1");
//        //https://www.geeksforgeeks.org/largest-subarray-with-equal-number-of-0s-and-1s/
//        largestSubarrayWithEqual0and1(new int[]{0,1,0,1});
//        largestSubarrayWithEqual0and1(new int[]{0,0,1,1,1,0,0});
//        largestSubarrayWithEqual0and1(new int[]{0,0,0,0});
//        largestSubarrayWithEqual0and1(new int[]{0,0,1,1,0,0,0,0,0,0,1,1,1});
//..............................................................................
//        System.out.println("Given k sorted list provide a single sorted list from all of them");
//        //https://www.geeksforgeeks.org/largest-subarray-with-equal-number-of-0s-and-1s/
//        List<List<Integer>> k = new ArrayList<>();
//        k.add(Arrays.asList(1,2,4,6));
//        k.add(Arrays.asList(10,13,17,20));
//        k.add(Arrays.asList(7,11,23,16));
//        kSortedListAsSingle(k);
//..............................................................................
//        System.out.println("Stack | Set 2 (Infix to Postfix)");
//        //https://www.geeksforgeeks.org/stack-set-2-infix-to-postfix/
//        //https://www.geeksforgeeks.org/amazon-interview-experience-off-campus-for-sde-1-2/?ref=leftbar-rightbar
//        expressionInfixToPostfix("a+b*(c^d-e)^(f+g*h)-i");
//        expressionInfixToPostfix("(a+b)^c");
//..............................................................................
//        System.out.println("Given an array, find the number of sub-arrays having even sum");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-off-campus-for-sde-1-2/?ref=leftbar-rightbar
//        //https://www.geeksforgeeks.org/find-number-subarrays-even-sum/
//        countSubarrayWithEvenSumOfElements(new int[]{1, 2, 2, 3, 4, 1});
//..............................................................................
//        System.out.println("Given an array of 0s and 1s, and a number m, you can flip maximum m zeroes, "
//                + "count the maximum length of 1s you can make by flipping at max m zeroes");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-off-campus-for-sde-1-2/?ref=leftbar-rightbar
//        //https://www.geeksforgeeks.org/find-zeroes-to-be-flipped-so-that-number-of-consecutive-1s-is-maximized/
//        m0sFlipToFindMaxLength1s(new int[]{1, 0, 0, 1, 1, 0, 1, 0, 1, 1}, 3);
//..............................................................................
//        System.out.println(" Print the boundary traversal of a binary tree.");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-off-campus-for-sde-1-2/?ref=leftbar-rightbar
//        //https://www.geeksforgeeks.org/boundary-traversal-of-binary-tree/
//        BinaryTree<String> root = new BinaryTree<>("A");
//        root.insert("B");
//        root.insert("C");
//        root.insert("D");
//        root.insert("E");
//        root.insert("F");
//        root.insert("G");
//        root.insert("H");
//        root.insert("I");
//        root.insert("J");
//        root.insert("K");
//        //level order
//        root.treeBFS();
//        System.out.println();
//        //inorder
//        root.treeInorder();
//        System.out.println();
//        //postorder
//        root.treePostorder();
//        System.out.println();
//        //preorder
//        root.treePreorder();
//        System.out.println();
//        //outer boundary
//        root.treeOuterBoundry();
//        System.out.println();
//        System.out.println("the height of tree is node-wise "+root.treeHeight());
//        System.out.println("the height of tree is node-edge-wise "+(root.treeHeight()-1));
//        System.out.println("tree left view "+ root.leftView());
//..............................................................................       
//        System.out.println("ZigZag Tree Traversal");
//        //https://www.geeksforgeeks.org/zigzag-tree-traversal/
//        BinaryTree<String> root = new BinaryTree<>("A");
//        root.insert("B");
//        root.insert("C");
//        root.insert("D");
//        root.insert("E");
//        root.insert("F");
//        root.insert("G");
//        root.insert("H");
//        root.insert("I");
//        root.insert("J");
//        root.insert("K");
//        System.out.println();
//        root.treeZigZag(true);
//        System.out.println();
//        root.treeZigZag(false);
//..............................................................................
//        System.out.println("Given a dictionary in which the length of all the words are equal, "
//                + "you are allowed to change just one character. "
//                + "Given a starting word and an ending word, "
//                + "what will be the smallest number of steps required to change the starting word to the ending word");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-off-campus-for-sde-1-2/?ref=leftbar-rightbar
//        //https://www.geeksforgeeks.org/word-ladder-length-of-shortest-chain-to-reach-a-target-word/
//        Set<String> D = new HashSet<String>();
//        D.add("poon");
//        D.add("plee");
//        D.add("same");
//        D.add("poie");
//        D.add("plie");
//        D.add("poin");
//        D.add("plea");
//        String start = "toon";
//        String target = "plea";
//        startToEndFromDictionary(D, start, target);
//..............................................................................
//        System.out.println("Given a pointer to a node in a linked list, delete the given node in O(1)");
//        
//        /*Given only a pointer/reference to a node to be deleted in a singly linked list, how do you delete it?
//        
//        note: Given a pointer to a node to be deleted, delete the node. 
//        Note that we don’t have pointer to head node.
//        
//        */
//        //https://www.geeksforgeeks.org/amazon-interview-experience-off-campus-for-sde-1-2/?ref=leftbar-rightbar
//        //https://www.geeksforgeeks.org/given-only-a-pointer-to-a-node-to-be-deleted-in-a-singly-linked-list-how-do-you-delete-it/
//        
//        /*
//        
//        head of singly linked list is not provided
//        only the node to be deleted is given
//        delete this node in O(1) time
//        
//        Head........ -> A -> B -> C -> Tail -> null
//        
//        lets assume we are provided only B node to delete and we have no access to Head
//        //solution:
//        //swapping up data and next ref is O(1)time
//        Node temp = B.next //basically B.next holds ref to C node so temp = C
//        B.data = temp.data;
//        B.next = temp.next;
//        temp = null; //un-ref temp (C) node to go System.gc()
//        
//        In this we have moved the C node to B node and un-ref C after this
//        
//        Head........ -> A -> B -> Tail -> null
//        
//        but B holds C data and technically B data has been made lost
//        
//        Head........ -> A -> C -> Tail -> null
//        
//        */
//..............................................................................
//        System.out.println(" Find the first non-repeated character in a string");
//        //geeksforgeeks.org/amazon-interview-set-41-campus/
//        //https://www.geeksforgeeks.org/given-a-string-find-its-first-non-repeating-character/
//        firstNonRepeatingCharInString("abcbcb");
//        firstNonRepeatingCharInString("xbcabcb");
//        firstNonRepeatingCharInString("sangeet");
//        firstNonRepeatingCharInString("xyz");
//        firstNonRepeatingCharInString("xxyyzz");
//..............................................................................
//        System.out.println("Check if a given tree is a Binary Search Tree or not. ");
//        //geeksforgeeks.org/amazon-interview-set-41-campus/
//        //https://www.geeksforgeeks.org/a-program-to-check-if-a-binary-tree-is-bst-or-not/
//        BinarySearchTree<Integer> root = new BinarySearchTree<>(10);
//        root.insert(5);
//        root.insert(15);
//        root.insert(2);
//        root.insert(7);
//        root.insert(12);
//        root.insert(17);
//        //level order
//        root.treeBFS();
//        System.out.println();
//        root.treeInorder();
//        try {
//            System.out.println("Tree is BST " + root.isBST());
//        } catch (Exception ex) {
//            Logger.getLogger(SomePracticeQuestion.class.getName()).log(Level.SEVERE, null, ex);
//        }
//..............................................................................
//        System.out.println("You are given an array whose each element represents the height of the tower. "
//                + "The width of every tower is 1. "
//                + "It starts raining. How much water is collected between the towers");
//        //geeksforgeeks.org/amazon-interview-set-41-campus/
//        rainWaterTrapping(new int[]{1, 5, 2, 7, 3});
//        rainWaterTrappingUsingStack(new int[]{1, 5, 2, 7, 3});
//        rainWaterTrappingUsingTwoPointers(new int[]{1, 5, 2, 7, 3});
//        rainWaterTrapping(new int[]{8, 2, 4, 6, 8});
//        rainWaterTrappingUsingStack(new int[]{8, 2, 4, 6, 8});
//        rainWaterTrappingUsingTwoPointers(new int[]{8, 2, 4, 6, 8});
//        rainWaterTrapping(new int[]{1, 3, 6, 3, 1}); //WRONG ANS
//        rainWaterTrappingUsingStack(new int[]{1, 3, 6, 3, 1}); //RIGHT ANS
//        rainWaterTrappingUsingTwoPointers(new int[]{1, 3, 6, 3, 1}); ////RIGHT ANS
//        rainWaterTrapping(new int[]{1, 0, 1});
//        rainWaterTrappingUsingStack(new int[]{1, 0, 1});
//        rainWaterTrappingUsingTwoPointers(new int[]{1, 0, 1});
//        rainWaterTrapping(new int[]{ 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1});
//        rainWaterTrappingUsingStack(new int[]{ 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1});
//        rainWaterTrappingUsingTwoPointers(new int[]{ 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1});
//..............................................................................
//        System.out.println("3) Given an array and a fixed window size X, "
//                + "you have to find out the minimum value "
//                + "from every window. with and without Deque");
//        //geeksforgeeks.org/amazon-interview-set-41-campus/
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-291-on-campus-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-size-k/
//        maxElementInKWindow_ON2(new int[]{1, 2, 3, 1, 4, 5, 2, 3, 6}, 3);
//        maxElementInKWindow_ON(new int[]{1, 2, 3, 1, 4, 5, 2, 3, 6}, 3);
//..............................................................................
//        System.out.println("You have an array whose elements firstly strictly "
//                + "increase and then strictly decrease. "
//                + "You have to find the point of change");
//        //geeksforgeeks.org/amazon-interview-set-41-campus/
//        //https://www.geeksforgeeks.org/find-element-bitonic-array/
//        bitonicArray(new int[]{-3, 1, 3, 5, 20, 17, 15, 0}); //O(n)
//        //finding a key in bitonic array in O(log n) 
//        bitonicArrayBinarySearch(new int[]{-3, 1, 3, 30, 20, 17, 15, 0}, 0, 8, 30); 
//..............................................................................
//        System.out.println("A complete path in a tree is from a root to a leaf. "
//                + "A k-heavy path is a complete path whose sum of elements is greater than k. "
//                + "Write a code to delete all nodes which are not in any of the k-heavy paths");
//        //geeksforgeeks.org/amazon-interview-set-41-campus/
//        //https://www.geeksforgeeks.org/remove-all-nodes-which-lie-on-a-path-having-sum-less-than-k/
//        BinaryTree<Integer> root = new BinaryTree<>(1);
//        root.insert(2);
//        root.insert(3);
//        root.insert(4);
//        root.insert(5);
//        root.insert(6);
//        root.insert(7);
//        root.insert(3);
//        root.insert(9);
//        root.insert(12);
//        root.insert(2);
//        root.insert(5);
//        //level order
//        root.treeBFS();
//        System.out.println();
//        root.kSumPathOfTree(15);
//        root.treeBFS();
//        System.out.println();
//.............................................................................. 
//        System.out.println("Creating binary tree from given inorder and preorder array");
//        //https://www.geeksforgeeks.org/construct-tree-from-given-inorder-and-preorder-traversal/
//        BinaryTree<String> tree = new BinaryTree<>();
//        
//        tree = tree.buildTreeFromInorderPreorder(new String[] {"D", "B", "E", "A", "F", "C"}, 
//        new String[] {"A", "B", "D", "E", "C", "F"});
//        tree.treeBFS();
//        
//        tree = tree.buildTreeFromInorderPreorder(new String[] {"H", "D", "I", "B", "J", "E", "K", "A", "F", "C", "G"}, 
//        new String[] {"A", "B", "D", "H", "I", "E", "J", "K", "C", "F", "G"});
//        System.out.println();
//        tree.treeBFS();
//..............................................................................
//        System.out.println("Boggle-Solver with direction option "
//                + "limited to up, down, right and left");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde1-2/?ref=rp
//        //https://www.geeksforgeeks.org/boggle-find-possible-words-board-characters/
//        char boggle[][] = {{'G', 'I', 'Z'},
//        {'U', 'E', 'K'},
//        {'Q', 'S', 'E'}};
//        Set<String> dictionary = new HashSet<>();
//        dictionary.add("GEEKS");
//        dictionary.add("FOR");
//        dictionary.add("QUIZ");
//        dictionary.add("GO");
//        boggleSolver(dictionary, boggle);
//        boggleSolver_Trie(dictionary, boggle);
//..............................................................................
//        System.out.println("Maximum product of a triplet (subsequence of size 3) in array");
//        //https://www.geeksforgeeks.org/find-maximum-product-of-a-triplet-in-array/
//        //O(n^3) > o(n * Log n) > O(n)
//        maxProductOfTriplet_ON3(new int[]{10, 3, 5, 6, 20});
//        maxProductOfTriplet_ONLogN(new int[]{10, 3, 5, 6, 20});
//        maxProductOfTriplet_ON(new int[]{10, 3, 5, 6, 20});
//        maxProductOfTriplet_ON(new int[]{-10, -3, -5, -6, -20});
//..............................................................................
//        System.out.println("Coin change problem Dynamic programming");
//        coinChangeProblem(new int[]{1, 5, 10}, 12);
//..............................................................................
//        System.out.println("Construct array problem hackerrank");
//        constructTheArray(4, 3, 2);
//..............................................................................
//        System.out.println("Trie Data structure implementation");
//        Trie trie = new Trie();
//        
//        trie.insert("h");
//        trie.insert("hello");
//        trie.insert("hell");
//        trie.insert("get");
//        trie.insert("getting");
//        trie.insert("geek");
//        trie.insert("garden");
//        trie.insert("hill");
//        trie.insert("hilltop");
//        trie.insert("geekook");
//
//        trie.print();
//        
//        trie.wordCount("hello");
//        trie.insert("hello");
//        trie.wordCount("hello");
//        
//        trie.autoSuggestQuery("h");
//        trie.autoSuggestQuery("g");
//        trie.autoSuggestQuery("gt");
//        
//        trie.query("hill");
//        trie.query("hell");
//        trie.query("helloo");
//        
//        trie.delete("get");
//        trie.wordCount("get");
//        trie.print();
//        
//        trie.insert("get");
//        trie.delete("getting");
//        trie.wordCount("get");
//        trie.wordCount("getting");
//        trie.print();
//        
//        trie.delete("get");
//        trie.print();
//        trie.delete("getting");
//        trie.print();
//        
//        trie.query("get");
//        
//        trie.update("geekook", "geeksforgeeks");
//        
//        trie.print();
//        
//        //...........case 1
////        trie.delete("hill");
////        trie.print();
////        trie.autoSuggestQuery("h");
////        
////        trie.delete("hilltop");
////        trie.print();
////        trie.autoSuggestQuery("h");
//        //...........case 1
//        
//        //...........case 2
//        trie.delete("hilltop");
//        trie.print();
//        trie.autoSuggestQuery("h");
//        
//        trie.delete("hill");
//        trie.print();
//        trie.autoSuggestQuery("h");
//        //...........case 2
//        
//        trie.query("hillt");
//        
//        trie.autoSuggestQuery("g");
//..............................................................................
//        System.out.println("Linked list Data structure implementation");
//        LinkedListUtil<Integer> ll = new LinkedListUtil<>();
//        ll.append(1);
//        ll.append(2);
//        ll.append(3);
//        ll.append(4);
//        ll.append(5);
//        ll.append(6);    
//        ll.print();
//        ll.addAtHead(0);
//        ll.addAtTail(10);
//        ll.print();
//        System.out.println(ll.length());
//        ll.add(7, 8);
//        ll.print();
//        ll.add(11, 10);
//        ll.print();
//        ll.deleteAtTail();
//        ll.print();
//        ll.deleteAtHead();
//        ll.print();
//        ll.delete(8);
//        ll.print();
//        ll.delete(3);
//        ll.print();
//        ll.delete(1);
//        ll.print();
//        LinkedListUtil<Integer> reversed = ll.reverse();
//        reversed.print();
//        for(int i=1; i<=5; i++){
//            ll.delete();
//        }
//        System.out.println(ll.length());
//..............................................................................
//        System.out.println("A number is represented by a LinkedList, where each node represent a digit of number."
//                + " You are given two such number, find sum of two numbers. "
//                + "You need to return head of LinkedList representing sum of two given numbers.");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde-1-feb-2020-exp-1-5-yr/?ref=leftbar-rightbar
//        LinkedListUtil<Integer> a = new LinkedListUtil<>();
//        a.append(5);
//        a.append(4);
//        a.append(8);
//        LinkedListUtil<Integer> b = new LinkedListUtil<>();
//        b.append(4);
//        b.append(6);
//        b.append(2);
//        sumOfNumberAsLinkedList(a, b);
//..............................................................................
//        System.out.println("Given a string say “ABAABCD”. "
//                + "Calculate minimum number of letters to be "
//                + "removed such that remaining letters can form a palindrome string");
//
//        /*
//         Answer for “ABAABCD” is : 2
//         Explanation : Remove C and D, remaining string is : “ABAAB” which can form a palindrome(BAAAB)
//         Approach : Simply count the number of odd characters. Since you can keep one character of odd count hence answer will be odd character -1. I used HashMap for storing characters and their count.
//         if(odd_characters==0) return 0;
//         return odd_characters-1;
//         */
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde-1-feb-2020-exp-1-5-yr/?ref=leftbar-rightbar
//        //......O(N)
//        removeCharToLeftPallindrome("ABAABCD");
//        removeCharToLeftPallindrome("ABAABCCD");
//        removeCharToLeftPallindrome("ABBBA");
//        removeCharToLeftPallindrome("ABA");
//..............................................................................
//        System.out.println("Given an sorted array of 0’s and 1’s in non-decreasing order. "
//                + "Find the sum of array in O(log n)");
//
//        /*
//         Approach : Apply Binary Search to find the position of first 1 and return n-position+1
//         */
//        https://www.geeksforgeeks.org/amazon-interview-experience-sde-1-feb-2020-exp-1-5-yr/?ref=leftbar-rightbar
//        count1sInNonDecrBinaryArray_OLongN(new int[]{0, 0, 0, 1, 1, 1});
//        count1sInNonDecrBinaryArray_OLongN(new int[]{1, 1, 1, 1, 1, 1, 1});
//        count1sInNonDecrBinaryArray_OLongN(new int[]{0, 0, 0, 0, 0, 0, 1});
//        count1sInNonDecrBinaryArray_OLongN(new int[]{0, 0, 0, 0, 0, 0, 0});
//        count1sInNonDecrBinaryArray_OLongN(new int[]{1});
//        System.out.println("Given an sorted array of 0’s and 1’s in non-increasing order. "
//                + "Find the sum of array in O(log n)");
//        /*
//         Approach : Apply Binary Search to find the position of last 1 and return position+1
//         */
//        count1sInNonIncrBinaryArray_OLongN(new int[]{1, 1, 0, 0, 0});
//        count1sInNonIncrBinaryArray_OLongN(new int[]{1, 1, 1, 1, 0});
//        count1sInNonIncrBinaryArray_OLongN(new int[]{1, 1, 1, 1, 1, 1, 1});
//        count1sInNonIncrBinaryArray_OLongN(new int[]{1});
//        count1sInNonIncrBinaryArray_OLongN(new int[]{1, 0});
//        count1sInNonIncrBinaryArray_OLongN(new int[]{0, 0, 0, 0, 0, 0, 0});
//..............................................................................
//        System.out.println("Given an array of Integers, find and replace next "
//                + "smaller element of each element in the given array in O(n)");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde-1-feb-2020-exp-1-5-yr/?ref=leftbar-rightbar
//        nextFirstSmallerElement_ON2(new int[]{11, 13, 21, 3});
//        nextFirstSmallerElement_ON(new int[]{11, 13, 21, 3});
//        //worst case time comlexity  as inner while will run for all element
//        nextFirstSmallerElement_ON(new int[]{2, 4, 6, 10, 13});
//        nextFirstSmallerElement_ON2(new int[]{2, 4, 6, 10, 13});
//        //worst case space complexity as stack will be filled with all elements
//        nextFirstSmallerElement_ON(new int[]{13, 10, 6, 4, 3});
//        nextFirstSmallerElement_ON2(new int[]{13, 10, 6, 4, 3});
//..............................................................................
//         System.out.println("Given a number n which represent total stairs. "
//                 + "Find in how many ways you can reach the nth stair with 1 or 2 steps at a time");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde-1-feb-2020-exp-1-5-yr/?ref=leftbar-rightbar
//        nStairWaysProblem_DP(3);
//        nStairWaysProblem_DP(4);
//..............................................................................
//        System.out.println("Given a binary search tree(BST), find top view of given BST.");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde-1-feb-2020-exp-1-5-yr/?ref=leftbar-rightbar
//        //https://www.geeksforgeeks.org/print-nodes-top-view-binary-tree/
//        BinaryTree<String> root = new BinaryTree<>("A");
//        root.insert("B");
//        root.insert("C");
//        root.insert("D");
//        root.insert("E");
//        root.insert("F");
//        root.insert("G");
//        System.out.println();
//        root.treeBFS();
//        System.out.println();
//        root.treeTopView();
//..............................................................................
//        System.out.println("Given a 2-D matrix of 0’s and 1’s, where 1 represents an infected person and 0 "
//                + "represents an uninfected person. After each second an infected person infects "
//                + "his 4 uninfected neighbors(L, R, U, D). "
//                + "Need to calculate time such that all becomes infected");
//        //similar to rotten oranges problem
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde-1-feb-2020-exp-1-5-yr/?ref=leftbar-rightbar
//        //https://www.geeksforgeeks.org/minimum-time-required-so-that-all-oranges-become-rotten/
//        int basket[][] = {{2, 1, 0, 2, 1},
//        {1, 0, 1, 2, 1},
//        {1, 0, 0, 2, 1}};
//        rottenOranges(basket);
//        int basket2[][] = {{2, 1, 0, 2, 1},
//        {0, 0, 1, 2, 1},
//        {1, 0, 0, 2, 1}};
//        //(2,0) will be left un rotten so ans is not possible -1
//        rottenOranges(basket2);
//..............................................................................
//        System.out.println("Kth Smallest element in stream of array");
//        //kth smallest element using heap
//        //https://www.geeksforgeeks.org/kth-smallestlargest-element-unsorted-array/
//        kthSmallestElementInArrayStream(new int[]{7, 10, 4, 20, 3, 15}, 3);
//        kthSmallestElementInArrayStream(new int[]{7, 10, 4, 20, 3, 15, 6}, 3);
//        kthSmallestElementInArrayStream(new int[]{7, 10, 4, 20, 3, 15}, 4);
//        kthSmallestElementInArrayStream(new int[]{7, 10, 4, 20, 3, 15, 6}, 4);
//        System.out.println("Kth largest element in stream of array");
//        kthLargestElementInArrayStream(new int[]{7, 10, 4, 20, 3, 15}, 3);
//        kthLargestElementInArrayStream(new int[]{7, 10, 4, 20, 3, 15, 6}, 4);
//..............................................................................
//        System.out.println("Sort a nearly sorted (or K sorted) array");
//        //https://www.geeksforgeeks.org/nearly-sorted-algorithm/
//        kSortedArray(new int[]{6, 10, 5, 1, 4, 3}, 3);
//        kSortedArray(new int[]{6, 10, 5, 1, 4, 3, 1, 11, 5}, 3);
//        kSortedArray(new int[]{6, 5, 3, 2, 8, 10, 9}, 3);
//        kSortedArray(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 3);
//        //worst case all the element has to be pushed in heap as min is at the very last index
//        kSortedArray(new int[]{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, 3); 
//..............................................................................
//        System.out.println("merging two incr sorted linkedlist as one incr sorted linkedlist");
//        Node<Integer> headLL1 = new Node(2);
//        headLL1.setNext(new Node<Integer>(5));
//        headLL1.getNext().setNext(new Node<Integer>(8));
//
//        Node<Integer> headLL2 = new Node(1);
//        headLL2.setNext(new Node<Integer>(3));
//        headLL2.getNext().setNext(new Node<Integer>(6));
//        Node<Integer> mergedAsc = mergeSortedLinkedListAsc_Recursive(headLL1, headLL2);
//        LinkedListUtil<Integer> ll = new LinkedListUtil<>(mergedAsc);
//        ll.print();
//..............................................................................
//        System.out.println("Given two Linkedlists in sorted increasing order. Merge them in decreasing order. "
//                + "You have to merge in place, "
//                + "you can’t create new linkedlist.\n"
//                + "Approach : Simply apply merge-sort concept and append characters "
//                + "at front of merged list instead of end.");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde-1-feb-2020-exp-1-5-yr/?ref=leftbar-rightbar
//        //https://www.geeksforgeeks.org/merge-two-sorted-lists-place/
//        //https://www.geeksforgeeks.org/merge-k-sorted-linked-lists/
//        Node<Integer> headLL1 = new Node(2);
//        headLL1.setNext(new Node<Integer>(5));
//        headLL1.getNext().setNext(new Node<Integer>(8));
//
//        Node<Integer> headLL2 = new Node(1);
//        headLL2.setNext(new Node<Integer>(3));
//        headLL2.getNext().setNext(new Node<Integer>(6));
//        mergeSortedLinkedListDesc_Recursive(headLL1, headLL2);
//        LinkedListUtil<Integer> ll1 = new LinkedListUtil<>(headNew);
//        ll1.print();
//
//        //15 is not included here in this case //partially failed case
//        Node<Integer> headLL3 = new Node(5);
//        headLL3.setNext(new Node<Integer>(10));
//        headLL3.getNext().setNext(new Node<Integer>(15));
//
//        Node<Integer> headLL4 = new Node(2);
//        headLL4.setNext(new Node<Integer>(4));
//        headLL4.getNext().setNext(new Node<Integer>(6));
//
//        mergeSortedLinkedListDesc_Recursive(headLL3, headLL4);
//        LinkedListUtil<Integer> ll2 = new LinkedListUtil<>(headNew);
//        ll2.print();
//..............................................................................
//        System.out.println("Kth smallest/largest element in Tree ");
//        //https://www.interviewbit.com/problems/kth-smallest-element-in-tree/
//        
//        /*
//        Brute force approach
//        convert a tree into list or array by traversing takes O(N) or O(H)
//        where N = nodes in tree H = height of tree
//        then sort this list or array using best sorting algo quick/merge takes O(LogN)
//        so sortng N Node will take O(N*LogN) time
//        sort asc for k smallest
//        sort desc for k largest
//        total by this time O(N) + O(NLogN)
//        getting kth element from this array or list will be O(1)
//        
//        */
//        
//        /*
//        similar to kth smallest/largest eleemnt in array
//        heap based approach is O(N*LogK)
//        because we are at a time heapfying only K element in the heap
//        */
//        
//        TreeNode<Integer>  root = new TreeNode<>(2);
//        root.setLeft(new TreeNode(1));
//        root.setRight(new TreeNode(3));
//        
//        kthSmallestElementInTree(root, 2); //2
//        kthLargestElementInTree(root, 2); //2
//        
//        TreeNode<Integer>  root1 = new TreeNode<>(10);
//        root1.setLeft(new TreeNode(1));
//        root1.getLeft().setLeft(new TreeNode(5));
//        root1.setRight(new TreeNode(3));
//        root1.getRight().setRight(new TreeNode(2));
//        
//        kthSmallestElementInTree(root1, 2); //2
//        kthSmallestElementInTree(root, 3); //3
//        
//        kthLargestElementInTree(root1, 2); //5
//        kthLargestElementInTree(root1, 3); //3
//..............................................................................
//        System.out.println("Sorted Linked List to Balanced BST");
//        //https://www.geeksforgeeks.org/sorted-linked-list-to-balanced-bst/
//        Node<Integer> head = new Node<>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        linkedListToBST(head);
//        Node<Integer> head1 = new Node<>(5);
//        head1.setNext(new Node<>(10));
//        head1.getNext().setNext(new Node<>(15));
//        head1.getNext().getNext().setNext(new Node<>(20));
//        head1.getNext().getNext().getNext().setNext(new Node<>(25));
//        head1.getNext().getNext().getNext().getNext().setNext(new Node<>(30));
//        linkedListToBST(head1);
//..............................................................................  
//        System.out.println("Zigzag String");
//        //https://www.geeksforgeeks.org/print-concatenation-of-zig-zag-string-form-in-n-rows/
//        //https://www.interviewbit.com/problems/zigzag-string/
//        rowWiseStringZigZag("PAYPALISHIRING", 3);
//        rowWiseStringZigZag("ABCD", 2);
//..............................................................................
//        System.out.println("Given a large stream of strings, return the top 10 most frequently occurring string");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-258-for-sde1/?ref=rp
//        /*
//        (Hash map + min heap of size 10 is the solution.)
//        */
//        List<String> streams  = new ArrayList<>();
//        streams.add("hello");
//        streams.add("hello");
//        streams.add("hello");
//        streams.add("hello");
//        streams.add("hello");
//        streams.add("geeks");
//        streams.add("geeks");
//        streams.add("java");
//        streams.add("java");
//        streams.add("java");
//        streams.add("c#");
//        streams.add("php");
//        streams.add("python");
//        streams.add("python");
//        kMostFrequentlyOccuringWordFromStream(streams, 3);
//..............................................................................
//        System.out.println("Printing a tree Inorder, Preorder, Postorder using iteration approach");
//        BinaryTree<Integer> bt = new BinaryTree<>();
//        bt.insert(2);
//        bt.insert(1);
//        bt.insert(3);
//        bt.treeBFS();
//        System.out.println();
//        bt.treeInorderIterative();
//        System.out.println();
//        bt.treePreorderIterative();
//        System.out.println();
//        bt.treePostorderIterative();
//        System.out.println();
//..............................................................................
//        System.out.println("Sort linked list which is already sorted on absolute values");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-258-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/sort-linked-list-already-sorted-absolute-values/
//        //............O(N)
//        Node<Integer> head1 = new Node<>(1);
//        head1.setNext(new Node<>(-2));
//        head1.getNext().setNext(new Node<>(-3));
//        head1.getNext().getNext().setNext(new Node<>(4));
//        head1.getNext().getNext().getNext().setNext(new Node<>(-5));
//        sortLinkedListThatIsSortedIAsAbsoluteValue(head1);
//        
//        Node<Integer> head2 = new Node<>(-5);
//        head2.setNext(new Node<>(-4));
//        head2.getNext().setNext(new Node<>(-3));
//        head2.getNext().getNext().setNext(new Node<>(-2));
//        head2.getNext().getNext().getNext().setNext(new Node<>(-1));
//        sortLinkedListThatIsSortedIAsAbsoluteValue(head2);
//..............................................................................
//        System.out.println("Sort linked list using merge sort");
//        //https://leetcode.com/problems/sort-list/discuss/46714/Java-merge-sort-solution  
//        //.........................O(NLogN) time............................... 
//        Node<Integer> head1 = new Node<>(1);
//        head1.setNext(new Node<>(-2));
//        head1.getNext().setNext(new Node<>(-3));
//        head1.getNext().getNext().setNext(new Node<>(4));
//        head1.getNext().getNext().getNext().setNext(new Node<>(-5));
//        LinkedListUtil<Integer> ll = new LinkedListUtil<>(sortLinkedListUsingMergeSort(head1));
//        ll.print();
//..............................................................................
//        System.out.println("Check whether two strings are anagram of each other");
//        validAnagrams_1("listen", "silent");
//        validAnagrams_2("listen", "silent");
//..............................................................................
//        System.out.println("49. Group Anagrams");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde1/?ref=rp
//        //https://leetcode.com/problems/group-anagrams/
//        /*
//        
//         Given an array of strings, group anagrams together.
//
//         Example:
//
//         Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
//         Output:
//         [
//         ["ate","eat","tea"],
//         ["nat","tan"],
//         ["bat"]
//         ]
//         Note:
//
//         All inputs will be in lowercase.
//         The order of your output does not matter.
//        
//         */
//
//        String[] strs = {"eat", "tea", "tan", "ate", "nat", "bat"};
//        groupAnagrams(strs);
//..............................................................................
//        System.out.println("438. Find All Anagrams in a String");
//        //https://leetcode.com/problems/find-all-anagrams-in-a-string/
//        /*
//        
//         Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.
//
//         Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.
//
//         The order of output does not matter.
//
//         Example 1:
//
//         Input:
//         s: "cbaebabacd" p: "abc"
//
//         Output:
//         [0, 6]
//
//         Explanation:
//         The substring with start index = 0 is "cba", which is an anagram of "abc".
//         The substring with start index = 6 is "bac", which is an anagram of "abc".
//        
//         */
//        
//        //...............O(M*NLogN).....
//        //M char in string s
//        //N char in string p for sorting using Arrays.sort() it takes O(LogN)
//        findAnagrams("cbaebabacd", "abc");
//        findAnagrams("abab", "ab");
//        findAnagrams("abab", "abab");
//..............................................................................
//        System.out.println("Reverse a linked list in place space O(1) time O(N)");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-258-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/reverse-a-linked-list/
//        Node<Integer> head1 = new Node<>(1);
//        head1.setNext(new Node<>(2));
//        head1.getNext().setNext(new Node<>(3));
//        head1.getNext().getNext().setNext(new Node<>(4));
//        head1.getNext().getNext().getNext().setNext(new Node<>(5));
//        inPlaceReverseLinkedList(head1);
//..............................................................................
//        System.out.println("92. Reverse Linked List II");
//        //https://leetcode.com/problems/reverse-linked-list-ii/
//        Node<Integer> head1 = new Node<>(1);
//        head1.setNext(new Node<>(2));
//        head1.getNext().setNext(new Node<>(3));
//        head1.getNext().getNext().setNext(new Node<>(4));
//        head1.getNext().getNext().getNext().setNext(new Node<>(5));
//        reverseLinkedList_2(head1, 2, 4); //reverse the list between the ranges
//..............................................................................
//        System.out.println("143. Reorder List");
//        //https://leetcode.com/problems/reorder-list/
//        //Given a singly linked list L: L0→L1→…→Ln-1→Ln,
//        //reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…
//        //Given 1->2->3->4, reorder it to 1->4->2->3.
//        //Given 1->2->3->4->5, reorder it to 1->5->2->4->3.
//        Node<Integer> head1 = new Node<>(1);
//        head1.setNext(new Node<>(2));
//        head1.getNext().setNext(new Node<>(3));
//        head1.getNext().getNext().setNext(new Node<>(4));
//        head1.getNext().getNext().getNext().setNext(new Node<>(5));
//        //........O(N^2)....
//        //inplace shifting
////        reorderList_1(head1);
//        //......time O(N) space O(N)...
//        //stack used
//        reorderList_2(head1);
//..............................................................................
//        System.out.println("61. Rotate List");
//        //https://leetcode.com/problems/rotate-list/
//        //https://leetcode.com/problems/rotate-list/discuss/22715/Share-my-java-solution-with-explanation
//        Node<Integer> head1 = new Node<>(1);
//        head1.setNext(new Node<>(2));
//        head1.getNext().setNext(new Node<>(3));
//        int k = 200000000;
//        LinkedListUtil<Integer> ll = new LinkedListUtil(rotateListUptoK(head1, k));
//        ll.print();
//        Node<Integer> head2 = new Node<>(1);
//        head2.setNext(new Node<>(2));
//        head2.getNext().setNext(new Node<>(3));
//        LinkedListUtil<Integer> ll2 = new LinkedListUtil(rotateListUptoK_Efficient(head2, k));
//        ll2.print();
//..............................................................................
//        System.out.println("987. Vertical Order Traversal of a Binary Tree");
//        //https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/
//        /*
//        
//         Given a binary tree, return the vertical order traversal of its nodes values.
//         For each node at position (X, Y), its left and right children respectively will be 
//         at positions (X-1, Y-1) and (X+1, Y-1).
//         Running a vertical line from X = -infinity to X = +infinity, whenever the vertical line touches some nodes, 
//         we report the values of the nodes in order from top to bottom (decreasing Y coordinates).
//         If two nodes have the same position, then the value of the node that is reported first is the value 
//         that is smaller.
//         Return an list of non-empty reports in order of X coordinate.  Every report will have a list of values of nodes.
//        
//         */
//        TreeNode<Integer>  root = new TreeNode<>(1);
//        root.setLeft(new TreeNode(2));
//        root.getLeft().setLeft(new TreeNode(4));
//        root.getLeft().setRight(new TreeNode(5));
//        root.setRight(new TreeNode(3));
//        root.getRight().setLeft(new TreeNode(6));
//        root.getRight().setRight(new TreeNode(7));
//        
//        BinaryTree<Integer> bt = new BinaryTree<>(root);
//        bt.treeBFS();
//        System.out.println();
//        //logic used is similar to Binary Tree topView print
//        verticalTraversal(root);
//..............................................................................
//        System.out.println("Transform to Sum Tree");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/transform-to-sum-tree/1
//        TreeNode<Integer>  root = new TreeNode<>(10);
//        root.setLeft(new TreeNode(-2));
//        root.getLeft().setLeft(new TreeNode(8));
//        root.getLeft().setRight(new TreeNode(-4));
//        root.setRight(new TreeNode(6));
//        root.getRight().setLeft(new TreeNode(7));
//        root.getRight().setRight(new TreeNode(5));
//        BinaryTree<Integer> bt = new BinaryTree<>(root);
//        bt.treeBFS();
//        System.out.println();
//        transformToSumTree(root);
//..............................................................................
//        System.out.println("Reverse a Linked List in groups of given size");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/reverse-a-linked-list-in-groups-of-given-size/1
//        //https://www.geeksforgeeks.org/reverse-a-list-in-groups-of-given-size/
//        Node<Integer> head1 = new Node<>(1);
//        head1.setNext(new Node<>(2));
//        head1.getNext().setNext(new Node<>(3));
//        head1.getNext().getNext().setNext(new Node<>(4));
//        head1.getNext().getNext().getNext().setNext(new Node<>(5));
//        LinkedListUtil<Integer> ll = new LinkedListUtil<>(reverseLinkedListInKGroups_1(head1, 2)); //EXACT
//        ll.print();
//        head1 = new Node<>(3);
//        head1.setNext(new Node<>(8));
//        head1.getNext().setNext(new Node<>(7));
//        head1.getNext().getNext().setNext(new Node<>(2));
//        head1.getNext().getNext().getNext().setNext(new Node<>(5));
//        head1.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        ll = new LinkedListUtil<>(reverseLinkedListInKGroups_1(head1, 4)); //EXACT
//        ll.print();
//        //different approach different question
//        //https://leetcode.com/problems/reverse-nodes-in-k-group/
//        head1 = new Node<>(1);
//        head1.setNext(new Node<>(2));
//        head1.getNext().setNext(new Node<>(3));
//        head1.getNext().getNext().setNext(new Node<>(4));
//        head1.getNext().getNext().getNext().setNext(new Node<>(5));
//        ll = new LinkedListUtil<>(reverseLinkedListInKGroups_2(head1, 2));
//        ll.print();
//        head1 = new Node<>(3);
//        head1.setNext(new Node<>(8));
//        head1.getNext().setNext(new Node<>(7));
//        head1.getNext().getNext().setNext(new Node<>(2));
//        head1.getNext().getNext().getNext().setNext(new Node<>(5));
//        head1.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        ll = new LinkedListUtil<>(reverseLinkedListInKGroups_2(head1, 4)); //EXACT
//        ll.print();
//..............................................................................
//        System.out.println("Maximum sum such that no two elements are adjacent");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/stickler-theif/0
//        int[] houses = {5,5,100,10,5};
//        int n = houses.length;
//        System.out.println(stiklerThief(houses, n));
//..............................................................................
//        System.out.println("K-th distinct element");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/k-th-distinct-element/0
//        int[] a = {1, 2, 1, 3, 4, 2};
//        int k = 2;
//        kThDistinctElementInArray(a, k);
//        int[] b = {1, 2, 50, 10, 20, 2};
//        k = 3;
//        kThDistinctElementInArray(b, k);
//        int[] c = {2, 2, 2, 2};
//        k = 2;
//        kThDistinctElementInArray(c, k);
//..............................................................................
//        System.out.println("Convert a given Binary Tree to Doubly Linked List");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/binary-tree-to-dll/1
//        //https://www.geeksforgeeks.org/convert-given-binary-tree-doubly-linked-list-set-3/
//        TreeNode<Integer> root = new TreeNode<>(10);
//        root.setLeft(new TreeNode(12));
//        root.getLeft().setLeft(new TreeNode(25));
//        root.getLeft().setRight(new TreeNode(30));
//        root.setRight(new TreeNode(15));
//        root.getRight().setLeft(new TreeNode(36));
//        TreeNode r = convertBinaryTreeToDoublyLL(root, null);
//        printBinaryTreeAsDoublyLL(r);
//
//        TreeNode<Integer> root1 = new TreeNode<>(10);
//        root1.setLeft(new TreeNode(12));
//        root1.getLeft().setLeft(new TreeNode(25));
//        root1.getLeft().setRight(new TreeNode(30));
//        root1.setRight(new TreeNode(15));
//        root1.getRight().setLeft(new TreeNode(36));
//        //optimized and stright forward approach
//        convertBinaryTreeToDoublyLL(root1);
//        printBinaryTreeAsDoublyLL(headOfDLL);
//..............................................................................
//        System.out.println("124. Binary Tree Maximum Path Sum");
//        //https://leetcode.com/problems/binary-tree-maximum-path-sum/
//        //https://leetcode.com/problems/binary-tree-maximum-path-sum/discuss/39775/Accepted-short-solution-in-Java
//        TreeNode<Integer> root1 = new TreeNode<>(-10);
//        root1.setLeft(new TreeNode(9));
//        root1.setRight(new TreeNode(20));
//        root1.getRight().setLeft(new TreeNode(15));
//        root1.getRight().setRight(new TreeNode(7));
//        maxPathSumOfBinaryTree(root1);
//        System.out.println(maxPathSum);
//..............................................................................
//        System.out.println("814. Binary Tree Pruning");
//        //https://leetcode.com/problems/binary-tree-pruning/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setRight(new TreeNode(0));
//        root1.getRight().setLeft(new TreeNode(0));
//        root1.getRight().setRight(new TreeNode(1));
//        BinaryTree<Integer> bt = new BinaryTree<>(treePrune(root1));
//        bt.treeBFS();
//..............................................................................
//        System.out.println("235. Lowest Common Ancestor of a Binary Tree");
//        //https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
//        //https://www.geeksforgeeks.org/lowest-common-ancestor-binary-tree-set-1/
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        TreeNode ans = lowestCommonAncestor(root1, new TreeNode<>(2), new TreeNode<>(8));
//        System.out.println("Lowest common ancesctor for 2 and 8 is "+ans.getData());
//        ans = lowestCommonAncestor(root1, new TreeNode<>(2), new TreeNode<>(4));
//        System.out.println("Lowest common ancesctor for 2 and 4 is "+ans.getData());
//        ans = lowestCommonAncestor(root1, new TreeNode<>(3), new TreeNode<>(7));
//        System.out.println("Lowest common ancesctor for 3 and 7 is "+ans.getData());
//..............................................................................
//        System.out.println("1123. Lowest Common Ancestor of Deepest Leaves");
//        //https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/
//        //https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/discuss/804531/Recursive-Solution-oror-0ms-beats-100-oror-Java
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        TreeNode ans = lcaDeepestLeaves(root1);
//        //deepest leaves are 3 and 5 
//        // 4 is lca to 3, 5 so printing the subtree of this lca 
//        System.out.println("Lowest common ancesctor for the deepest leaf node "+ans.getData());
//        BinaryTree<Integer> bt = new BinaryTree<>(ans);
//        bt.treeBFS();
//..............................................................................
//        System.out.println("lowest common ancestor in Binary search tree");
//        //https://www.geeksforgeeks.org/lowest-common-ancestor-in-a-binary-search-tree/
//        TreeNode<Integer> root = new TreeNode(20); 
//        root.setLeft(new TreeNode<>(8)); 
//        root.setRight(new TreeNode<>(22)); 
//        root.getLeft().setLeft(new TreeNode<>(4)); 
//        root.getLeft().setRight(new TreeNode<>(12)); 
//        root.getLeft().getRight().setLeft(new TreeNode<>(10)); 
//        root.getLeft().getRight().setRight(new TreeNode<>(14));
//        //this works for binary tree and bst both......
//        TreeNode ans = lowestCommonAncestor(root, new TreeNode<>(8), new TreeNode<>(14));
//        System.out.println("Lowest common ancesctor for 8 and 14 is "+ans.getData()); 
//        //............
//        System.out.println("LCA in BST for 8, 14 using recursion "+lowestCommonAncestorInBinarySearchTree_Recursion(root, 8, 14).getData());
//        System.out.println("LCA in BST for 8, 22  using recursion "+lowestCommonAncestorInBinarySearchTree_Recursion(root, 8, 22).getData());
//        System.out.println("LCA in BST for 8, 14  using iteration "+lowestCommonAncestorInBinarySearchTree_Iterative(root, 8, 14).getData());
//        System.out.println("LCA in BST for 8, 22  using iteration "+lowestCommonAncestorInBinarySearchTree_Iterative(root, 8, 22).getData());
//        ans = lowestCommonAncestor(root, new TreeNode<>(8), new TreeNode<>(22));
//        System.out.println("Lowest common ancesctor for 8 and 22 is "+ans.getData()); //this works for binary tree
//        
//        //non bst tree
//        TreeNode<Integer> root1 = new TreeNode(2); 
//        root1.setLeft(new TreeNode<>(1)); 
//        root1.setRight(new TreeNode<>(1)); 
//        //this works for binary tree and bst both......
//        ans = lowestCommonAncestor(root1, new TreeNode<>(1), new TreeNode<>(1));
//        System.out.println("Lowest common ancesctor for 1 and 1 is "+ans.getData());
//        //.......................
//        //this below method call fails for non bst tree.............
//        System.out.println("LCA in non-BST for 1, 1  using recursion "+lowestCommonAncestorInBinarySearchTree_Recursion(root1, 1, 1).getData());
//        System.out.println("LCA in non-BST for 1, 1  using iteration "+lowestCommonAncestorInBinarySearchTree_Iterative(root1, 1, 1).getData());
//..............................................................................
//        System.out.println("How to determine if a binary tree is height-balanced?");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-186-for-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/check-for-balanced-tree/1
//        //https://www.geeksforgeeks.org/how-to-determine-if-a-binary-tree-is-balanced/
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode(2));
//        root.setRight(new TreeNode(3));
//        System.out.println("height is balanced in binary tree? "+isBinaryTreeHeightBalanced(root, new Height()));
//        
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        System.out.println("height is balanced in binary tree? "+isBinaryTreeHeightBalanced(root1, new Height()));
//        
//        //Skwed binary tree
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode(2));
//        root2.getLeft().setLeft(new TreeNode(3));
//        System.out.println("height is balanced in binary tree? "+isBinaryTreeHeightBalanced(root2, new Height()));
//..............................................................................
//        System.out.println("Diameter of tree | DP on tree");
//        //https://www.geeksforgeeks.org/diameter-of-a-binary-tree/
//        /*
//        Def: 
//        The diameter of a tree (sometimes called the width) is the
//        number of nodes on the longest path between two leaf nodes. 
//        */
//        //below root1 is diagramatic example 1 in the gfg link
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().setRight(new TreeNode<>(5));
//        root1.getLeft().getRight().setLeft(new TreeNode<>(6));
//        root1.getLeft().getRight().setRight(new TreeNode<>(7));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setRight(new TreeNode<>(8));
//        root1.getRight().getRight().setRight(new TreeNode<>(9));
//        root1.getRight().getRight().getRight().setLeft(new TreeNode<>(10));
//        root1.getRight().getRight().getRight().getLeft().setRight(new TreeNode<>(12));
//        root1.getRight().getRight().getRight().setRight(new TreeNode<>(11));
//        
//        BinaryTree<Integer> bt = new BinaryTree<>(root1);
//        bt.treeBFS();
//        System.out.println("Diameter of above tree b/w node 7(leaf) to 12(leaf)"+diameterOfTree(root1, new Height()));
//        
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(2));
//        root2.getLeft().setLeft(new TreeNode<>(4));
//        root2.getLeft().setRight(new TreeNode<>(5));
//        root2.getLeft().getRight().setRight(new TreeNode<>(6));
//        root2.getLeft().getRight().getRight().setRight(new TreeNode<>(7));
//        root2.getLeft().getRight().getRight().getRight().setLeft(new TreeNode<>(8));
//        bt = new BinaryTree<>(root2);
//        bt.treeBFS();
//        System.out.println("Diameter of above tree b/w node 4(leaf) to 8(leaf)"+diameterOfTree(root2, new Height()));
//..............................................................................
//        System.out.println("Max path sum of tree any node to any node | DP on tree");
//        //https://www.geeksforgeeks.org/find-maximum-path-sum-in-a-binary-tree/
//        TreeNode<Integer> root = new TreeNode<>(10);
//        root.setLeft(new TreeNode(2));
//        root.getLeft().setLeft(new TreeNode<>(20));
//        root.getLeft().setRight(new TreeNode<>(1));
//        root.setRight(new TreeNode(10));
//        root.getRight().setRight(new TreeNode(-25));
//        root.getRight().getRight().setLeft(new TreeNode<>(3));
//        root.getRight().getRight().setRight(new TreeNode<>(4));
//        Result res = new Result();
//        maxPathSumOfTreeAnyNodeToAnyNode(root, res);
//        System.out.println("max path sum of tree "+res.result);
//        
//        root = new TreeNode<>(-10);
//        root.setLeft(new TreeNode(9));
//        root.setRight(new TreeNode(20));
//        root.getRight().setLeft(new TreeNode<>(15));
//        root.getRight().setRight(new TreeNode<>(7));
//        res = new Result();
//        maxPathSumOfTreeAnyNodeToAnyNode(root, res);
//        System.out.println("max path sum of tree "+res.result);
//        
//        root = new TreeNode<>(-10);
//        res = new Result();
//        maxPathSumOfTreeAnyNodeToAnyNode(root, res);
//        System.out.println("max path sum of tree "+res.result);
//..............................................................................
//        System.out.println("Max path sum of tree leaf node to leaf node| DP on tree");
//        //https://www.geeksforgeeks.org/find-maximum-path-sum-two-leaves-binary-tree/
//        TreeNode<Integer> root = new TreeNode<>(-15);
//        root.setLeft(new TreeNode<>(5));
//        root.getLeft().setLeft(new TreeNode<>(-8));
//        root.getLeft().getLeft().setLeft(new TreeNode<>(2));
//        root.getLeft().getLeft().setRight(new TreeNode<>(6));
//        root.getLeft().setRight(new TreeNode<>(1));
//        root.setRight(new TreeNode<>(6));
//        root.getRight().setLeft(new TreeNode<>(3));
//        root.getRight().setRight(new TreeNode<>(9));
//        root.getRight().getRight().setRight(new TreeNode<>(0));
//        root.getRight().getRight().getRight().setLeft(new TreeNode<>(4));
//        root.getRight().getRight().getRight().setRight(new TreeNode<>(-1));
//        root.getRight().getRight().getRight().getRight().setLeft(new TreeNode<>(10));
//
//        Result res = new Result();
//        maxPathSumOfTreeLeafNodeToLeafNode(root, res);
//        System.out.println("max path sum of tree " + res.result);
//
//        TreeNode<Integer> root1 = new TreeNode<>(10);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode<>(20));
//        root1.getLeft().setRight(new TreeNode<>(1));
//        root1.setRight(new TreeNode(10));
//        root1.getRight().setRight(new TreeNode(-25));
//        root1.getRight().getRight().setLeft(new TreeNode<>(3));
//        root1.getRight().getRight().setRight(new TreeNode<>(4));
//        res = new Result();
//        maxPathSumOfTreeLeafNodeToLeafNode(root1, res);
//        System.out.println("max path sum of tree " + res.result);
//        
//        root1 = new TreeNode<>(-10);
//        res = new Result();
//        maxPathSumOfTreeLeafNodeToLeafNode(root1, res);
//        System.out.println("max path sum of tree " + res.result);
//        
//        root1 = new TreeNode<>(-1);
//        root1.setLeft(new TreeNode<>(1000));
//        root1.setRight(new TreeNode<>(3));
//        res = new Result();
//        maxPathSumOfTreeLeafNodeToLeafNode(root1, res);
//        System.out.println("max path sum of tree " + res.result);
//..............................................................................
//        System.out.println(" Find if the given tree is the subtree of the big tree.");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-186-for-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/check-if-subtree/1
//        //https://www.geeksforgeeks.org/check-if-a-binary-tree-is-subtree-of-another-binary-tree/
//        //https://www.geeksforgeeks.org/check-binary-tree-subtree-another-binary-tree-set-2/
//        TreeNode<Integer> sub = new TreeNode<>(10);
//        sub.setLeft(new TreeNode(4));
//        sub.setRight(new TreeNode(6));
//        sub.getRight().setLeft(new TreeNode(30));
//        
//        TreeNode<Integer> main = new TreeNode<>(26);
//        main.setLeft(new TreeNode(10));
//        main.getLeft().setLeft(new TreeNode(4));
//        main.getLeft().setRight(new TreeNode(6));
//        main.getLeft().getRight().setLeft(new TreeNode(30));
//        main.setRight(new TreeNode(3));
//        main.getRight().setRight(new TreeNode(3));
//        
//        System.out.println("is Subtree "+isSubTree(main, sub));
//        
//        TreeNode<Integer> sub1 = new TreeNode<>(10);
//        sub1.setLeft(new TreeNode(20));
//        
//        TreeNode<Integer> main1 = new TreeNode<>(10);
//        main1.setLeft(new TreeNode(20));
//        main1.getLeft().setLeft(new TreeNode(30));
//        main1.getLeft().setRight(new TreeNode(40));
//        main1.setRight(new TreeNode(50));
//        //main - 10-20-30 sub - 10-20
//        //20 should not have any further child in main  
//        System.out.println("is Subtree "+isSubTree(main1, sub1));        
//..............................................................................
//        System.out.println("101. Symmetric Tree");
//        //https://leetcode.com/problems/symmetric-tree/
//        //https://leetcode.com/problems/symmetric-tree/solution/
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode(2));
//        root.getLeft().setLeft(new TreeNode(3));
//        root.getLeft().setRight(new TreeNode(4));
//        root.setRight(new TreeNode(2));
//        root.getRight().setLeft(new TreeNode(4));
//        root.getRight().setRight(new TreeNode(3));
//        
//        System.out.println("is tree symmetric? recursive "+isTreeSymmetric_Recursion(root));
//        System.out.println("is tree symmetric? iterative "+isTreeSymmetric_Iteraative(root));
//        
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(null);
//        root1.getLeft().setRight(new TreeNode(3));
//        root1.setRight(new TreeNode(2));
//        root1.getRight().setLeft(null);
//        root1.getRight().setRight(new TreeNode(3));
//        
//        System.out.println("is tree symmetric? recursive "+isTreeSymmetric_Recursion(root1));
//        System.out.println("is tree symmetric? iterative "+isTreeSymmetric_Iteraative(root1));       
//..............................................................................
//        System.out.println("Is Tree binary search tree");
//        TreeNode<Integer> root1 = new TreeNode<>(10);
//        root1.setLeft(new TreeNode(5));
//        root1.setRight(new TreeNode<>(15));
//        System.out.println("is tree bst? "+isTreeBST(root1));
//        
//        root1 = new TreeNode<>(-10);
//        root1.setLeft(new TreeNode(5));
//        root1.setRight(new TreeNode<>(15));
//        System.out.println("is tree bst? "+isTreeBST(root1));
//..............................................................................
//        System.out.println("207. Course Schedule | Figure out strategy to find the order of execution of processes");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-185-for-sde1/?ref=rp
//        //https://leetcode.com/problems/course-schedule/
//        int[][] coursesPre = {{1, 0}};
//        System.out.println("can courses be completed? "+canFinish(2, coursesPre));
//        int[][] coursesPre2 = {{1, 0}, {0, 1}};
//        System.out.println("can courses be completed? "+canFinish(2, coursesPre2));
//..............................................................................
//        System.out.println("Remove middle element from the stack");
//        //you can not use any other datastructure to do this
//        Stack<Integer> s = new Stack<>();
//        s.push(1);
//        s.push(2);
//        s.push(3);
//        s.push(4);
//        s.push(5);
//        removeMiddleElementFromStack(s, s.size(), 1);
//        System.out.println(s.toString());
//..............................................................................
//        System.out.println("Reverse the stack using recurion");
//        stackReverse.push(1);
//        stackReverse.push(2);
//        stackReverse.push(3);
//        stackReverse.push(4);
//        stackReverse.push(5);
//        System.out.println(stackReverse.toString());
//        reverseStackRecursion();
//        System.out.println(stackReverse.toString());
//..............................................................................
//        System.out.println("The minimum cost from hackerearth");
//
//        /*
//        
//         The minimum cost
//         You are given an integer X. In one operation, you are allowed to add an even divisor d (d != X) of X, therefore, X
//         becomes X + d.
//         For example: if X = 8 and you add d = 4 to X, then X becomes 12. Now, you can select even divisor of 12 and
//         add it to 12 (d = 6), then X becomes 18 and so on.
//         The cost of one operation (X > X+d)is X/d.
//         You are given two integers N and M. You are required to find the minimum steps to change N to M by
//         performing the provided operation optimally. If it is impossible to convert N into M using the provided operation,
//         then print —1.
//         Input format
//         ¢ The first line contains an integer T denoting the number of test cases. Description of each test case as follows.
//         ¢ The first line of each test case contains two integers N and M.
//         Output format
//         For each test case, print an integer denoting the required answer.
//         */
//        int N = 6, M = 24;
//        System.out.println(theMinimumCost_HackerEarth(N, M, 0));
//        N = 8;
//        M = 12;
//        System.out.println(theMinimumCost_HackerEarth(N, M, 0));
//        N = 9;
//        M = 17;
//        System.out.println(theMinimumCost_HackerEarth(N, M, 0));
//..............................................................................
//        System.out.println("Inplace rotate square matrix by 90 degrees");
//        //https://www.geeksforgeeks.org/inplace-rotate-square-matrix-by-90-degrees/
//        int mat[][] = {
//            {1, 2, 3, 4},
//            {5, 6, 7, 8},
//            {9, 10, 11, 12},
//            {13, 14, 15, 16}
//        };
//        inPlace90AntiClockRotateMatrix(mat);
//..............................................................................
//        System.out.println("779. K-th Symbol in Grammar");
//        //https://leetcode.com/problems/k-th-symbol-in-grammar/
//        /*
//         Explanation:
//         row 1: 0
//         row 2: 01
//         row 3: 0110
//         row 4: 01101001
//        
//         */
//        int N = 1, K = 1;
//        System.out.println("1st approch "+kthGrammar(N, K));
//        System.out.println("2nd approach"+kthGrammar_2(N, K));
//        N = 2;
//        K = 2;
//        System.out.println("1st approch "+kthGrammar(N, K));
//        System.out.println("2nd approach"+kthGrammar_2(N, K));
//        N = 4;
//        K = 6;
//        System.out.println("1st approch "+kthGrammar(N, K));
//        System.out.println("2nd approach"+kthGrammar_2(N, K));
//..............................................................................
//        System.out.println("Connect n ropes with minimum cost");
//        //https://www.geeksforgeeks.org/connect-n-ropes-minimum-cost/
//        int ropesLength[] = { 4, 3, 2, 6 };
//        connectRopesWithMinCost(ropesLength);
//..............................................................................
//        System.out.println("12. Integer to Roman");
//        //https://leetcode.com/problems/integer-to-roman/
//        System.out.println("roman : "+intToRoman(3));
//        System.out.println("roman : "+intToRoman(432));
//        System.out.println("roman : "+intToRoman(501));
//        System.out.println("roman : "+intToRoman(3999));
//..............................................................................
//        System.out.println("13. Roman to Integer");
//        //https://leetcode.com/problems/roman-to-integer/
//        System.out.println("int : " + romanToInt("MMMCMXCIX"));
//        System.out.println("int : " + romanToInt("VIII"));
//        System.out.println("int : " + romanToInt("DI"));
//        System.out.println("int : " + romanToInt("MCMXCIV"));
//..............................................................................
//        System.out.println("Sliding window array with fixed given window size K bsic algo");
//        slidingWindowArrayWithFixedWindowK_BasicAlgo(new int[]{2, 5, 1, 8, 4, 3}, 3);
//        slidingWindowArrayWithFixedWindowK_BasicAlgo(new int[]{2, 5, 1, 8, 4, 3, 6}, 3);
//        slidingWindowArrayWithFixedWindowK_BasicAlgo(new int[]{2, 5, 1}, 3);
//        slidingWindowArrayWithFixedWindowK_BasicAlgo(new int[]{2}, 3);
//..............................................................................
//        System.out.println("Sliding window string with fixed given window size K bsic algo");
//        slidingWindowStringWithFixedWindowK_BasicAlgo("sangeet", 3);
//        slidingWindowStringWithFixedWindowK_BasicAlgo("abcdefghi", 3);
//        slidingWindowStringWithFixedWindowK_BasicAlgo("aabcbcdbca", 4);
//..............................................................................
//        System.out.println("Max sum of sub array with given fixed size window K");
//        //Aditya verma youtube playlist
//        slidingWindowMaxSumSubArrayOfWindowSizeK(new int[]{1, 2, 3, 4, 5, 6, 7}, 3); //should be 5+6+7
//        slidingWindowMaxSumSubArrayOfWindowSizeK(new int[]{2, 5, 1, 8, 4, 3}, 2); //should be 8+4
//..............................................................................
//        System.out.println("Smallest window that contains all characters of string itself");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-213-off-campus-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/smallest-window-contains-characters-string/
//        String str = "aabcbcdbca";
//        System.out.println("Smallest window containing all distinct characters is: " + smallestWindowContainingAllCharOfString(str));
//..............................................................................
//        System.out.println("239. Sliding Window Maximum");
//        //https://leetcode.com/problems/sliding-window-maximum/
//        //https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-size-k-using-stack-in-on-time/
//        slidingWindowMaximum_ON2(new int[]{1,3,-1,-3,5,3,6,7}, 3);
//        slidingWindowMaximum_ON(new int[]{1,3,-1,-3,5,3,6,7}, 3);
//..............................................................................
//        System.out.println("Root to leaf path sum ");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-off-campus-for-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/root-to-leaf-path-sum/1
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.setRight(new TreeNode<>(3));
//        allTreePathRootToLeafWithWithGivenSum(root, 2);
//
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(3));
//        allTreePathRootToLeafWithWithGivenSum(root1, 4);
//        
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.setRight(new TreeNode<>(3));
//        allTreePathRootToLeafWithWithGivenSum(root2, 4);
//..............................................................................
//        System.out.println("Nodes at given distance in binary tree");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-off-campus-for-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/nodes-at-given-distance-in-binary-tree/1
//        //https://www.geeksforgeeks.org/print-nodes-distance-k-given-node-binary-tree/
//        TreeNode<Integer> root2 = new TreeNode<>(20);
//        root2.setLeft(new TreeNode<>(8));
//        root2.getLeft().setLeft(new TreeNode<>(4));
//        root2.getLeft().setRight(new TreeNode<>(12));
//        root2.getLeft().getRight().setLeft(new TreeNode<>(10));
//        root2.getLeft().getRight().setRight(new TreeNode<>(14));
//        root2.setRight(new TreeNode<>(22));
//        binaryTreeNodeAtKDistanceFromTarget(root2, 8, 2);
//        binaryTreeNodeAtKDistanceFromTarget(root2, 8, 1);
//        binaryTreeNodeAtKDistanceFromTarget(root2, 13, 2);
//..............................................................................
//        System.out.println("Search in a row wise and column wise sorted matrix");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-off-campus-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/search-in-row-wise-and-column-wise-sorted-matrix/
//        int mat[][] = {{10, 20, 30, 40},
//        {15, 25, 35, 45},
//        {27, 29, 37, 48},
//        {32, 33, 39, 50}};
//
//        findXInSortedMatrix(mat, 29);
//        findXInSortedMatrix(mat, 100);
//..............................................................................
//        System.out.println("114. Flatten Binary Tree to Linked List");
//        //https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
//        //https://leetcode.com/problems/flatten-binary-tree-to-linked-list/discuss/825970/Standard-Java-Solution
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(2));
//        root2.getLeft().setLeft(new TreeNode<>(3));
//        root2.getLeft().setRight(new TreeNode<>(4));
//        root2.setRight(new TreeNode<>(5));
//        root2.getRight().setRight(new TreeNode<>(6));
//        binaryTreeFlattening(root2);
//..............................................................................
//        System.out.println("1328. Break a Palindrome");
//        //https://leetcode.com/problems/break-a-palindrome/
//        System.out.println(breakPalindrome("aba"));
//        System.out.println(breakPalindrome("abccba"));
//        System.out.println(breakPalindrome("aaa"));
//        System.out.println(breakPalindrome("bbb"));
//        System.out.println(breakPalindrome("b"));
//..............................................................................
//        System.out.println("648. Replace Words");
//        //https://leetcode.com/problems/replace-words/
//        System.out.println(replaceWords(Arrays.asList("cat","bat","rat"), "the cattle was rattled by the battery"));
//..............................................................................
//        System.out.println(" Largest length of subarray with given sum | handles positive integers");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-291-on-campus-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/find-subarray-with-given-sum/
//        int arr[] = { 15, 2, 4, 8, 9, 5, 10, 23 }; 
//        int sum = 23; 
//        continousSubArrayWhoseSumIsK(arr, sum);
//        
//        int ar[] = { 1, 1, 1, 1, 1 }; 
//        sum = 5; 
//        continousSubArrayWhoseSumIsK(ar, sum);
//        continousSubArrayWhoseSumIsK(ar, 32);
//..............................................................................
//        System.out.println(" Largest length of subarray with given sum | handles negative integers");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-291-on-campus-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/find-subarray-with-given-sum-in-array-of-integers/
//        int[] arr = {10, 2, -2, -20, 10}; 
//        int n = arr.length; 
//        int sum = -10; 
//        continousSubArrayWhoseSumIsK_HandlesNegtiveNum(arr, sum); 
//..............................................................................
//        System.out.println("Queue Implementation using 2 stacks");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-291-on-campus-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/queue-using-stacks/
//        QueueImplmentationUsingTwoStacks sq = new QueueImplmentationUsingTwoStacks();
//        sq.enQueue(1);
//        sq.enQueue(2);
//        sq.enQueue(3);
//        sq.print();
//        System.out.println("deQueued "+sq.deQueue());
//        sq.enQueue(4);
//        sq.enQueue(5);
//        sq.print();
//        System.out.println("deQueued "+sq.deQueue());
//        System.out.println("peek "+sq.peek());
//        sq.print();
//..............................................................................
//        System.out.println("Given an array with all "
//                + "even elements present even number of "
//                + "times except one which is present odd number "
//                + "of times. Find that element.");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-291-on-campus-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/find-the-number-occurring-odd-number-of-times/
//        int arr[] = {1, 2, 3, 2, 3, 1, 3}; 
//        findOddOccurringNumber_UsingHashing(arr);
//        findOddOccurringNumber_UsingXOR(arr);
//        
//        int ar[] = {1, 2, 3, 2, 3, 1}; 
//        findOddOccurringNumber_UsingHashing(ar);
//        findOddOccurringNumber_UsingXOR(ar);
//        
//        int a[] = {1, 2, 3}; 
//        findOddOccurringNumber_UsingHashing(a);
//        findOddOccurringNumber_UsingXOR(a);
//..............................................................................
//        System.out.println("Find if there is a rectangle in binary matrix with corners as 1");
//        //https://www.interviewbit.com/problems/find-rectangle-in-binary-matrix/
//        //https://www.geeksforgeeks.org/find-rectangle-binary-matrix-corners-1/
//        int mat[][] = {{1, 0, 0, 1, 0},
//        {0, 0, 1, 0, 1},
//        {0, 0, 0, 1, 0},
//        {1, 0, 1, 0, 1}};
//        if (isRectangle(mat)) {
//            System.out.println("Yes");
//        } else {
//            System.out.println("No");
//        }
//
//        int mat2[][] = {{0, 1, 1},
//        {0, 1, 1},
//        {0, 1, 0}};
//        if (isRectangle(mat2)) {
//            System.out.println("Yes");
//        } else {
//            System.out.println("No");
//        }
//        
//        int mat3[][] = {{0, 1, 1},
//        {0, 0, 1},
//        {0, 1, 0}};
//        if (isRectangle(mat3)) {
//            System.out.println("Yes");
//        } else {
//            System.out.println("No");
//        }
//..............................................................................
//        System.out.println("Remove duplicates from linkedlist using iterative and hashing ways");
//        //https://leetcode.com/problems/remove-duplicates-from-sorted-list/submissions/
//        //https://www.geeksforgeeks.org/remove-duplicates-from-an-unsorted-linked-list/
//        Node<Integer> n1 =  new Node<>(1);
//        n1.setNext(new Node<>(1));
//        n1.getNext().setNext(new Node<>(2));
//        
//        Node<Integer> n2 =  new Node<>(1);
//        n2.setNext(new Node<>(1));
//        n2.getNext().setNext(new Node<>(2));
//        
//        removeDuplicateFromLinkedList_Iterative(n1);
//        removeDuplicateFromLinkedList_Hashing(n2);
//        
//        n1 = new Node<>(5);
//        n1.setNext(new Node<>(1));
//        n1.getNext().setNext(new Node<>(1));
//        n1.getNext().getNext().setNext(new Node<>(1));
//        n1.getNext().getNext().getNext().setNext(new Node<>(6));
//        n1.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        n1.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        n1.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        n1.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        n1.getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        n1.getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        n1.getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        
//        n2 = new Node<>(5);
//        n2.setNext(new Node<>(1));
//        n2.getNext().setNext(new Node<>(1));
//        n2.getNext().getNext().setNext(new Node<>(1));
//        n2.getNext().getNext().getNext().setNext(new Node<>(6));
//        n2.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        n2.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        n2.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        n2.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        n2.getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        n2.getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        n2.getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        
//        removeDuplicateFromLinkedList_Iterative(n1);
//        removeDuplicateFromLinkedList_Hashing(n2);
//..............................................................................
//        System.out.println("Remove nodes from BST that are not in the given range");
//        //youtube  coding simplified
//        TreeNode<Integer> root = new TreeNode<>(25);
//        root.setLeft(new TreeNode<>(12));
//        root.getLeft().setLeft(new TreeNode<>(10));
//        root.getLeft().setRight(new TreeNode<>(17));
//        root.getLeft().getRight().setLeft(new TreeNode<>(15));
//        root.setRight(new TreeNode<>(37));
//        root.getRight().setLeft(new TreeNode<>(32));
//        root.getRight().setRight(new TreeNode<>(40));
//        new BinaryTree<Integer>(root).treeBFS();
//        System.out.println();
//        TreeNode<Integer> rootMod = removeNodeFromBSTNotInRange(root, 12, 35);
//        new BinaryTree<Integer>(rootMod).treeBFS();
//        System.out.println();
//        
//        root = new TreeNode<>(25);
//        root.setLeft(new TreeNode<>(12));
//        root.getLeft().setLeft(new TreeNode<>(10));
//        root.getLeft().setRight(new TreeNode<>(17));
//        root.getLeft().getRight().setLeft(new TreeNode<>(15));
//        root.setRight(new TreeNode<>(37));
//        root.getRight().setLeft(new TreeNode<>(32));
//        root.getRight().setRight(new TreeNode<>(40));
//        new BinaryTree<Integer>(root).treeBFS();
//        System.out.println();
//        //out of range case
//        rootMod = removeNodeFromBSTNotInRange(root, 41, 100);
//        //tree  got empty :)
//        new BinaryTree<Integer>(rootMod).treeBFS();
//        System.out.println();
//..............................................................................
//        System.out.println("Detect a loop in the linked list");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-213-off-campus-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/detect-loop-in-a-linked-list/
//        //for iterative approach explanation : https://www.geeksforgeeks.org/how-does-floyds-slow-and-fast-pointers-approach-work/
//        //10 -> 15 -> [20] -> 25 -> 30 -> [20]
//        Node<Integer> node = new Node<>(10);
//        node.setNext(new Node<>(15));
//        node.getNext().setNext(new Node<>(20));
//        node.getNext().getNext().setNext(new Node<>(25));
//        node.getNext().getNext().getNext().setNext(new Node<>(30));
//        node.getNext().getNext().getNext().getNext().setNext(node.getNext().getNext());
//        detectLoopInLinkedList_Hashing(node);
//        detectLoopInLinkedList_Iterative(node);
//        
//        node = new Node<>(10);
//        node.setNext(new Node<>(15));
//        node.getNext().setNext(new Node<>(20));
//        node.getNext().getNext().setNext(new Node<>(25));
//        node.getNext().getNext().getNext().setNext(new Node<>(30));
//        detectLoopInLinkedList_Hashing(node);
//        detectLoopInLinkedList_Iterative(node);      
//..............................................................................
//        System.out.println("Detect and removing the loop in the linked list");
//        //https://www.geeksforgeeks.org/detect-and-remove-loop-in-a-linked-list/
//        //10 -> 15 -> [20] -> 25 -> 30 -> [20]
//        Node<Integer> node = new Node<>(10);
//        node.setNext(new Node<>(15));
//        node.getNext().setNext(new Node<>(20));
//        node.getNext().getNext().setNext(new Node<>(25));
//        node.getNext().getNext().getNext().setNext(new Node<>(30));
//        node.getNext().getNext().getNext().getNext().setNext(node.getNext().getNext());
//        detectAndRemovingLoopInLinkedList_UsingHashing(node);
//
//        node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(node.getNext().getNext());
//        detectAndRemovingLoopInLinkedList_UsingIteration(node);
//..............................................................................
//        System.out.println("Relative Sorting");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-213-off-campus-for-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/relative-sorting/0
//        //https://leetcode.com/problems/relative-sort-array/
//        int[] a = {2, 1, 2, 5, 7, 1, 9, 3, 6, 8, 8};
//        int[] b = {2, 1, 8, 3};
//        sortAListInRelativeOrderOfBList(a, b);
//        
//        int[] a2 = {2,21,43,38,0,42,33,7,24,13,12,27,12,24,5,23,29,48,30,31};
//        int[] b2 = {2,42,38,0,43,21};
//        sortAListInRelativeOrderOfBList(a2, b2);
//..............................................................................
//        System.out.println("Remove the Nth node from end in linked list");
//        //https://leetcode.com/problems/remove-nth-node-from-end-of-list/
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        new LinkedListUtil<Integer>(removeNthFromEnd(node, 3)).print();
//        node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        new LinkedListUtil<Integer>(removeNthFromEnd(node, 5)).print();
//        node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        new LinkedListUtil<Integer>(removeNthFromEnd(node, 1)).print();
//..............................................................................
//        System.out.println("Children sum property of tree");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-213-off-campus-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/check-for-children-sum-property-in-a-binary-tree/
//        TreeNode<Integer> root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(8));
//        root.getLeft().setLeft(new TreeNode<>(3));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(2));
//        root.getRight().setLeft(new TreeNode<>(2));
//        System.out.println("Has children sum property: "+childrenSumPropertyOfTree(root));
//        
//        root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(3));
//        root.getLeft().setLeft(new TreeNode<>(3));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(2));
//        root.getRight().setLeft(new TreeNode<>(2));
//        System.out.println("Has children sum property: "+childrenSumPropertyOfTree(root));
//..............................................................................
//        System.out.println("Buy and sell stocks to get maximum profit");
//        //https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
//        //https://www.geeksforgeeks.org/stock-buy-sell/
//        System.out.println(buyAndSellStockMaxProfit(new int[]{7, 1, 5, 3, 6, 4}));
//        stockBuySellTime(new int[]{7, 1, 5, 3, 6, 4}, 6);
//        System.out.println(buyAndSellStockMaxProfit(new int[]{7, 6, 4, 3, 1}));
//        stockBuySellTime(new int[]{7, 6, 4, 3, 1}, 5);
//        System.out.println(buyAndSellStockMaxProfit(new int[]{2, 4, 1}));
//        stockBuySellTime(new int[]{2,4,1}, 3);
//        System.out.println(buyAndSellStockMaxProfit(new int[]{100, 180, 260, 310, 40, 535, 695}));
//        stockBuySellTime(new int[]{100, 180, 260, 310, 40, 535, 695 }, 7);
//..............................................................................
//        System.out.println("Valid paranthesis pair checker");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-213-off-campus-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/check-for-balanced-parentheses-in-an-expression/
//        System.out.println("valid paranthesis? "+parenthesisPairChecker("([{}])"));
//        System.out.println("valid paranthesis? "+parenthesisPairChecker("([)]"));
//        System.out.println("valid paranthesis? "+parenthesisPairChecker("]})({["));
//        System.out.println("valid paranthesis? "+parenthesisPairChecker("({["));
//        System.out.println("valid paranthesis? "+parenthesisPairChecker("]})"));
//..............................................................................
//        System.out.println("Longest valid pair of prenthesis '(' and ')'");
//        //https://leetcode.com/problems/longest-valid-parentheses/
//        System.out.println("longest valid paranthesis? " + longestValidParentheses("(()"));
//        System.out.println("longest valid paranthesis? " + longestValidParentheses(")()())"));
//        System.out.println("longest valid paranthesis? " + longestValidParentheses(")))"));
//        System.out.println("longest valid paranthesis? " + longestValidParentheses("((("));
//        System.out.println("longest valid paranthesis? " + longestValidParentheses(")))((("));
//..............................................................................
//        System.out.println("Valid Parenthesis String");
//        //https://leetcode.com/problems/valid-parenthesis-string/
//        System.out.println("valid paranthesis? " + checkValidString("(*)"));
//        System.out.println("valid paranthesis? " + checkValidString("(*)*)"));
//..............................................................................
//        System.out.println("Peek element in the array O(LogN)");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-150-sde1-1-year-experienced/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/peak-element/1
//        //https://www.geeksforgeeks.org/find-a-peak-in-a-given-array/
//        int[] arr = {1, 3, 20, 4, 1, 0 };
//        System.out.println("peek element at "+peakElement_OLogN(arr, arr.length));
//        int[] arrr = {10, 20, 30, 40, 50};
//        System.out.println("peek element at "+peakElement_OLogN(arrr, arrr.length));
//        int[] arrrr = {50, 40, 30, 20, 10};
//        System.out.println("peek element at "+peakElement_OLogN(arrrr, arrrr.length));
//..............................................................................
//        System.out.println("Invert a binary tree");
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(3));
//        root.getRight().setLeft(new TreeNode<>(6));
//        root.getRight().setRight(new TreeNode<>(7));
//        BinaryTree<Integer> bt = new BinaryTree<>(root);
//        bt.treeBFS();
//        System.out.println();
//        invertABinaryTree(root);
//        
//        System.out.println();
//        root = new TreeNode<>(1);
//        root.setRight(new TreeNode<>(3));
//        root.getRight().setLeft(new TreeNode<>(6));
//        root.getRight().setRight(new TreeNode<>(7));
//        bt = new BinaryTree<>(root);
//        bt.treeBFS();
//        System.out.println();
//        invertABinaryTree(root);
//..............................................................................
//        System.out.println("Count the no of words in the stream");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-150-sde1-1-year-experienced/?ref=rp
//        //https://www.geeksforgeeks.org/count-words-in-a-given-string/
//        countNoOfWords("One two       three\n four\tfive  ");
//        countNoOfWords("\n\t \n\t ");
//        countNoOfWords("\n\tbcdef\n\t ");
//..............................................................................
//        System.out.println("Rotate and delete");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-150-sde1-1-year-experienced/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/rotate-and-delete/0
//        //http://geeksalgorithms.blogspot.com/2017/01/rotate-and-delete.html
//        rotateAndDelete();
//..............................................................................
//        System.out.println("1038. Binary Search Tree to Greater Sum Tree");
//        //https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/
//        TreeNode<Integer> root = new TreeNode<>(4);
//        root.setLeft(new TreeNode<>(1));
//        root.getLeft().setLeft(new TreeNode<>(0));
//        root.getLeft().setRight(new TreeNode<>(2));
//        root.getLeft().getRight().setRight(new TreeNode<>(3));
//        root.setRight(new TreeNode<>(6));
//        root.getRight().setLeft(new TreeNode<>(5));
//        root.getRight().setRight(new TreeNode<>(7));
//        root.getRight().getRight().setRight(new TreeNode<>(8));
//        bstToGreaterSumTree(root); //expected output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
//..............................................................................
//        System.out.println("Find minimum difference between any two elements");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-254-off-campus-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/minimum-difference-pair/0
//        //https://www.geeksforgeeks.org/find-minimum-difference-pair/
//        int arr[] = new int[]{1, 5, 3, 19, 18, 25}; 
//        System.out.println("Minimum difference is "+ minDifferenceBetweenTwoPairOfElement_ONLogN(arr));
//..............................................................................
//        System.out.println("832. Flipping an Image");
//        //https://leetcode.com/problems/flipping-an-image/
//        int[][] A = {
//            {1,1,0},
//            {1,0,1},
//            {0,0,0}
//        };
//        flipAndInvertImage(A);
//..............................................................................
//        System.out.println("Rearrange a Linked List in Zig-Zag fashion");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-254-off-campus-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/linked-list-in-zig-zag-fashion/
//        Node<Integer> head = new Node<>(11);
//        head.setNext(new Node<>(15));
//        head.getNext().setNext(new Node<>(20));
//        head.getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().setNext(new Node<>(10));
//        zigZagLinkedList(head);
//        //asc sorted liste
//        head = new Node<>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        zigZagLinkedList(head);
//        //desc sorted liste
//        head = new Node<>(6);
//        head.setNext(new Node<>(5));
//        head.getNext().setNext(new Node<>(4));
//        head.getNext().getNext().setNext(new Node<>(3));
//        head.getNext().getNext().getNext().setNext(new Node<>(2));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(1));
//        zigZagLinkedList(head);
//..............................................................................
//        System.out.println("Sum of all the left childs of any node in the given tree");
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.getLeft().getRight().setLeft(new TreeNode<>(6));
//        root.setRight(new TreeNode<>(3));
//        root.getRight().setLeft(new TreeNode<>(7));
//        root.getRight().setRight(new TreeNode<>(8));
//        root.getRight().getRight().setLeft(new TreeNode<>(9));
//        sumOfOnlyLeftChildOfNodeInTree(root);
//
//        root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.setRight(new TreeNode<>(3));
//        sumOfOnlyLeftChildOfNodeInTree(root);
//..............................................................................
//        System.out.println("Sum of Left Leaf Nodes");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-254-off-campus-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/sum-of-left-leaf-nodes/1
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.getLeft().getRight().setLeft(new TreeNode<>(6));
//        root.setRight(new TreeNode<>(3));
//        root.getRight().setLeft(new TreeNode<>(7));
//        root.getRight().setRight(new TreeNode<>(8));
//        root.getRight().getRight().setLeft(new TreeNode<>(9));
//        sumOfLeftLeafNodesOfTree(root);
//        
//        root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.setRight(new TreeNode<>(3));
//        sumOfLeftLeafNodesOfTree(root);
//..............................................................................
//        System.out.println("Word Break Problem");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-254-off-campus-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/word-break-problem-dp-32/ 
//        String temp_dictionary[] = {"mobile","samsung","sam","sung",  
//                            "man","mango","icecream","and",  
//                            "go","i","like","ice","cream"}; 
//                              
//        // loop to add all strings in dictionary set  
//        Set<String> dictionary = new HashSet<>();
//        for (String temp :temp_dictionary) { 
//            dictionary.add(temp); 
//        } 
//          
//        // sample input cases 
//        System.out.println(wordBreak(dictionary, "ilikesamsung")); 
//        System.out.println(wordBreak(dictionary, "iiiiiiii")); 
//        System.out.println(wordBreak(dictionary, "")); 
//        System.out.println(wordBreak(dictionary, "ilikelikeimangoiii")); 
//        System.out.println(wordBreak(dictionary, "samsungandmango")); 
//        System.out.println(wordBreak(dictionary, "samsungandmangok")); 
//..............................................................................
//        System.out.println("Common prefix | HackerRank | Akash Inst.");
//        List<String> testCases = Arrays.asList("abcabcd", "ababaa", "aa");
//        System.out.println("Output per test case: " + commonPrefix(testCases));
//..............................................................................
//        System.out.println("Generate BST from preorder List/Array");
//        int[] preorder = {2,1,3};
//        bstFromPreOrderArray(preorder);
//        int[] preorder2 = {4,1,0,2,3,6,5,10};
//        bstFromPreOrderArray(preorder2);
//..............................................................................
//        System.out.println("Generate BST from postorder List/Array");
//        //https://www.geeksforgeeks.org/construct-a-binary-search-tree-from-given-postorder/
//        int[] postorder = {1,3,2};
//        bstFromPostOrderArray(postorder);
//        int[] postorder2 = {1, 7, 5, 50, 40, 10};
//        bstFromPostOrderArray(postorder2);
//..............................................................................
//        System.out.println("1026. Maximum Difference Between Node and Ancestor");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-259-1-yr-experienced-for-sde1/?ref=rp
//        //https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/
//        TreeNode<Integer> root = new TreeNode<>(8);
//        root.setLeft(new TreeNode<>(3));
//        root.getLeft().setLeft(new TreeNode<>(1));
//        root.getLeft().setRight(new TreeNode<>(6));
//        root.getLeft().getRight().setLeft(new TreeNode<>(4));
//        root.getLeft().getRight().setRight(new TreeNode<>(7));
//        root.setRight(new TreeNode<>(10));
//        root.getRight().setRight(new TreeNode<>(14));
//        root.getRight().getRight().setLeft(new TreeNode<>(13));
//        System.out.println("max diff between node and its ancestor "+maxDiffBetweenNodeAndAncestor(root));
//..............................................................................
//        System.out.println("Largest Rectangular Area in a Histogram");
//        int hist[] = {6, 2, 5, 4, 5, 1, 6};
//        maxAreaOfHistogram(hist, hist.length);
//..............................................................................
//        System.out.println("Move all zeroes to end of array");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-259-1-yr-experienced-for-sde1/?ref=rp
//        //https://practice.geeksforgeeks.org/problems/move-all-zeroes-to-end-of-array0751/1
//        //https://leetcode.com/problems/move-zeroes/
//        int arr[] = {3, 5, 0, 0, 4};
//        pushZerosToEnd(arr);
//        int arrr[] = {0, 0, 0, 4};
//        pushZerosToEnd(arrr);
//..............................................................................
//        System.out.println("Calculate the angle between hour hand and minute hand");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-281-for-sde1/?ref=rp
//        //https://www.geeksforgeeks.org/calculate-angle-hour-hand-minute-hand/
//        calculateAngleBetweenHourAndMin(9, 60);
//        calculateAngleBetweenHourAndMin(3, 30);
//        calculateAngleBetweenHourAndMin(3, 15);
//..............................................................................
//        System.out.println("Find given no first and last occurence in the sorted array");
//        //https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
//        //https://www.geeksforgeeks.org/find-first-and-last-positions-of-an-element-in-a-sorted-array/
//        //BINARY SEARCH
//        firstAndLastOccurenceInSortedArray(new int[]{5,7,7,8,8,10}, 8);
//        firstAndLastOccurenceInSortedArray(new int[]{5,7,7,8,8,10}, 3);
//..............................................................................
//        System.out.println("longest common subsequence 3 ways");
//        String a = "abcdefg";
//        String b = "bdfg";
//
//        System.out.println("recursive O(2^N)");
//        //recursive -> optimal substructure but overlapping sub problem is not resolved
//        System.out.println(longestCommonSubsequence_Recursive(a, b, a.length(), b.length()));
//
//        System.out.println("dp memoization O(m*n) as memo[m][n]");
//        //recursive -> optimal substructure and overlapping sub problem is resolved using memo 
//        int[][] memo = new int[a.length() + 1][b.length() + 1];
//        for (int[] r : memo) {
//            Arrays.fill(r, -1);
//        }
//        System.out.println(longestCommonSubsequence_DP_Memoization(a, b, a.length(), b.length(), memo));
////        check how memo  changes after call
////        for (int[] r : memo) {
////            for (int c : r) {
////                System.out.print(c + "\t");
////            }
////            System.out.println();
////        }
//
//        System.out.println("dp top down O(m*n) as memo[m][n]");
//        //iterative -> recursive calls converted to top down calculations 
//        System.out.println(longestCommonSubsequence_DP_TopDown(a, b, a.length(), b.length()));
//..............................................................................
//        System.out.println("0 1 knapsack 3 ways");
//        int val[] = new int[]{60, 100, 120};
//        int wt[] = new int[]{10, 20, 30};
//        int W = 50;
//        int n = val.length;
//        System.out.println("recursive O(2^N)");
//        System.out.println(knapSack01_Recusrion(val, wt, n, W));
//
//        System.out.println("dp memoization O(m*n) as memo[m][n]");
//        int[][] memo = new int[n + 1][W + 1];
//        for (int[] r : memo) {
//            Arrays.fill(r, 0);
//        }
//        System.out.println(knapSack01_DP_Memoization(val, wt, n, W, memo));
//        System.out.println("dp top down O(m*n) as memo[m][n]");
//        System.out.println(knapSack01_DP_TopDown(val, wt, n, W));
//..............................................................................
//        System.out.println("subset sum 3 ways(variation in 01knapsack)");
//        int set[] = { 3, 34, 4, 12, 5, 2 }; 
//        int sum = 9; 
//        int n = set.length; 
//        System.out.println("recursive O(2^N)");
//        System.out.println(subsetSum_Recursion(set, n, sum));
//        System.out.println("dp memoization O(m*n) as memo[m][n]");
//        //base cond
//        boolean[][] memo = new boolean[n+1][sum+1];
//        for(int x=0; x<n+1; x++){
//            for(int y=0; y<sum+1; y++){
//                if(x == 0){
//                    memo[x][y] = false;
//                }
//                
//                if(y == 0){
//                    memo[x][y] = true;
//                }
//            }
//        }
//        System.out.println(subsetSum_DP_Memoization(set, n, sum, memo));
//        System.out.println("dp top down O(m*n) as memo[m][n]");
//        System.out.println(subsetSum_DP_TopDown(set, n, sum));
//..............................................................................
//        System.out.println("Equal sum subset (variation in 01knapsack)");
//        int arr[] = {3, 1, 5, 9, 12}; 
//        int n = arr.length; 
//        //same can be found using dp_memoization and dp_topdown
//        System.out.println(equalSumSubsetPartition(arr));
//..............................................................................
//        System.out.println("counting equal sum subset (variation in 01knapsack)");
//        int arr[] = {1, 2, 3, 3};
//        int n = arr.length;
//        int sum = 6;
//        System.out.println("recursive O(2^N)");
//        System.out.println(countSubsetSum_Recursion(arr, n, sum));
//        System.out.println("dp memoization O(m*n) as memo[m][n]");
//        //base cond
//        int[][] memo = new int[n + 1][sum + 1];
//        for (int x = 0; x < n + 1; x++) {
//            for (int y = 0; y < sum + 1; y++) {
//                if (x == 0) {
//                    memo[x][y] = 0;
//                }
//
//                if (y == 0) {
//                    memo[x][y] = 1;
//                }
//
//            }
//        }
//
//        System.out.println(countSubsetSum_DP_Memoization(arr, n, sum, memo));
//        System.out.println("dp top down O(m*n) as memo[m][n]");
//        System.out.println(countSubsetSum_DP_TopDown(arr, n, sum));
//..............................................................................
//        System.out.println("min diff possible using sum of subsets (variation in 01knapsack)");
//        int arr[] = {3, 1, 4, 2, 2, 1}; 
//        System.out.println(minDiffInEqualSubset(arr));
//        int ar[] = {1, 2, 5}; 
//        System.out.println(minDiffInEqualSubset(ar));
//..............................................................................
//        System.out.println("subset sum to from a given diff (variation in 01knapsack)");
//        int arr[] = {1, 2, 1, 3}; 
//        int n = arr.length; 
//        int diff = 1;
//        System.out.println(subsetSumToAGivenDifference(arr, n, diff));
//..............................................................................
//        System.out.println("target sum by assigning sign to array element (variation in 01knapsack)");
//        //https://leetcode.com/problems/target-sum/
//        int arr[] = {1, 1, 1, 1, 1}; 
//        int n = arr.length; 
//        int target = 3;
//        System.out.println(waysToFormAGivenTargetSumByAssigningSigns(arr, n, target));
//..............................................................................
//        System.out.println("rod cutting (variation in unbounded knapsack)");
//        int L = 8;
//        int price[] = new int[]{1, 5, 8, 9, 10, 17, 17, 20};
//        int length[] = new int[L];
//        for (int i = 0; i < L; i++) {
//            length[i] = i + 1;
//        }
//        System.out.println(rodCutting(length, price, L));
//..............................................................................
//        System.out.println("max ways in which a Change can be made with given unbounded supply of coin (variation in unbounded knapsack)");
//        int coins[] = {1, 2, 3}; 
//        int n = coins.length; 
//        int K = 5;
//        System.out.println( waysToMakeCoinChange(coins, n, K)); 
//..............................................................................
//        System.out.println("min no of coins used to make Change with given unbounded supply of coin (variation in unbounded knapsack and coin change)");
//        int coins[] = {1, 2, 3}; 
//        int n = coins.length; 
//        int K = 5;
//        System.out.println( minNoOfCoinsUsedForChange(coins, n, K)); //2
//        System.out.println( minNoOfCoinsUsedForChange(new int[]{25, 10, 5}, 3, 30)); //2
//        //2+3, 1+1+1+2, 1+1+3, 1+1+1+1+1, 1+2+2
//        //max ways to make change for K=5 is above 5 ways
//        //out of that we used just 2 coins in 2,3 that are min no of coins used in above 5 ways. 
//..............................................................................
//        System.out.println("longest common substring (variation in longest common subseq)");
//        String X = "OldSite:GeeksforGeeks.org";
//        String Y = "NewSite:GeeksQuiz.com";
//        int m = X.length();
//        int n = Y.length();
//        System.out.println(longestCommonSubstring(X, Y, m, n));
//        X = "xyzabcpqrs";
//        Y = "jklmnoabcefgh";
//        m = X.length();
//        n = Y.length();
//        System.out.println(longestCommonSubstring(X, Y, m, n));
//..............................................................................        
//        System.out.println("printing longest common subsequence string");
//        String a = "abcdefg";
//        String b = "bxdlfng";
//        printLongestCommonSubsequence_DP_TopDown(a, b, a.length(), b.length());
//..............................................................................
//        System.out.println("shortest common supersequence 3 ways");
//        String X = "AGGTAB";
//        String Y = "GXTXAYB";
//        System.out.println("recursive O(2^N)");
//        System.out.println(shortestCommonSuperSequence_Recusrion(X, Y, X.length(), Y.length()));
//        System.out.println("dp memoization O(m*n) as memo[m][n]");
//        int[][] memo = new int[X.length() + 1][Y.length() + 1];
//        //base cond 
//        for (int x = 0; x < X.length() + 1; x++) {
//            for (int y = 0; y < Y.length() + 1; y++) {
//
//                if (x != 0 || y != 0) {
//                    memo[x][y] = -1;
//                }
//
//                //if x=0 row string X is empty  return 
//                //Y
//                if (x == 0) {
//                    memo[x][y] = y;
//                }
//
//                //if y=0 row string Y is empty  return 
//                //X
//                if (y == 0) {
//                    memo[x][y] = x;
//                }
//
//            }
//        }
//        System.out.println(shortestCommonSuperSequence_DP_Memoization(X, Y, X.length(), Y.length(), memo));
//        System.out.println("dp top down O(m*n) as memo[m][n]");
//        System.out.println(shortestCommonSuperSequence_DP_TopDown(X, Y, X.length(), Y.length()));
//        System.out.println("SCS approach based on using LCS");
//        System.out.println(shortestCommonSuperSequnce_BasedDirectOnLCS(X, Y, X.length(), Y.length()));
//..............................................................................
//        System.out.println("Minimum number of deletions and insertions to transform one string into another");
//        //https://www.geeksforgeeks.org/minimum-number-deletions-insertions-transform-one-string-another/
//        //youtube aditya verma
//        String str1 = "heap";
//        String str2 = "pea";
//        minInsertOrDeleteReqToConvertAToBString(str1, str2);
//..............................................................................
//        System.out.println("Longest pallindromic subseq using longest common subseq as parent");
//        String seq = "GEEKSFORGEEKS";
//        System.out.println(longestPallindromicSubsequence_LCSubseqAsParent(seq));
//..............................................................................
//        System.out.println("min deletion required to make string pallindromic");
//        //.......O(N^2)
//        System.out.println(minNoOfDeletionInStringToMakeItPallindrome_LPSBasedApproach("ABAABCD"));
//        System.out.println(minNoOfDeletionInStringToMakeItPallindrome_LPSBasedApproach("ABAABCCD"));
//        System.out.println(minNoOfDeletionInStringToMakeItPallindrome_LPSBasedApproach("ABA"));
//..............................................................................
//        System.out.println("printing shortest common superseq");
//        String X = "AGGTAB";
//        String Y = "GXTXAYB";
//        printShortestCommonSuperSequence(X, Y, X.length(), Y.length());
//..............................................................................
//        System.out.println("Longest Repeating subseq LCSModification");
//        String s = "aabebcdd"; //abd is repeating subseq
//        System.out.println("recursive O(2^N)");
//        System.out.println(longestRepeatingSubsequence_Recursion(s));
//        System.out.println("dp memoization O(m*n) as memo[m][n]");
//        //memoization
//        int[][] memo = new int[s.length()+1][s.length()+1];
//        //base cond same as LCS
//        for(int[] r: memo){
//            Arrays.fill(r, 0);
//        }
//        System.out.println(longestRepeatingSubsequence_DP_Memoization(s, s, s.length(), s.length(), memo));
//        System.out.println("dp top down O(m*n) as memo[m][n]");
//        System.out.println(longestRepeatingSubsequence_DP_TopDown(s, s, s.length(), s.length()));
//..............................................................................
//        System.out.println("Sequence pattern matching");
//        sequencePatternMatching("achxgyij", "axy");
//        sequencePatternMatching("achxgyij", "zxy");
//        sequencePatternMatching("achxgyij", "axya");
//        sequencePatternMatching("achxgyij", "achxgyij");
//        sequencePatternMatching("achxgyij", "achxgyijachxgyij");
//..............................................................................
//        System.out.println("min no of insertion to make a string pallindromic");
//        System.out.println(minNoOfInsertionInStringToMakeItPallindrome("agbcba"));
//        System.out.println(minNoOfInsertionInStringToMakeItPallindrome("aebcbda"));
//        System.out.println(minNoOfInsertionInStringToMakeItPallindrome("aba"));
//..............................................................................
//        System.out.println("Matrix chain multiplication for low cost");
//        int arr[] = new int[]{1, 2, 3, 4, 3};
//        int n = arr.length;
//        System.out.println("recursive O(2^N)");
//        System.out.println(matrixChainMultiplicationLowCost_Recursion(arr, 1, n - 1));
//        System.out.println("dp memoization O(m*n) as memo[m][n]");
//        //memoization
//        int[][] memo = new int[arr.length+1][arr.length+1];
//        for(int[] r: memo){
//            Arrays.fill(r, -1);
//        }
//        System.out.println(matrixChainMultiplicationLowCost_DP_Memoization(arr, 1, n - 1, memo));
//..............................................................................
//        System.out.println("pallindrome partioning");
//        String a = "nitin";
//        System.out.println(pallindromePartitioning_Recursion(a, 0, a.length())); //0
//        //memoization
//        String b = "nitik";
//        int[][] memo = new int[b.length() + 1][b.length() + 1];
//        for (int[] r : memo) {
//            Arrays.fill(r, -1);
//        }
//        System.out.println(pallindromePartitioning_DP_Memoization(b, 0, b.length(), memo));
////        String b = "ababbbabbababa";
////        System.out.println(pallindromePartioninng(b, 0, b.length()));
//..............................................................................
//        System.out.println("boolean parenthesization");
//        String expr =  "T|T&F^T";
//        System.out.println(booleanParenthersizationThatEqualsMatch(expr, 0, expr.length(), true));
//..............................................................................
//        System.out.println("scrambled strings");
////        String S1 = "great";
////        String S2 = "grate";
//        String S1 = "coder";
//        String S2 = "ocred";
//        if (scrambledString(S1, S2)) {
//            System.out.println("yes");
//        } else {
//            System.out.println("no");
//        }
//..............................................................................
        System.out.println("egg dropping problem");
        int floor = 10;
        int egg = 2;
        System.out.println(eggDroppingTrials(floor, egg));
    }

}
