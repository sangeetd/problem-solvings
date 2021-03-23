/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
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

    private void nextPermutation_Print(int[] nums) {
        for (int x : nums) {
            System.out.print(x);
        }
        System.out.println();
    }

    public void nextPermutation(int[] nums) {

        int N = nums.length;

        //length == 1
        if (N == 1) {
            //output:
            nextPermutation_Print(nums);
            return;
        }

        //check if asc sorted
        boolean ascSorted = false;
        for (int i = 0; i < N - 1; i++) {
            if (nums[i] < nums[i + 1]) {
                ascSorted = true;
            } else {
                ascSorted = false;
                break;
            }
        }

        if (ascSorted) {
            //swap only last 2 digit
            swapIntArray(nums, N - 1, N - 2);

            //output:
            nextPermutation_Print(nums);
            return;
        }

        //check if desc sorted
        boolean descSorted = false;
        for (int i = 0; i < N - 1; i++) {
            if (nums[i] > nums[i + 1]) {
                descSorted = true;
            } else {
                descSorted = false;
                break;
            }
        }

        if (descSorted) {
            //just reverse the num
            for (int i = 0; i < N / 2; i++) {
                swapIntArray(nums, i, N - i - 1);
            }

            //output:
            nextPermutation_Print(nums);
            return;
        }

        //any other cases
        int firstDecValueFromRight = nums[N - 1];
        int index = 0;
        for (int i = N - 2; i >= 0; i--) {
            if (nums[i] < firstDecValueFromRight) {
                index = i;
                break;
            } else {
                firstDecValueFromRight = nums[i];
            }
        }

        int justLargerNumFromFirstDecValue = nums[index + 1] - nums[index];
        int swapWith = index + 1;
        for (int i = index + 2; i < N; i++) {
            if ((nums[i] - nums[index] > 0) && nums[i] - nums[index] < justLargerNumFromFirstDecValue) {
                justLargerNumFromFirstDecValue = nums[i] - nums[index];
                swapWith = i;
            }
        }

        swapIntArray(nums, index, swapWith);
        Arrays.sort(nums, index + 1, N);

        //output:
        nextPermutation_Print(nums);
    }

    private int factorialLargeNumber_Multiply(int x, int[] res, int resSize) {

        int carry = 0;
        for (int i = 0; i < resSize; i++) {
            int prod = res[i] * x + carry;
            res[i] = prod % 10;
            carry = prod / 10;
        }

        while (carry != 0) {

            res[resSize] = carry % 10;
            carry = carry / 10;
            resSize++;
        }

        return resSize;
    }

    public void factorialLargeNumber(int N) {
        int[] res = new int[Integer.MAX_VALUE / 200];
        res[0] = 1;

        int resSize = 1;
        for (int x = 2; x <= N; x++) {
            resSize = factorialLargeNumber_Multiply(x, res, resSize);
        }

        //output
        for (int i = resSize - 1; i >= 0; i--) {
            System.out.print(res[i]);
        }
        System.out.println();
    }

    public void rainWaterTrappingUsingStack(int[] height) {

        //https://leetcode.com/problems/trapping-rain-water/solution/
        //..................T: O(N)
        //..................S: O(N)
        int ans = 0;
        int current = 0;
        int N = height.length;
        Stack<Integer> s = new Stack<>();
        while (current < N) {

            while (!s.isEmpty() && height[current] > height[s.peek()]) {

                int top = s.pop();
                if (s.isEmpty()) {
                    break;
                }

                int distance = current - s.peek() - 1;
                int boundedHeight = Math.min(height[current], height[s.peek()]) - height[top];
                ans += distance * boundedHeight;

            }

            s.push(current++);
        }

        //output
        System.out.println("Rain water trapping using stack: " + ans);

    }

    public void rainWaterTrappingUsingTwoPointers(int[] height) {

        //https://leetcode.com/problems/trapping-rain-water/solution/
        //OPTIMISED than stack
        //..................T: O(N)
        //..................S: O(1)
        int left = 0;
        int right = height.length - 1;
        int ans = 0;
        int leftMax = 0;
        int rightMax = 0;
        while (left < right) {
            if (height[left] < height[right]) {

                if (height[left] >= leftMax) {
                    leftMax = height[left];
                } else {
                    ans += (leftMax - height[left]);
                }

                left++;
            } else {

                if (height[right] >= rightMax) {
                    rightMax = height[right];
                } else {
                    ans += (rightMax - height[right]);
                }

                right--;
            }
        }

        //output
        System.out.println("Rain water trapping using tow pointers: " + ans);

    }

    public void findMaximumProductSubarray(int[] arr) {

        //Explanation: https://www.youtube.com/watch?v=lXVy6YWFcRM
        int result = arr[0];
        int currMax = 1;
        int currMin = 1;
        for (int i = 0; i < arr.length; i++) {

            if (arr[i] == 0) {
                //just reset
                currMax = 1;
                currMin = 1;
                result = Math.max(arr[i], result);
                continue;
            }

            int tempCurrMax = arr[i] * currMax;
            currMax = Math.max(Math.max(arr[i] * currMax, arr[i] * currMin), arr[i]);
            currMin = Math.min(Math.min(tempCurrMax, arr[i] * currMin), arr[i]);
            result = Math.max(currMax, result);
        }

        //output:
        System.out.println("Maximum product subarray: " + result);

    }

    public void kadaneAlgorithm(int[] arr) {

        //for finding maximum sum subarray
        int maxSum = arr[0];
        int currMaxSum = arr[0];
        for (int i = 1; i < arr.length; i++) {
            currMaxSum = Math.max(arr[i], currMaxSum + arr[i]);
            maxSum = Math.max(maxSum, currMaxSum);
        }

        //output
        System.out.println("Max sum subarray: " + maxSum);

    }

    public void kadaneAlgorithm_PointingIndexes(int[] arr) {

        int maxSum = arr[0];
        int currMaxSum = arr[0];

        int start = 0;
        int end = 0;
        int index = 0;
        for (int i = 1; i < arr.length; i++) {

            currMaxSum += arr[i];
            if (maxSum < currMaxSum) {

                maxSum = currMaxSum;
                start = index;
                end = i;

            }

            if (currMaxSum < 0) {
                currMaxSum = 0;
                index = i + 1;
            }

        }

        //output:
        System.out.println("Max sum subarray with start & end: " + maxSum + " Start: " + start + " end: " + end);

    }

    public void moveNegativeElementsToOneSideOfArray(int[] arr) {

        //Two pointer approach
        //...........................T: O(N)
        //actual:
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();

        int f = 0;
        int h = arr.length - 1;

        while (h > f) {

            //as we are planning to shift all the -ve elements to left side of array
            //so cond. check that any +ve element (arr[f] > 0) in left side AND any -ve element(arr[h] <0)
            //on right side should be swapped 
            if ((arr[f] > 0 && arr[h] < 0)) {
//                System.out.println(arr[f]+" "+arr[h]+" : "+f+" "+h);
                swapIntArray(arr, f, h);
                f++;
                h--;
            }

            //if any element in left side id already a -ve then that element should be taken into consideration
            //move to next element
            if (arr[f] < 0) {
                f++;
            }

            //same way any +ve no on the right side should not be counted and move to next element
            if (arr[h] > 0) {
                h--;
            }

        }

        //output:
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void findUnionAndIntersectionOfTwoArrays(int[] a, int[] b) {

        int m = a.length;
        int n = b.length;
        int maxLen = Math.max(m, n);
        Set<Integer> unionSet = new HashSet<>();
        for (int i = 0; i < maxLen; i++) {

            if (i < m) {
                unionSet.add(a[i]);
            }

            if (i < n) {
                unionSet.add(b[i]);
            }

        }

        //output
        System.out.println("No of union element: " + unionSet.size() + " elements: " + unionSet);

        //finding intersection of two array
        Set<Integer> aSet = new HashSet<>();
        Set<Integer> bSet = new HashSet<>();
        for (int x : a) {
            aSet.add(x);
        }
        for (int x : b) {
            bSet.add(x);
        }

        Set<Integer> intersectionSet = new HashSet<>();
        for (int i = 0; i < maxLen; i++) {

            if (i < m) {
                if (aSet.contains(a[i]) && bSet.contains(a[i])) {
                    intersectionSet.add(a[i]);
                }
            }

            if (i < n) {
                if (aSet.contains(b[i]) && bSet.contains(b[i])) {
                    intersectionSet.add(b[i]);
                }
            }

        }

        //output
        System.out.println("No of intersection element: " + intersectionSet.size() + " elements: " + intersectionSet);

    }

    public void rotateArrayByK(int[] arr, int k) {

        //actual:
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();

        int n = arr.length;

        while (k-- != 0) {
            int last = arr[n - 1];
            for (int i = n - 1; i >= 1; i--) {
                arr[i] = arr[i - 1];
            }

            arr[0] = last;
        }

        //output:
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void minimizeDifferenceBetweenHeights(int[] arr, int k) {

        //problem statement: https://practice.geeksforgeeks.org/problems/minimize-the-heights3351/1
        //sol: https://www.geeksforgeeks.org/minimize-the-maximum-difference-between-the-heights/
        int n = arr.length;

        Arrays.sort(arr);

        int ans = arr[n - 1] - arr[0];

        int big = arr[0] + k;
        int small = arr[n - 1] - k;

        int temp = big;
        big = Math.max(big, small);
        small = Math.min(temp, small);

        //all in between a[0] to a[n-1] i.e, a[1] -> a[n-2]
        for (int i = 1; i < n - 1; i++) {

            int subtract = arr[i] - k;
            int add = arr[i] + k;

            // If both subtraction and addition 
            // do not change diff 
            if (subtract >= small || add <= big) {
                continue;
            }

            // Either subtraction causes a smaller 
            // number or addition causes a greater 
            // number. Update small or big using 
            // greedy approach (If big - subtract 
            // causes smaller diff, update small 
            // Else update big) 
            if (big - subtract <= add - small) {
                small = subtract;
            } else {
                big = add;
            }

        }

        //output:
        System.out.println("Min height: " + Math.min(ans, big - small));

    }

    public void bestProfitToBuySellStock(int[] prices) {

        //https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for (int i = 0; i < prices.length; i++) {
            //buy any stock at min price, so find a price < minPrice
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            }

            //if any price > minPrice, we can sell that stock to earn profit
            //maxProfit = max(maxProfit, price - minPrice)
            maxProfit = Math.max(maxProfit, prices[i] - minPrice);

        }

        //output:
        System.out.println("Maximum profit from buying and selling the stocks: " + maxProfit);

    }

    public void countAllPairsInArrayThatSumIsK(int[] arr, int K) {
        /*
         //brute force apprach
         //........................T: O(N^2)
         int pairCount = 0;
         for(int i=0; i<arr.length; i++){
         for(int j=i+1; j<arr.length; j++){
         if(arr[i] + arr[j] == K){
         pairCount++;
         }
         }
         }
        
         System.out.println("Count of pairs whose sum is equal to K: "+pairCount);
         */

        //Time optimised approach
        //https://www.geeksforgeeks.org/count-pairs-with-given-sum/
        //.......................T: O(N)
        //.......................S: O(N)
        Map<Integer, Integer> map = new HashMap<>();
        for (int element : arr) {
            map.put(element, map.getOrDefault(element, 0) + 1);
        }

        int pairCount = 0;
        for (int element : arr) {
            if (map.get(K - element) != null) {
                pairCount += map.get(K - element);
            }

            if (K - element == element) {
                pairCount--;
            }

        }

        System.out.println("Count of pairs whose sum is equal to K: " + pairCount / 2);
    }

    public boolean checkIfSubarrayWithSum0(int[] arr) {

        int n = arr.length;
        int sum = 0;
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < n; i++) {
            sum += arr[i];

            if (arr[i] == 0 || sum == 0 || set.contains(sum)) {
                return true;
            }
            set.add(sum);
        }
        return false;

    }

    public void bestProfitToBuySellStockAtMostTwice(int[] prices) {

        int n = prices.length;
        int[] maxProfits = new int[n];

        int currMaxPrice = prices[n - 1];
        for (int i = n - 2; i >= 0; i--) {

            if (prices[i] > currMaxPrice) {
                currMaxPrice = prices[i];
            }

            maxProfits[i] = Math.max(maxProfits[i + 1], currMaxPrice - prices[i]);

        }

        int currMinPrice = prices[0];
        for (int i = 1; i < n; i++) {

            if (currMinPrice > prices[i]) {
                currMaxPrice = prices[i];
            }

            maxProfits[i] = Math.max(maxProfits[i - 1], maxProfits[i] + (prices[i] - currMinPrice));

        }

        //output:
        System.out.println("Max profit frm buying selling stock atmost twice: " + maxProfits[n - 1]);

    }

    public void mergeIntervals_1(int[][] intervals) {

        //.................................T: O(N.LogN)
        System.out.println("approach 1");
        List<int[]> result = new ArrayList<>();

        if (intervals == null || intervals.length == 0) {
            //return result.toArray(new int[0][]);
            return;
        }

        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        int start = intervals[0][0];
        int end = intervals[0][1];

        for (int i = 1; i < intervals.length; i++) {

            int start_ = intervals[i][0];
            int end_ = intervals[i][1];

            //no overlapp situation
            if (end < start_) {
                result.add(new int[]{start, end});
                start = start_;
                end = end_;
            } else if (end_ >= end) {
                end = end_;
            }
        }

        //final pair
        result.add(new int[]{start, end});
        //output:
        int[][] output = result.toArray(new int[result.size()][]);
        for (int[] r : output) {
            System.out.print("[");
            for (int c : r) {
                System.out.print(c + " ");
            }
            System.out.println("]");
            System.out.println();
        }

    }

    public void mergeIntervals_2(int[][] intervals) {

        //.................................T: O(N.LogN)
        System.out.println("approach 2");
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        LinkedList<int[]> merged = new LinkedList<>();
        for (int[] interval : intervals) {

            if (merged.isEmpty() || merged.getLast()[1] < interval[0]) {
                merged.add(interval);
            } else {
                merged.getLast()[1] = Math.max(merged.getLast()[1], interval[1]);
            }
        }

        //output:
        int[][] output = merged.toArray(new int[merged.size()][]);
        for (int[] r : output) {
            System.out.print("[");
            for (int c : r) {
                System.out.print(c + " ");
            }
            System.out.println("]");
            System.out.println();
        }

    }

    public void minOperationsToMakeArrayPallindrome(int[] arr) {

        //TWO POINTERS
        int n = arr.length;
        int i = 0;
        int j = n - 1;
        int minOpr = 0;
        while (j >= i) {

            if (arr[i] == arr[j]) {
                i++;
                j--;
            } else if (arr[i] > arr[j]) {
                j--;
                arr[j] += arr[j + 1];
                minOpr++;
            } else {
                i++;
                arr[i] += arr[i - 1];
                minOpr++;
            }

        }

        //output:
        System.out.println("Minimum operation to make array pallindrome: " + minOpr);

    }

    public void productOfArrayExcludingElementItself_BruteForce(int[] arr) {

        //.....................T: O(N^2)
        //SOLUTION CAN GIVE TLE
        int n = arr.length;
        int[] result = new int[n];

        for (int i = 0; i < n; i++) {
            int prod = 1;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    prod *= arr[j];
                }
            }
            result[i] = prod;
        }

        //output:
        for (int x : result) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void productOfArrayExcludingElementItself_Optimised1(int[] arr) {

        //.....................T: O(N)
        //.....................S: O(N)
        int n = arr.length;
        int[] result = new int[n];
        int[] leftProd = new int[n];
        int[] rightProd = new int[n];

        leftProd[0] = 1;
        rightProd[n - 1] = 1;

        for (int i = 1; i < n; i++) {
            leftProd[i] = leftProd[i - 1] * arr[i - 1];
        }

        for (int i = n - 2; i >= 0; i--) {
            rightProd[i] = rightProd[i + 1] * arr[i + 1];
        }

        for (int i = 0; i < n; i++) {
            result[i] = leftProd[i] * rightProd[i];
        }

        //output:
        for (int x : result) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void productOfArrayExcludingElementItself_Optimised2(int[] arr) {

        //.....................T: O(N)
        //.....................S: O(1) //result[] is needed to save output 
        int n = arr.length;
        int[] result = new int[n];

        result[0] = 1;
        for (int i = 1; i < n; i++) {
            result[i] = result[i - 1] * arr[i - 1];
        }

        int right = 1;
        for (int i = n - 1; i >= 0; i--) {
            result[i] = result[i] * right;
            right *= arr[i];
        }

        //output:
        for (int x : result) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void maximumOfAllSubArrayOfSizeK(int[] arr, int K) {

        List<Integer> result = new ArrayList<>();
        int n = arr.length;
        int i;
        Deque<Integer> q = new LinkedList<>();
        for (i = 0; i < K; i++) {

            while (!q.isEmpty() && arr[q.peekFirst()] <= arr[i]) {
                q.removeFirst();
            }
            q.addLast(i);
        }

        for (; i < n; i++) {

            result.add(arr[q.peekFirst()]);

            while (!q.isEmpty() && q.peekFirst() <= i - K) {
                q.removeFirst();
            }

            while (!q.isEmpty() && arr[q.peekFirst()] <= arr[i]) {
                q.removeFirst();
            }
            q.addLast(i);
        }
        result.add(arr[q.peekFirst()]);

        //output:
        System.out.println("output: " + result);

    }

    public void averageWaitingTime(int[][] customer) {

        double prepTime = 0;
        double totalTime = 0;
        for (int i = 0; i < customer.length; i++) {
            prepTime = prepTime > customer[i][0] ? prepTime + customer[i][1] : customer[i][0] + customer[i][1];
            totalTime += (prepTime - customer[i][0]);
        }

        //output:
        System.out.println("Average waitig time: " + (totalTime / (double) customer.length));

    }

    public void threeSum(int[] arr) {

        //problem statement: https://leetcode.com/problems/3sum/
        //explanation: https://youtu.be/qJSPYnS35SE
        int n = arr.length;
        Arrays.sort(arr);
        List<List<Integer>> result = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            //if the current arr element is not same as prev 
            if (i == 0 || (i > 0 && arr[i] != arr[i - 1])) {

                int start = i + 1;
                int end = n - 1;
                int sum = 0 - arr[i];

                while (end > start) {

                    if (arr[start] + arr[end] == sum) {
                        result.add(Arrays.asList(arr[i], arr[start], arr[end]));

                        //before updating start and end
                        //check wheather there is any duplicates are there or not
                        //if there are duplactes update start and end until we reach the last
                        //duplicated element
                        while (start < end && arr[start] == arr[start + 1]) {
                            start++;
                        }

                        while (start < end && arr[end] == arr[end - 1]) {
                            end--;
                        }

                        //update start and end from the last duplicated element
                        start++;
                        end--;

                    } else if (arr[start] + arr[end] < sum) {
                        start++;
                    } else {
                        end--;
                    }
                }
            }

        }

        //output:
        System.out.println("All the triplets that sum to 0: " + result);

    }

    public int minimumTimeDifference(List<String> timePoints) {

        //problem statement: https://leetcode.com/problems/minimum-time-difference/
        //explanation: https://youtu.be/c5ecNf7JM1Q
        //1 hour = 60 min
        //24 hour = 24 * 60 min == 1440 min
        boolean[] everyMinute = new boolean[24 * 60]; //[1440]

        //convert the string "HH:MM" time format to minute only 
        for (String time : timePoints) {
            String[] timeSplit = time.split(":");
            int hour = Integer.parseInt(timeSplit[0]);
            int minute = Integer.parseInt(timeSplit[1]);

            //total minute in an hour = hour * 60 + minute
            //that total minute is minute based index in everyMinute
            int minuteIndex = (hour * 60) + minute;
            //if a minute is already seen and that same timePoint comes again
            //the diff b/w these two timePoints is minimum
            //ex: "10:02", "10:02"
            if (everyMinute[minuteIndex]) {
                return 0;
            }
            everyMinute[minuteIndex] = true;
        }

        int firstTime = -1;
        int prevTime = -1;
        int minDiff = Integer.MAX_VALUE;

        for (int i = 0; i < 1440; i++) {

            if (everyMinute[i]) {

                if (firstTime == -1) {

                    firstTime = i;
                    prevTime = i;

                } else {

                    minDiff = Math.min(minDiff, Math.min(
                            i - prevTime, //clockwise dir
                            1440 - i + prevTime //anti clockwise dir
                    ));

                    prevTime = i;
                }
            }

        }

        minDiff = Math.min(minDiff, Math.min(
                prevTime - firstTime, //clockwise dir
                1440 - prevTime + firstTime //anti clockwise dir
        ));

        return minDiff;

    }

    public void contigousArrayWithEqualZeroAndOne(int[] arr) {

        //https://leetcode.com/problems/contiguous-array/
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int count = 0;
        int maxLen = 0;

        for (int i = 0; i < arr.length; i++) {

            count += arr[i] == 1 ? 1 : -1;

            if (map.containsKey(count)) {
                maxLen = Math.max(maxLen, i - map.get(count));
            } else {
                map.put(count, i);
            }
        }

        //output:
        System.out.println("Max length: " + maxLen);

    }

    public void maxSumPathInTwoSortedArrays(int[] arr1, int[] arr2) {

        //https://www.geeksforgeeks.org/maximum-sum-path-across-two-arrays/
        int m = arr1.length;
        int n = arr2.length;

        int result = 0;
        int arrSum1 = 0;
        int arrSum2 = 0;

        int i = 0; // for arr1
        int j = 0; // for arr2

        while (i < m && j < n) {

            if (arr1[i] < arr2[j]) {
                arrSum1 += arr1[i++];
            } else if (arr1[i] > arr2[j]) {
                arrSum2 += arr2[j++];
            } else {
                //common point
                result += Math.max(arrSum1, arrSum2);

                arrSum1 = 0;
                arrSum2 = 0;

                int temp = i;
                while (i < m && arr1[i] == arr2[j]) {
                    arrSum1 += arr1[i++];
                }

                while (j < n && arr1[temp] == arr2[j]) {
                    arrSum2 += arr2[j++];
                }

                result += Math.max(arrSum1, arrSum2);

                arrSum1 = 0;
                arrSum2 = 0;

            }

        }

        while (i < m) {
            arrSum1 += arr1[i++];
        }

        while (j < n) {
            arrSum2 += arr2[j++];
        }

        result += Math.max(arrSum1, arrSum2);

        //output:
        System.out.println("Max path sum: " + result);

    }

    public void asteroidCollision(int[] asteroid) {

        //https://leetcode.com/problems/asteroid-collision/
        //explanation: https://youtu.be/6GGTBM7mwfs

        /*
         cond when two collide
         peek = -ve, incoming = -ve = left, left dir no collision //1 if cond in while()
         peek = -ve, incoming = +ve = left, right dir no collision //1 if cond in for()
         peek = +ve, incoming = +ve = right, right dir no collision //1 if cond in for()
         peek = +ve, incoming = -ve = right, left dir will collision
         if(abs(incoming) > peek) all peek will be destroyed and incoming will remain in sack //last else cond   
         if(abs(incoming) < peek) incoming will be destroyed and stack remain same //3 else if cond
         if(abs(incoming) == peek) both will be destroyed and stack need to pop out peek value //2 else if cond
        
         */
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < asteroid.length; i++) {

            if (stack.isEmpty() || asteroid[i] > 0) {
                stack.push(asteroid[i]);
            } else {

                while (true) {

                    int peek = stack.peek();
                    if (peek < 0) {
                        //peek is less that 0 means -ve(move to left)
                        //incoming aseroids[i] is either +ve or -ve
                        //they will never collide
                        //ex: stack[-4], incoming = -10 both move to left, left dir(no collision)
                        //ex: stack[-4], incoming = 10 both move to left, right dir(no collision)
                        stack.push(asteroid[i]);
                        break;
                    } else if (peek == -asteroid[i]) {
                        //stack[4], incoming = -4 both move to right, left fir(collision)
                        //magnitude of both are same so both will destroy 
                        //(incoming = -4 will not be pushed and 4 is also poped)
                        // If both are the same size, both will explode
                        stack.pop();
                        break;
                    } else if (peek > -asteroid[i]) {
                        //peek is greater means +ve(move to right dir)
                        //aseroids[i] is lesser -ve(move to left dir)
                        //If two asteroids meet, the smaller one will explode
                        break;
                    } else {
                        //case of stack[3,2,1], incoming is -10
                        // -10 will collide with all the asteroids and blow them up until the stack is 
                        //empty and when sack is empty put the -10 as the only asteroids
                        stack.pop();
                        if (stack.isEmpty()) {
                            stack.push(asteroid[i]);
                            break;
                        }
                    }
                }

            }
        }

        //output:
        int[] output = new int[stack.size()];
        int index = stack.size() - 1;
        while (!stack.isEmpty()) {
            output[index--] = stack.pop();
        }

        for (int x : output) {
            System.out.print(x + " ");
        }

        System.out.println();

    }

    public void jumpGame(int[] nums) {

        //https://leetcode.com/problems/jump-game/
        //Explanantion: https://youtu.be/Zb4eRjuPHbM
        //approach is to use reverse way assume that we can actually end of nums[]
        //now traverse backwards to check if we can reach 0 index (reachablePoint == 0)
        int reachablePoint = nums[nums.length - 1];
        for (int i = nums.length - 1; i >= 0; i--) {
            if (i + nums[i] >= reachablePoint) {
                reachablePoint = i;
            }
        }

        System.out.println("can we reach the end of nums array from 0th index: " + (reachablePoint == 0));
    }

    public void nextGreaterElement2_CyclicArray(int[] arr) {

        //explanation: https://leetcode.com/problems/next-greater-element-ii/solution/
        //array to be considered as cyclic
        int n = arr.length;
        int[] output = new int[n];
        Stack<Integer> stack = new Stack<>();
        for (int i = 2 * n - 1; i >= 0; i--) {

            while (!stack.isEmpty() && arr[stack.peek()] <= arr[i % n]) {
                stack.pop();
            }

            if (stack.isEmpty()) {
                output[i % n] = -1;
            } else {
                output[i % n] = arr[stack.peek()];
            }

            stack.push(i % n);
        }

        //output:
        for (int x : output) {
            System.out.print(x + " ");
        }

        System.out.println();

    }

    public void findMedianInDataStream(int[] stream) {

        //explanantion: https://leetcode.com/problems/find-median-from-data-stream/solution/ [HEAP BASED]
        PriorityQueue<Integer> minHeap = new PriorityQueue<>((a, b) -> a.compareTo(b));
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b.compareTo(a));

        for (int num : stream) {

            maxHeap.add(num);
            minHeap.add(maxHeap.poll());

            if (maxHeap.size() < minHeap.size()) {
                maxHeap.add(minHeap.poll());
            }

            double median = maxHeap.size() > minHeap.size()
                    ? maxHeap.peek()
                    : (double) (maxHeap.peek() + minHeap.peek()) * 0.5;

            System.out.println("Median: " + median);

        }
    }

    public void numPairsDivisibleBy60(int[] time) {

        //https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/
        int[] remainder = new int[60];
        int count = 0;
        for (int t : time) {
            if (t % 60 == 0) {
                count += remainder[0];
            } else {
                count += remainder[60 - t % 60];
            }
            remainder[t % 60]++;
        }

        //output:
        System.out.println("Total pair: " + count);
    }

    private int mergeSort(int[] arr, int start, int mid, int end) {

        int[] left = Arrays.copyOfRange(arr, start, mid + 1);
        int[] right = Arrays.copyOfRange(arr, mid + 1, end + 1);

        int i = 0;
        int j = 0;
        int k = start;
        int swaps = 0;
        while (i < left.length && j < right.length) {

            if (left[i] <= right[j]) {
                arr[k++] = left[i++];
            } else {
                arr[k++] = right[j++];
                swaps += (mid + 1) - (start + i);
            }
        }

        while (i < left.length) {
            arr[k++] = left[i++];
        }

        while (j < right.length) {
            arr[k++] = right[j++];
        }

        return swaps;
    }

    private int divideAndMerge(int[] arr, int start, int end) {
        int count = 0;
        if (end > start) {

            int mid = start + (end - start) / 2;
            count += divideAndMerge(arr, start, mid);
            count += divideAndMerge(arr, mid + 1, end);
            count += mergeSort(arr, start, mid, end);
        }
        return count;
    }

    public void countInversion(int[] arr) {

        //..........................T: O(n log n)
        //..........................S: O(N) temp(left, right arrays)
        //https://www.geeksforgeeks.org/counting-inversions/
        int n = arr.length;
        int countInversion = divideAndMerge(arr, 0, n - 1);
        System.out.println("Count inversion: " + countInversion);
        //array is also got sorted
//        for(int x: arr){
//            System.out.print(x+" ");
//        }
    }

    public void minimumWindowSubarrayForTargetSumK(int[] arr, int K) {

        //SLIDING WINDOW BASIC
        //explanation: https://youtu.be/jKF9AcyBZ6E
        int start = 0;
        int end = 0;
        int sum = 0;
        int win = Integer.MAX_VALUE;
        int index = 0;
        while (end < arr.length) {

            sum += arr[end];

            while (sum >= K) {
                win = Math.min(win, end - start + 1);
                index = start;
                sum -= arr[start];
                start++;
            }
            end++;
        }

        //output:
        System.out.println("Minimum length of subarrays whose sum (>= K): " + (win >= Integer.MAX_VALUE ? 0 : win));
        System.out.println("array element:");
        if (win != Integer.MAX_VALUE) {
            for (int i = 0; i < win; i++) {
                System.out.print(arr[index + i] + " ");
            }
        }
        System.out.println();
    }

    public void flipMZerosFindMaxLengthOfConsecutiveOnes(int[] arr, int M) {

        int start = 0;
        int end = 0;
        int zeroCount = 0;
        int bestWin = 0;
        int bestStart = 0;

        while (end < arr.length) {

            if (zeroCount <= M) {
                if (arr[end] == 0) {
                    zeroCount++;
                }
                end++;
            }

            if (zeroCount > M) {
                if (arr[start] == 0) {
                    zeroCount--;
                }
                start++;
            }

            if (end - start > bestWin && zeroCount <= M) {
                bestWin = end - start;
                bestStart = start;
            }

        }

        //output
        System.out.println("Length of consecutive ones after flipping M zeros: " + bestWin);
        System.out.println("Indexs of zeros to flip");
        for (int i = 0; i < bestWin; i++) {
            if (arr[bestStart + i] == 0) {
                System.out.print((bestStart + i) + " ");
            }
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

    private int areaPerRow(int[] hist) {

        //same as laregstAreaHistogram method
        Stack<Integer> stack = new Stack<>();
        int n = hist.length;
        int maxArea = 0;
        int top = 0;
        int areaWithTop = 0;
        int i = 0;
        while (i < n) {

            if (stack.isEmpty() || hist[stack.peek()] <= hist[i]) {
                stack.push(i++);
            } else {
                top = stack.pop();
                areaWithTop = hist[top] * (stack.isEmpty() ? i : i - stack.peek() - 1);
                maxArea = Math.max(maxArea, areaWithTop);
            }
        }

        while (!stack.isEmpty()) {
            top = stack.pop();
            areaWithTop = hist[top] * (stack.isEmpty() ? i : i - stack.peek() - 1);
            maxArea = Math.max(maxArea, areaWithTop);
        }

        return maxArea;

    }

    public void maxAreaOfRectangleInBinaryMatrix(int[][] mat) {

        //problem statment & sol: https://www.geeksforgeeks.org/maximum-size-rectangle-binary-sub-matrix-1s/
        //find max area of per row int the matrix
        //each row in the matrix is histogram
        //use max area histogram
        int R = mat.length;
        int C = mat[0].length;

        int maxArea = areaPerRow(mat[0]);

        for (int r = 1; r < R; r++) {
            for (int c = 0; c < C; c++) {
                if (mat[r][c] == 1) {
                    mat[r][c] += mat[r - 1][c];
                }
                maxArea = Math.max(maxArea, areaPerRow(mat[r]));
            }
        }

        //output:
        System.out.println("Max area in binary matrix: " + maxArea);

    }

    public void maximumOnesInRowOfABinarySortedMatrix_1(int[][] mat) {

        //.....................................T; O(M*N)
        //problem statement: https://www.geeksforgeeks.org/find-the-row-with-maximum-number-1s/
        int maxOnes = 0;
        int index = 0;
        for (int i = 0; i < mat.length; i++) {

            int onePerRow = areaPerRow(mat[i]);
            if (maxOnes < onePerRow) {
                maxOnes = onePerRow;
                index = i;
            }
        }

        //output;
        System.out.println("Max 1(s) found at index: " + (maxOnes == 0 ? -1 : index) + " counts of is are: " + maxOnes);

    }

    public void maximumOnesInRowOfABinarySortedMatrix_2(int[][] mat) {

        //............................T: O(M.LogN)
        //OPTIMISED
        int maxOnes = 0;
        int index = 0;
        for (int r = 0; r < mat.length; r++) {
            int C = mat[r].length;
            int firstIndexOfOne = findFirstOccurenceKInSortedArray(mat[r], 1, 0, C - 1, C);

            //if no index is found
            if (firstIndexOfOne == -1) {
                continue;
            }

            int onePerRow = C - firstIndexOfOne;
            if (maxOnes < onePerRow) {
                maxOnes = onePerRow;
                index = r;
            }
        }

        //output;
        System.out.println("Max 1(s) found at index: " + (maxOnes == 0 ? -1 : index) + " counts of is are: " + maxOnes);
    }

    public void findAValueInRowWiseSortedMatrix(int[][] mat, int K) {

        int M = mat.length;
        int N = mat[0].length;

        int i = 0;
        int j = N - 1;

        //search starts from top right corner
        while (i < M && j >= 0) {

            if (mat[i][j] == K) {
                System.out.println("Found at: " + i + ", " + j);
                return;
            } else if (K < mat[i][j]) {
                j--;
            } else {
                i++;
            }

        }

        //K is not there in the matrix at all
        System.out.println("Not found");

    }

    public void spiralMatrixTraversal(int[][] mat) {

        List<Integer> result = new ArrayList<>();

        int R = mat.length;
        int C = mat[0].length;

        int top = 0; // top row
        int bottom = R - 1; //bottom row
        int left = 0; //left col
        int right = C - 1; //right col
        int totalElement = R * C;

        //with each row and col processing we will shrink the matrix bounds (top++, bottom--, left++, right--)
        //result list is going to hold all the elements in the matrix
        while (result.size() < totalElement) {

            //get the top row from left to right
            for (int i = left; i <= right && result.size() < totalElement; i++) {
                result.add(mat[top][i]); //top row and ith col left to right
            }
            top++; //since we traversed top row now to next row next time.

            //top to bottom but right col
            for (int i = top; i <= bottom && result.size() < totalElement; i++) {
                result.add(mat[i][right]); //top to bottom but right col
            }
            right--; //now till here we have traversed the very right col of mat. for next time right col is prev

            //right to left but bottom row only
            for (int i = right; i >= left && result.size() < totalElement; i--) {
                result.add(mat[bottom][i]); //bottom row from right to left
            }
            bottom--; //completed bottom row also next time bottom to be prev one

            //bottom to top but only left col
            for (int i = bottom; i >= top && result.size() < totalElement; i--) {
                result.add(mat[i][left]); //left col
            }
            left++; //for next itr left will be moved ahead

        }

        //output:
        System.out.println("Spiral matrix: " + result);

    }

    public void diagonalMatrixTraversal(int[][] mat) {

        int R = mat.length;
        int C = mat[0].length;
        int x = 0;
        int y = 0;
        boolean isGoingUp = true;
        int totalElements = R * C;
        int element = 0;
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> diagonal = new ArrayList<>();
        while (element < totalElements) {

            if (isGoingUp) {

                while (x >= 0 && y < C) {
                    diagonal.add(mat[x][y]);
                    x--;
                    y++;
                    element++;
                }

                if (x < 0 && y <= C - 1) {
                    x = 0;
                }

                if (y == C) {
                    x += 2;
                    y--;
                }
            } else {

                while (x < R && y >= 0) {
                    diagonal.add(mat[x][y]);
                    x++;
                    y--;
                    element++;
                }

                if (x <= R - 1 && y < 0) {
                    y = 0;
                }

                if (x == R) {
                    y += 2;
                    x--;
                }
            }

            isGoingUp = !isGoingUp;
            result.add(diagonal);
            diagonal = new ArrayList<>();
        }

        //output:
        System.out.println("Diagonal matrix: " + result);

    }

    public void sumOfElementsInMatrixExceptGivenRowAndCol(int[][] matrix, int[][] rowAndCol) {

        //OPTIMISED
        //..........................T: O((R * C) + n) where n = rowAndCol.length
        //..........................S: O(R + C) because rowSum and colSum array of size R & C respec
        //https://www.geeksforgeeks.org/find-sum-of-all-elements-in-a-matrix-except-the-elements-in-given-row-andor-column-2/
        int R = matrix.length;
        int C = matrix[0].length;

        int[] rowSum = new int[R];
        int[] colSum = new int[C];

        int sumOfElements = 0;

        for (int x = 0; x < R; x++) {
            for (int y = 0; y < C; y++) {
                sumOfElements += matrix[x][y]; //total sum of elements of matrix
                rowSum[x] += matrix[x][y]; //all the sum of elements in x row
                colSum[y] += matrix[x][y]; //all the sum of elements in y col
            }
        }

        for (int[] except : rowAndCol) {

            int row = except[0];
            int col = except[1];
            int sumWithoutRowAndCol = sumOfElements - rowSum[row] - colSum[col] + matrix[row][col];
            System.out.println("Sum of elements except row: " + row + " & col: " + col + " sum: " + sumWithoutRowAndCol);

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

        int n = s.length();
        int[] lps = new int[n];
        int j = 0, i = 1;
        while (i < n) {
            if (s.charAt(j) == s.charAt(i)) {
                j++;
                lps[i] = j;
                i++;
            } else {
                if (j == 0) {
                    i++;
                } else {
                    j = lps[j - 1];
                }
            }
        }

        System.out.println("Length of longest prefix: " + (j == 0 ? j : j));
        System.out.println("Longest prefix substring: " + (j == 0 ? "" : s.substring(0, j)));
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

    public void secondMostOccuringWordInStringList(String[] list) {

        Map<String, Integer> map = new HashMap<>();
        for (String s : list) {
            map.put(s, map.getOrDefault(s, 0) + 1);
        }

        PriorityQueue<Map.Entry<String, Integer>> minHeap = new PriorityQueue<>(
                (e1, e2) -> e1.getValue() - e2.getValue()
        );

        for (Map.Entry<String, Integer> e : map.entrySet()) {
            minHeap.add(e);
            if (minHeap.size() > 2) {
                minHeap.poll();
            }
        }

        System.out.println("Second most occuring word: " + minHeap.poll().getKey());

    }

    public boolean checkIsomorphicStrings_1(String s1, String s2) {

        int m = s1.length();
        int n = s2.length();

        if (m != n) {
            System.out.println("Not isomorphic strings");
            return false;
        }

        int SIZE = 256; //to handle numric & alphabetic ascii ranges
        boolean[] marked = new boolean[SIZE];
        int[] map = new int[SIZE];
        Arrays.fill(map, -1);

        for (int i = 0; i < m; i++) {
            if (map[s1.charAt(i)] == -1) {

                if (marked[s2.charAt(i)] == true) {
                    return false;
                }

                marked[s2.charAt(i)] = true;
                map[s1.charAt(i)] = s2.charAt(i);

            } else if (map[s1.charAt(i)] != (s2.charAt(i))) {
                return false;
            }
        }

        return true;

    }

    public boolean checkIsomorphicStrings_2(String s1, String s2) {

        //........................T: O(N)
        //EASIER EXPLAINATION
        int m = s1.length();
        int n = s2.length();

        if (m != n) {
            System.out.println("Not isomorphic strings");
            return false;
        }

        Map<Character, Character> map = new HashMap<>();
        for (int i = 0; i < m; i++) {
            char sChar = s1.charAt(i);
            char tChar = s2.charAt(i);

            if (map.containsKey(sChar) && map.get(sChar) != tChar) {
                return false;
            }

            map.put(sChar, tChar);

        }

        map.clear();

        for (int i = 0; i < m; i++) {
            char sChar = s1.charAt(i);
            char tChar = s2.charAt(i);

            if (map.containsKey(tChar) && map.get(tChar) != sChar) {
                return false;
            }

            map.put(tChar, sChar);

        }

        return true;

    }

    public int transformOneStringToAnotherWithMinOprn(String src, String target) {

        //count if two strings are same length or not
        int m = src.length();
        int n = target.length();

        if (m != n) {
            //if length are not same, strings can't be transformed
            return -1;
        }

        //check if the two strings contain same char and their count should also be same
        int[] charCount = new int[256];
        for (int i = 0; i < m; i++) {

            charCount[src.charAt(i)]++;
            charCount[target.charAt(i)]--;

        }

        //if same char are there and count are equal then charCount shoul have been balanced out to 0
        for (int count : charCount) {
            if (count != 0) {
                return -1;
            }
        }

        int i = m - 1;
        int j = n - 1;
        int result = 0;
        while (i >= 0) {

            if (src.charAt(i) != target.charAt(j)) {
                result++;
            } else {
                j--;
            }

            i--;

        }

        return result;
    }

    public void arrangeAllWordsAsTheirAnagrams(List<String> words) {

        Map<String, List<String>> anagramGroups = new HashMap<>();
        for (String str : words) {

            char[] ch = str.toCharArray();
            Arrays.sort(ch);
            String sortedString = String.valueOf(ch);

            anagramGroups.putIfAbsent(sortedString, new ArrayList<>());
            anagramGroups.get(sortedString).add(str);

        }

        //output:
        System.out.println("Output: " + anagramGroups);

    }

    public void characterAddedAtFrontToMakeStringPallindrome_1(String str) {

        //https://www.geeksforgeeks.org/minimum-characters-added-front-make-string-palindrome/
        int charCount = 0;
        while (str.length() > 0) {

            if (isStringPallindrome(str)) {
                break;
            } else {
                charCount++;
                //removing 1 char from end until we get a subtring which is pallindrome
                //the no of char removed (charCount) is the number that needs to be added at front
                str = str.substring(0, str.length() - 1);

            }

        }

        //output:
        System.out.println("No. of character to be added at front to make it pallindrome: " + charCount);
    }

    public void characterAddedAtFrontToMakeStringPallindrome_2(String str) {

        //https://www.geeksforgeeks.org/minimum-characters-added-front-make-string-palindrome/
        StringBuilder s = new StringBuilder();
        s.append(str);

        // Get concatenation of string, special character  
        // and reverse string  
        String rev = s.reverse().toString();
        s.reverse().append("$").append(rev);

        // Get LPS array of this concatenated string 
        int lps[] = KMP_PatternMatching_Algorithm_LPSArray(s.toString(), s.toString().length());

        //output:
        System.out.println("No. of character to be added at front to make it pallindrome KMP based: " + (str.length() - lps[s.length() - 1]));
    }

    public boolean checkIfOneStringRotationOfOtherString(String str1, String str2) {
        return (str1.length() == str2.length())
                && ((str1 + str1).indexOf(str2) != -1);
    }

    private int countOccurenceOfGivenStringInCharArray_Count = 0;

    private void countOccurenceOfGivenStringInCharArray_Helper(char[][] charArr, int x, int y,
            int startPoint, String str, StringBuilder sb) {

        if (sb.toString().equals(str)) {
            //once set of string is found reset stringbuilder
            sb.setLength(0);
            countOccurenceOfGivenStringInCharArray_Count++;
            return;
        }

        if (x < 0 || x >= charArr.length || y < 0 || y >= charArr[0].length
                || startPoint >= str.length()
                || charArr[x][y] != str.charAt(startPoint)
                || charArr[x][y] == '-') {
            return;
        }

        char original = charArr[x][y];
        sb.append(charArr[x][y]);
        charArr[x][y] = '-';

        //UP
        countOccurenceOfGivenStringInCharArray_Helper(charArr, x - 1, y, startPoint + 1, str, sb);

        //Down
        countOccurenceOfGivenStringInCharArray_Helper(charArr, x + 1, y, startPoint + 1, str, sb);

        //Left
        countOccurenceOfGivenStringInCharArray_Helper(charArr, x, y - 1, startPoint + 1, str, sb);

        //Right
        countOccurenceOfGivenStringInCharArray_Helper(charArr, x, y + 1, startPoint + 1, str, sb);

        charArr[x][y] = original;

    }

    public void countOccurenceOfGivenStringInCharArray(char[][] charArr, String str) {

        countOccurenceOfGivenStringInCharArray_Count = 0; //reset/init
        StringBuilder sb = new StringBuilder();
        int N = charArr.length;
        int startPoint = 0;
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
                countOccurenceOfGivenStringInCharArray_Helper(charArr, x, y, startPoint, str, sb);
            }
        }

        //output
        System.out.println("Count of the given string is: " + countOccurenceOfGivenStringInCharArray_Count);

    }

    private void printAllSubSequencesOfAString_Helper(String str, int start, int N,
            String current, Set<String> subseq) {

        if (start == N) {
            subseq.add(current);
            return;
        }

        for (int i = start; i < N; i++) {
            printAllSubSequencesOfAString_Helper(str, i + 1, N, current + str.charAt(i), subseq);
            printAllSubSequencesOfAString_Helper(str, i + 1, N, current, subseq);
        }

    }

    public void printAllSubSequencesOfAString(String str) {

        int N = str.length();
        Set<String> subseq = new HashSet<>();
        printAllSubSequencesOfAString_Helper(str, 0, N, "", subseq);

        //output:
        System.out.println("All possible subsequences of string: " + subseq);
    }

    public boolean balancedParenthesisEvaluation(String s) {

        Stack<Character> stack = new Stack<>();
        for (char ch : s.toCharArray()) {

            if (ch == '{' || ch == '[' || ch == '(') {
                stack.push(ch);
                continue;
            } else if (!stack.isEmpty() && stack.peek() == '(' && ch == ')') {
                stack.pop();
            } else if (!stack.isEmpty() && stack.peek() == '{' && ch == '}') {
                stack.pop();
            } else if (!stack.isEmpty() && stack.peek() == '[' && ch == ']') {
                stack.pop();
            } else {
                return false;
            }
        }

        return stack.isEmpty();

    }

    public void firstNonRepeatingCharacterFromStream(String stream) {

        int MAX_CHAR = 256;
        List<Character> list = new ArrayList<>();
        boolean[] vis = new boolean[MAX_CHAR];
        for (int i = 0; i < stream.length(); i++) {
            char ch = stream.charAt(i);
            if (vis[ch] != true) {

                if (list.contains(ch)) {
                    list.remove((Character) ch);
                    vis[ch] = true;
                } else {
                    list.add(ch);
                }

            }

            System.out.println("First non repeating character till " + stream.substring(0, i + 1));
            if (list.size() != 0) {
                System.out.println(list.get(0) + " ");
            }

        }

    }

    public boolean wordBreak_Recursive(String str, Set<String> set) {

        //https://www.geeksforgeeks.org/word-break-problem-dp-32/
        int n = str.length();
        if (n == 0) {
            return true;
        }

        if (set.contains(str)) {
            return true;
        }

        for (int i = 1; i <= n; i++) {
            if (set.contains(str.substring(0, i)) && wordBreak_Recursive(str.substring(i, n), set)) {
                return true;
            }
        }
        return false;
    }

    public boolean wordBreak_DP_Problem(String str, Set<String> set) {

        //https://leetcode.com/problems/word-break/discuss/1068441/Detailed-Explanation-of-Top-Down-and-Bottom-Up-DP
        boolean[] memo = new boolean[str.length() + 1];
        //base cond
        memo[0] = true; // str with no length is also true

        for (int i = 1; i <= str.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (set.contains(str.substring(j, i)) && memo[j]) {
                    memo[i] = true;
                    break;
                }
            }
        }

        return memo[str.length()];

    }

    public void longestSubstringWithoutRepeatingChar(String str) {

        //SLIDING WINDOW ALGO
        //https://leetcode.com/problems/longest-substring-without-repeating-characters/
        int n = str.length();
        int i = 0;
        int j = 0;
        int maxLen = 0;
        int[] count = new int[256];
        String substr = "";
        while (j < n) {

            char ch = str.charAt(j);
            if (count[ch] < 1) {
                count[ch]++;
                maxLen = Math.max(maxLen, j - i + 1);
                //the new subtring len is greater than prev substring len update that substring
                substr = str.substring(i, j + 1).length() > substr.length() ? str.substring(i, j + 1) : substr;
                j++;
            } else {
                count[str.charAt(i)]--;
                i++;
            }

        }

        //output:
        System.out.println("Longest subtring without repeating char: " + maxLen + " (" + substr + ")");

    }

    public void minimumWindowSubstring(String s, String t) {

        //Explanation: https://www.youtube.com/watch?v=nMaKzLWceFg&feature=youtu.be
        //SLIDING WINDOW ALGO
        //prepare the count map for string t 
        //to know how many char we need to find in string s
        Map<Character, Integer> tMap = new HashMap<>();
        for (char ch : t.toCharArray()) {
            tMap.put(ch, tMap.getOrDefault(ch, 0) + 1);
        }

        int tCount = 0;
        int start = 0;
        int end = 0;
        int minLenWindow = Integer.MAX_VALUE;
        int substrIndex = 0;
        int N = s.length();
        while (end < N) {

            char chEnd = s.charAt(end);
            //each we find the char in string s that is also in string t
            //we decreament the count of that char from map
            //to know how much of that char (chEnd) we needed and how much we have found
            if (tMap.containsKey(chEnd)) {
                tMap.put(chEnd, tMap.get(chEnd) - 1);
                if (tMap.get(chEnd) >= 0) {
                    tCount++;
                }
            }

            while (tCount == t.length()) {

                if (minLenWindow > (end - start + 1)) {
                    minLenWindow = end - start + 1;
                    substrIndex = start;
                }

                //adjust the start pointer now
                char chStart = s.charAt(start);
                if (tMap.containsKey(chStart)) {
                    tMap.put(chStart, tMap.get(chStart) + 1);
                    if (tMap.get(chStart) > 0) {
                        tCount--;
                    }
                }
                start++;
            }

            end++;
        }

        //output:
        String output = minLenWindow > s.length() ? "" : s.substring(substrIndex, substrIndex + minLenWindow);
        System.out.println("Min window substring containg all char of string t in string s: "
                + (minLenWindow > s.length() ? -1 : minLenWindow) + " : "
                + output);

    }

    public void countAllOccurencesOfPatternInGivenString(String txt, String pat) {

        //........................T: O(N)
        //SLIDING WINDOW
        Map<Character, Integer> patMap = new HashMap<>();
        for (char chPat : pat.toCharArray()) {
            patMap.put(chPat, patMap.getOrDefault(chPat, 0) + 1);
        }

        int totalOccurences = 0;
        int patCharCount = 0;
        int patLen = pat.length();
        int txtLen = txt.length();
        int start = 0;
        int end = 0;
        List<Integer> occuerencesIndex = new ArrayList<>();
        List<String> occuerencesSubstring = new ArrayList<>();
        while (end < txtLen) {

            char chEnd = txt.charAt(end);
            if (patMap.containsKey(chEnd)) {
                patMap.put(chEnd, patMap.get(chEnd) - 1);
                if (patMap.get(chEnd) >= 0) {
                    patCharCount++;
                }
            }

            //once window for pattern is reached
            //balance out by moving this window
            //and removing start from patCharCount
            while ((end - start + 1) == patLen) {

                if (patCharCount == patLen) {
                    totalOccurences++;
                    occuerencesIndex.add(start);
                    occuerencesSubstring.add(txt.substring(start, start + patLen));
                }

                char chStart = txt.charAt(start);
                if (patMap.containsKey(chStart)) {
                    patMap.put(chStart, patMap.get(chStart) + 1);
                    if (patMap.get(chStart) > 0) {
                        patCharCount--;
                    }
                }
                start++;
            }
            end++;
        }

        //output:
        System.out.println("Total occurences: " + totalOccurences);
        System.out.println("All the indexes: " + occuerencesIndex);
        System.out.println("All the substring of anagrams: " + occuerencesSubstring);
    }

    public void partitionLabels(String str) {

        //problem: https://leetcode.com/problems/partition-labels/
        //explanation: https://youtu.be/5NCjHqx2v-k
        List<Integer> result = new ArrayList<>();
        int n = str.length();
        int[] lastIndexes = new int[26];
        for (int i = 0; i < n; i++) {
            lastIndexes[str.charAt(i) - 'a'] = i;
        }

        int start = 0;
        int end = 0;
        for (int i = 0; i < n; i++) {
            end = Math.max(end, lastIndexes[str.charAt(i) - 'a']);
            if (i == end) {
                result.add(end - start + 1);
                start = end + 1;
            }
        }

        //output:
        System.out.println("All the partition lengths: " + result);

    }

    public void longestRepeatingCharacterByKReplacement(String str, int K) {

        //explanation: https://youtu.be/00FmUN1pkGE
        //SLIDING WINDOW
        int[] charCount = new int[26];
        int start = 0;
        int end = 0;
        int maxLen = 0;
        int maxCount = 0;
        int n = str.length();
        while (end < n) {

            char chEnd = str.charAt(end);
            charCount[chEnd - 'A']++;
            maxCount = Math.max(maxCount, charCount[chEnd - 'A']);

            while ((end - start + 1) - maxCount > K) {
                charCount[str.charAt(start) - 'A']--;
                start++;
            }

            maxLen = Math.max(maxLen, end - start + 1);
            end++;
        }

        //output:
        System.out.println("Max length: " + maxLen);

    }

    private void generateBalancedParenthesis_Helper(int n, String curr, int open, int close, List<String> result) {

        //any req parenthesis of size n needs 2 * n parenthesis to balance
        //ex: if n = 1 => "{" to balance it need "}" which "{}" is n * 2
        if (curr.length() == 2 * n) {
            result.add(curr);
            return;
        }

        if (open < n) {
            generateBalancedParenthesis_Helper(n, curr + "{", open + 1, close, result);
        }

        if (open > close) {
            generateBalancedParenthesis_Helper(n, curr + "}", open, close + 1, result);
        }
    }

    public void generateBalancedParenthesis(int n) {

        //https://www.geeksforgeeks.org/print-all-combinations-of-balanced-parentheses/
        //https://leetcode.com/problems/generate-parentheses/solution/
        List<String> result = new ArrayList<>();
        generateBalancedParenthesis_Helper(n, "", 0, 0, result);

        //output:
        System.out.println("All balanced parenthesis: " + result);

    }

    public void scoreOfParenthesis(String str) {

        //https://leetcode.com/problems/score-of-parentheses
        //explanatin: https://youtu.be/jfmJusJ0qKM
        int score = 0;
        Stack<Integer> stack = new Stack<>();
        for (char ch : str.toCharArray()) {

            if (ch == '(') {
                stack.push(score);
                score = 0;
            } else {
                score = stack.pop() + Math.max(score * 2, 1);
            }
        }

        //output:
        System.out.println("Score: " + score);
    }

    public void minimumCharRemovalToMakeValidParenthesis(String str) {

        //https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses
        class CharIndex {

            char ch;
            int index;

            public CharIndex(char ch, int index) {
                this.ch = ch;
                this.index = index;
            }
        }

        StringBuilder sb = new StringBuilder();
        Stack<CharIndex> stack = new Stack<>();
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            sb.append(c);
            if (c == '(' || c == ')') {

                if (!stack.isEmpty() && stack.peek().ch == '(' && c == ')') {
                    stack.pop();
                    continue;
                }
                stack.push(new CharIndex(c, i));
            }
        }

        while (!stack.isEmpty()) {
            sb.deleteCharAt(stack.pop().index);
        }

        //output:
        System.out.println("Balanced string: " + sb.toString());

    }

    public boolean repeatedSubstringPattern(String str) {

        //https://leetcode.com/problems/repeated-substring-pattern/
        //explanantion: https://youtu.be/bClIZj66dVE
        int len = str.length();
        for (int i = len / 2; i >= 1; i--) {

            if (len % i == 0) {

                int numRepeats = len / i;
                String sub = str.substring(0, i);
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < numRepeats; j++) {
                    sb.append(sub);
                }

                if (sb.toString().equals(str)) {
                    return true;
                }
            }
        }

        return false;
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

    private int longestPallindromicSubstring_ExpandFromMiddle(String str, int left, int right) {

        if (str == null || left > right) {
            return 0;
        }

        while (left >= 0 && right < str.length() && str.charAt(left) == str.charAt(right)) {
            left--;
            right++;
        }

        return right - left - 1;

    }

    public void longestPallindromicSubstring(String str) {

        //.....................T: O(N^2)
        //https://leetcode.com/problems/longest-palindromic-substring/
        //explanation: https://youtu.be/y2BD4MJqV20
        int start = 0;
        int end = 0;

        for (int i = 0; i < str.length(); i++) {

            //case to handle odd length string 
            //there will be exactly middle char in that
            //ex: "racecar" middle char is 'e' 
            int len1 = longestPallindromicSubstring_ExpandFromMiddle(str, i, i);
            //case to handle even length string
            //the middle will in b/w the two char of (str.length / 2)  and ((str.length /2) + 1)
            //ex: "aabbaa" middle char will be in b/w b|b
            int len2 = longestPallindromicSubstring_ExpandFromMiddle(str, i, i + 1);

            int len = Math.max(len1, len2);

            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + (len) / 2;
            }

        }

        //output:
        System.out.println("Longest pallindromic substring: " + str.substring(start, end + 1));

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

        //............................T: O(N*K*LogK)
        //............................S: O(K)
        //https://www.geeksforgeeks.org/merge-k-sorted-linked-lists-set-2-using-min-heap/
        //HEAP based method
        PriorityQueue<Node<Integer>> minHeap = new PriorityQueue<>(
                (o1, o2) -> o1.getData().compareTo(o2.getData())
        );

        //at this point we have added K node heads in minHeap
        //so the minheap size after this loop will K
        for (Node<Integer> node : nodes) {
            if (node != null) {
                minHeap.add(node);
            }
        }

        //head to point arbitary infinite value to start with
        Node<Integer> head = new Node<>(Integer.MIN_VALUE);
        //saving the actual head's ref
        Node<Integer> copyHead = head;
        while (!minHeap.isEmpty()) {

            //we poll out one node from heap
            Node<Integer> curr = minHeap.poll();
            //and add in one node 
            //if its next node is not null
            //so the size of the min heap remains atmost K
            if (curr.getNext() != null) {
                minHeap.add(curr.getNext());
            }
            copyHead.setNext(curr);
            copyHead = copyHead.getNext();

        }

        copyHead.setNext(null);

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
        //https://leetcode.com/problems/swap-nodes-in-pairs/
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

    public Node<Integer> reverseLinkedListInKGroupsAlternatively(Node<Integer> head, int K) {

        //Adaptation of reverseLinkedListInKGroups method
        //reverse first K groups
        Node<Integer> curr = head;
        Node<Integer> prev = null;
        Node<Integer> next = null;

        int count = 0;

        while (count < K && curr != null) {

            next = curr.getNext();
            curr.setNext(prev);
            prev = curr;
            curr = next;

            count++;
        }

        //move straight K nodes without reversing
        count = 0;
        Node<Integer> temp = next;
        while (count < K && temp != null) {

            head.setNext(temp);
            head = head.getNext();
            temp = temp.getNext();
            count++;
        }

        //pass the alternate K group recursively
        if (temp != null) {
            head.setNext(reverseLinkedListInKGroupsAlternatively(temp, K));
        }

        return prev;

    }

    public boolean detectLoopCycleInLinkedList_HashBased(Node node) {

        //......................T: O(N)
        //......................S: O(N)
        Set<Node> set = new HashSet<>();
        Node temp = node;
        while (temp != null) {

            if (set.contains(temp)) {
                System.out.println("Hash Based Cycle at: " + temp.getData());
                return true;
            }

            set.add(temp);
            temp = temp.getNext();
        }

        System.out.println("Hash Based No cycle found");
        return false;

    }

    public boolean detectLoopCycleInLinkedList_Iterative(Node node) {

        //......................T: O(N)
        //......................S: O(1)
        Node slow = node;
        Node fast = node.getNext();
        while (slow != null && fast != null && fast.getNext() != null) {

            if (slow == fast) {
                break;
            }

            slow = slow.getNext();
            fast = fast.getNext().getNext();

        }

        if (slow == fast) {
            slow = node;
            while (slow != fast.getNext()) {
                slow = slow.getNext();
                fast = fast.getNext();
            }

            //fast.next is where the loop starts...
            System.out.println("Iterative approach Cycle at: " + fast.getNext().getData());
            return true;
        }

        System.out.println("Iterative approach No cycle found");
        return false;

    }

    public void detectAndRemoveLoopCycleInLinkedList_HashBased(Node node) {

        //......................T: O(N)
        //......................S: O(N)
        Set<Node> set = new HashSet<>();
        Node loopEnd = null;
        Node temp = node;
        while (temp != null) {

            if (set.contains(temp)) {
                loopEnd.setNext(null);
                break;
            }

            set.add(temp);
            loopEnd = temp;
            temp = temp.getNext();

        }

        //output;
        System.out.println("Hash Based approach detect and remove a loop cycle in linked list output:");
        new LinkedListUtil(node).print();

    }

    public void detectAndRemoveLoopCycleInLinkedList_Iterative(Node node) {

        //......................T: O(N)
        //......................S: O(1)
        Node slow = node;
        Node fast = node.getNext();
        while (slow != null && fast != null && fast.getNext() != null) {

            if (slow == fast) {
                break;
            }

            slow = slow.getNext();
            fast = fast.getNext().getNext();

        }

        //if there is a loop in linked list
        if (slow == fast) {

            slow = node;
            while (slow != fast.getNext()) {
                slow = slow.getNext();
                fast = fast.getNext();
            }

            //fast is the node where it should end the loop
            fast.setNext(null);

        }

        //output
        System.out.println("Iterative approach detect and remove a loop cycle in linked list output:");
        new LinkedListUtil(node).print();

    }

    public void removeDuplicatesFromUnSortedLinkedListOnlyConsecutive(Node<Integer> node) {

        //...............................T: O(N)
        //...............................S: O(1)
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
        System.out.println("Remove duplicates that are consecutive in lisnked list output:");
        new LinkedListUtil<>(node).print();

    }

    public void removeDuplicatesFromUnSortedLinkedListAllExtraOccuernce(Node<Integer> node) {

        //...............................T: O(N)
        //...............................S: O(N)
        Set<Integer> set = new HashSet<>();
        Node<Integer> curr = node;
        Node<Integer> temp = node.getNext();
        set.add(curr.getData());
        while (temp != null) {

            if (curr.getData() != temp.getData() && !set.contains(temp.getData())) {
                curr.setNext(temp);
                curr = temp;
            }
            set.add(temp.getData());
            temp = temp.getNext();

        }

        curr.setNext(temp);

        //output
        System.out.println("Remove duplicates all extra occuernce in lisnked list output:");
        new LinkedListUtil<>(node).print();

    }

    public Node<Integer> findMiddleNodeOfLinkedList(Node<Integer> node) {

        if (node == null || node.getNext() == null) {
            return node;
        }

        Node<Integer> slow = node;
        Node<Integer> fast = node.getNext();
        while (fast != null && fast.getNext() != null) {

            slow = slow.getNext();
            fast = fast.getNext().getNext();

        }

        //middle node = slow
        return slow;

    }

    public Node<Integer> mergeSortInLinkedList_Asc_Recursion(Node<Integer> n1, Node<Integer> n2) {

        if (n1 == null) {
            return n2;
        }

        if (n2 == null) {
            return n1;
        }

        if (n1.getData() <= n2.getData()) {
            Node<Integer> a = mergeSortInLinkedList_Asc_Recursion(n1.getNext(), n2);
            n1.setNext(a);
            return n1;
        } else {
            Node<Integer> b = mergeSortInLinkedList_Asc_Recursion(n1, n2.getNext());
            n2.setNext(b);
            return n2;
        }

    }

    public Node<Integer> mergeSortDivideAndMerge(Node<Integer> node) {

        if (node == null || node.getNext() == null) {
            return node;
        }

        Node<Integer> middle = findMiddleNodeOfLinkedList(node);
        Node<Integer> secondHalf = middle.getNext();
        //from node to middle is first half, so middle.next = null 
        //splites as 1. node -> middle.next->NULL 2. middle.next -> tail.next->NULL
        middle.setNext(null);

        return mergeSortInLinkedList_Asc_Recursion(mergeSortDivideAndMerge(node),
                mergeSortDivideAndMerge(secondHalf));
    }

    public boolean checkIfLinkedListIsCircularLinkedList(Node node) {

        if (node == null || node.getNext() == node) {
            return true;
        }

        Node headRef = node;
        Node temp = node;
        while (temp.getNext() != headRef && temp.getNext() != null) {
            temp = temp.getNext();
        }
        return temp.getNext() == headRef;
    }

    private int quickSortInLinkedList_Partition(List<Integer> arr, int low, int high) {

        int pivot = arr.get(high);
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr.get(j) < pivot) {

                i++;
                //swap
                int temp = arr.get(i);
                arr.set(i, arr.get(j));
                arr.set(j, temp);

            }
        }

        int temp = arr.get(i + 1);
        arr.set(i + 1, arr.get(high));
        arr.set(high, temp);

        return i + 1;

    }

    private void quickSortInLinkedList_Helper(List<Integer> arr, int low, int high) {

        if (high >= low) {

            int pivotIndex = quickSortInLinkedList_Partition(arr, low, high);
            quickSortInLinkedList_Helper(arr, low, pivotIndex - 1);
            quickSortInLinkedList_Helper(arr, pivotIndex + 1, high);
        }

    }

    public void quickSortInLinkedList(Node<Integer> node) {

        if (node == null || node.getNext() == null) {
            new LinkedListUtil<Integer>(node).print();
            return;
        }

        List<Integer> intArr = new ArrayList<>();
        Node<Integer> temp = node;
        while (temp != null) {
            intArr.add(temp.getData());
            temp = temp.getNext();
        }

        quickSortInLinkedList_Helper(intArr, 0, intArr.size() - 1);

        // System.out.println(intArr);
        temp = node;
        for (int x : intArr) {

            temp.setData(x);
            temp = temp.getNext();

        }

        //output
        new LinkedListUtil<Integer>(node).print();

    }

    public void moveLastNodeToFrontOfLinkedList(Node<Integer> node) {

        Node curr = node;
        Node prev = null;

        while (curr.getNext() != null) {
            prev = curr;
            curr = curr.getNext();
        }

        prev.setNext(curr.getNext());
        curr.setNext(node);
        node = curr;

        //output:
        new LinkedListUtil<Integer>(node).print();

    }

    private int addOneToLinkedList_Helper(Node<Integer> node) {

        if (node.getNext() == null) {
            int sum = node.getData() + 1; //adding 1 to very last node(or last digit of number in linkedlist form)
            node.setData(sum % 10);
            return sum / 10;
        }

        int carry = addOneToLinkedList_Helper(node.getNext());
        int sum = carry + node.getData();
        node.setData(sum % 10);
        return sum / 10;

    }

    public void addOneToLinkedList(Node<Integer> head) {

        if (head == null) {
            return;
        }

        int carry = addOneToLinkedList_Helper(head);
        //edge case for L [9 -> 9 -> 9 -> NULL] + 1 = [1 -> 0 -> 0 -> 0 -> NULL]
        //extra 1 is the newHead in this case...
        if (carry > 0) {
            Node<Integer> newHead = new Node<>(carry);
            newHead.setNext(head);
            head = newHead;
        }

        //output
        new LinkedListUtil<Integer>(head).print();

    }

    public void sortLinkedListOf012_2(Node<Integer> node) {

        //approach 1 is just using merger sort on linked list. 
        //merge sort method is already been implemented
        //approach 2 will be similar to my approach of solving
        //sortArrayOf012_1()
        int[] count = new int[3]; //we just have 3 digits (0, 1, 2)
        Node<Integer> curr = node;
        while (curr != null) {
            count[curr.getData()]++;
            curr = curr.getNext();
        }

        //manipulate the linked list 
        curr = node;
        for (int i = 0; i < 3; i++) { //O(3)

            while (count[i]-- != 0) {
                //O(N) as N = count[0]+count[1]+cout[3] == total no of node already in the linked list
                curr.setData(i);
                curr = curr.getNext();
            }

        }

        //output:
        new LinkedListUtil<Integer>(node).print();

    }

    public void reverseDoublyLinkedList(Node node) {

        //actual
        new LinkedListUtil(node).print();

        Node curr = node;
        Node nextToCurr = null;
        Node prevToCurr = null;
        while (curr != null) {

            nextToCurr = curr.getNext();
            curr.setNext(prevToCurr);
            curr.setPrevious(nextToCurr);
            prevToCurr = curr;
            curr = nextToCurr;

        }

        //output:
        //new head will pre prevToCurr
        new LinkedListUtil(prevToCurr).print();

    }

    public void intersectionOfTwoSortedLinkedList(Node<Integer> node1, Node<Integer> node2) {

        //....................T: O(M+N)
        //....................S: O(M+N)
        Set<Integer> node1Set = new HashSet<>();
        while (node1 != null) {
            node1Set.add(node1.getData());
            node1 = node1.getNext();
        }

        Set<Integer> node2Set = new HashSet<>();
        Node<Integer> newHead = new Node<>(Integer.MIN_VALUE);
        Node<Integer> copy = newHead;
        while (node2 != null) {

            //all the in node2 that is present in node1 set but same node2 should not be repested in node2 set
            if (node1Set.contains(node2.getData()) && !node2Set.contains(node2.getData())) {
                copy.setNext(new Node<>(node2.getData()));
                copy = copy.getNext();
            }
            node2Set.add(node2.getData());
            node2 = node2.getNext();
        }

        //output:
        new LinkedListUtil<Integer>(newHead.getNext()).print();

    }

    private int lengthOfLinkedList(Node<Integer> node) {
        int len = 0;
        Node<Integer> curr = node;
        while (curr != null) {
            len++;
            curr = curr.getNext();
        }

        return len;
    }

    private Node<Integer> moveLinkedListNodeByDiff(Node<Integer> node, int diff) {

        int index = 0;
        Node<Integer> curr = node;
        while (index++ < diff) { //evaluates as index++ -> 0+1 -> 1 then 1 < diff
            curr = curr.getNext();
        }
        return curr;
    }

    private int intersectionPointOfTwoLinkedListByRef_Helper(Node<Integer> n1, Node<Integer> n2) {

        Node<Integer> currN1 = n1;
        Node<Integer> currN2 = n2;
        while (currN1 != null && currN2 != null) {

            //nodes get common by ref
            if (currN1 == currN2) {
                return currN1.getData();
            }

            currN1 = currN1.getNext();
            currN2 = currN2.getNext();

        }

        return -1;
    }

    public void intersectionPointOfTwoLinkedListByRef(Node<Integer> node1, Node<Integer> node2) {

        //find length of node1 T: O(M)
        int M = lengthOfLinkedList(node1);
        //find length of node2 T: O(N)
        int N = lengthOfLinkedList(node2);

        //find the absolute diff in both the length
        //diff = abs(M - N)
        int diff = Math.abs(M - N);

        //if M > N move ptr in node1 by diff forward else move ptr in node2
        //once ptr is available move ptr and node1 or node2 till null and find the intersection point
        //by ref
        int intersectedData = -1;
        Node<Integer> curr = null;
        if (M > N) {
            curr = moveLinkedListNodeByDiff(node1, diff);
            intersectedData = intersectionPointOfTwoLinkedListByRef_Helper(curr, node2);
        } else {
            curr = moveLinkedListNodeByDiff(node2, diff);
            intersectedData = intersectionPointOfTwoLinkedListByRef_Helper(curr, node1);
        }

        //output:
        System.out.println("Two linked list are intersected at: " + intersectedData);

    }

    public void intersectionPointOfTwoLinkedListByRef_HashBased(Node<Integer> node1, Node<Integer> node2) {

        //................................T: O(N)
        //................................S: O(N)
        Set<Node<Integer>> set1 = new HashSet<>();
        Node<Integer> curr = node1;
        while (curr != null) {
            set1.add(curr);
            curr = curr.getNext();
        }

        int intersectedData = -1;
        curr = node2;
        while (curr != null) {

            if (set1.contains(curr)) {
                intersectedData = curr.getData();
                break;
            }
            curr = curr.getNext();
        }

        //output:
        System.out.println("Two linked list are intersected at (hashbased): " + intersectedData);

    }

    public boolean checkIfLinkedListPallindrome_1(Node<Integer> node) {

        //.........................T: O(N)
        //.........................S: O(N)
        //empty list or 1 node list is by default true
        if (node == null || node.getNext() == null) {
            return true;
        }

        Node<Integer> curr = node;
        Stack<Node<Integer>> stack = new Stack<>();
        while (curr != null) {
            stack.push(curr);
            curr = curr.getNext();
        }

        //pop stack and start checking from the head of list
        curr = node;
        while (!stack.isEmpty()) {

            Node<Integer> popped = stack.pop();
            if (curr.getData() != popped.getData()) {
                return false;
            }
            curr = curr.getNext();
        }

        //if while loop doesn't prove false
        return true;

    }

    public boolean checkIfLinkedListPallindrome_2(Node<Integer> node) {

        //.........................T: O(N)
        //.........................S: O(1)
        //https://leetcode.com/problems/palindrome-linked-list/
        //empty list or 1 node list is by default true
        if (node == null || node.getNext() == null) {
            return true;
        }

        //ex LL = 1->2->2->1->N
        //find the mid of the linked list
        Node<Integer> slow = node;
        Node<Integer> fast = node;
        while (fast != null && fast.getNext() != null) {
            slow = slow.getNext();
            fast = fast.getNext().getNext();
        }

        //LL head = 1->2->2->1->N
        //LL slow = 2->1->N
        //from slow ptr reverse it
        Node<Integer> curr = slow;
        Node<Integer> prev = null;
        Node<Integer> next = null;
        while (curr != null) {
            next = curr.getNext();
            curr.setNext(prev);
            prev = curr;
            curr = next;
        }
        //after reversing slow ptr
        //LL new Head(prev) = 1-.>2-N

        //now prev is the new head for the reversed-slow ptr
        curr = node; //actual head
        //now comparing
        //LL head = 1->2->2->1->N
        //prev = 1->2->N
        //if reversed-half of the linked list(prev) is equal to first-half-part of the actual linked list (head)
        //its pallindrme otherwise ret false
        while (prev != null) {
            if (curr.getData() != prev.getData()) {
                return false;
            }

            prev = prev.getNext();
            curr = curr.getNext();
        }

        //if while loop doesn't prove false
        return true;

    }

    public void reorderLinkedList(Node<Integer> head) {

        //First 2 approaches are in SomePracticeQuestion reorderList_1/reorderList_2
        //actual
        new LinkedListUtil<Integer>(head).print();

        //explanantion: https://youtu.be/xRYPjDMSUFw
        //ex: 1-> 2-> 3-> 4-> 5-> NULL
        //break it into 2 list (2 half)
        //l1 = 1-> 2-> 3-> NULL
        //l2 = 4-> 5-> NULL
        //reverse(l2) = 5-> 4-> NULL
        //reorderMerger(l1, l2) = 1-> 5-> 2-> 4-> 3-> NULL
        Node<Integer> slowPrev = null;
        Node<Integer> slow = head;
        Node<Integer> fast = head;
        while (fast != null && fast.getNext() != null) {
            slowPrev = slow;
            slow = slow.getNext();
            fast = fast.getNext().getNext();
        }

        //list 1 head-> slowPrev-> NULL
        Node<Integer> list1 = head;
        slowPrev.setNext(null);

        //reverse slow ptr
        Node<Integer> curr = slow;
        Node<Integer> next = null;
        Node<Integer> prev = null;

        while (curr != null) {
            next = curr.getNext();
            curr.setNext(prev);
            prev = curr;
            curr = next;
        }

        //list 2 would be slow-reversed 2nd half (prev)
        Node<Integer> list2 = prev;

        //reorder merge(list1, list2)
        while (list1 != null) {

            Node<Integer> list1_Next = list1.getNext();
            Node<Integer> list2_Next = list2.getNext();

            list1.setNext(list2);

            if (list1_Next == null) {
                break;
            }

            list2.setNext(list1_Next);

            list1 = list1_Next;
            list2 = list2_Next;
        }

        //output
        new LinkedListUtil<Integer>(head).print();
    }

    public void rearrangeLinkedListAsOddIndexFirstAndEvenIndexAtEnd(Node<Integer> head) {

        //rearrange all node in linked list in such a way that nodes at odd index comes first and nodes at even index 
        //comes last and nodex should maintain the order of their occurence in actual list
        //nodes data is not to be consider for even and odd but their INDEX ONLY
        //https://leetcode.com/problems/odd-even-linked-list/
        //explanation: https://youtu.be/C_LA6SOwVTM
        //actual
        Node<Integer> forPrinting = head;
        int index = 0;
        while (forPrinting != null) {
            System.out.print((++index) + ") " + forPrinting.getData() + "\t");
            forPrinting = forPrinting.getNext();
        }
        System.out.println();

        Node<Integer> odd = head; //at index 1)
        Node<Integer> even = head.getNext(); //at index 2)
        Node<Integer> evenHead = even; //saving the starting ref of even pointr i.e, index 2) in evenHead

        while (even != null && even.getNext() != null) {
            odd.setNext(even.getNext()); //next odd index we will get after even index i.e, 3) after 2)
            odd = odd.getNext(); //index 1) now pointing to the update i.e, index 3) and so on...
            even.setNext(odd.getNext()); //similarly next even index we will get after odd index i.e, 4) after 3)
            even = even.getNext(); //index 2) now pointing to the update i.e, index 4) and so on...
        }

        //odd last ref will be having even starting index ref (last nth odd index -> index 2)
        odd.setNext(evenHead);

        //output:
        new LinkedListUtil<Integer>(head).print();

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
        System.out.println("Level order iterative: ");
        for (List l : levels) {
            System.out.println(l);
        }

    }

    public void levelOrderTraversal_Recursive_Helper(TreeNode<Integer> root, int level,
            Map<Integer, List<Integer>> levelOrder) {

        if (root == null) {
            return;
        }

        if (levelOrder.containsKey(level)) {
            levelOrder.get(level).add(root.getData());
        } else {
            List<Integer> nodeAtLevel = new ArrayList<>();
            nodeAtLevel.add(root.getData());
            levelOrder.put(level, nodeAtLevel);
        }

        levelOrderTraversal_Recursive_Helper(root.getLeft(), level + 1, levelOrder);
        levelOrderTraversal_Recursive_Helper(root.getRight(), level + 1, levelOrder);

    }

    public void levelOrderTraversal_Recursive(TreeNode<Integer> root) {
        Map<Integer, List<Integer>> levelOrder = new TreeMap<>();
        levelOrderTraversal_Recursive_Helper(root, 0, levelOrder);

        //output:
        System.out.println("Level order recursive: ");
        for (List l : levelOrder.values()) {
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

    private void treeToDoublyLinkedList_Print(TreeNode head) {

        while (head != null) {

            System.out.print(head.getData() + " ");
            head = head.getRight();

        }
        System.out.println();
    }

    public void treeToDoublyLinkedList(TreeNode root) {
        treeToDoublyLinkedList_Helper(root);
        treeToDoublyLinkedList_Print(treeToDoublyLinkedList_HeadOfDLL);
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

    private TreeNode<Integer> lowestCommonAncestorOfTree_Helper(TreeNode<Integer> root, int N1, int N2) {

        if (root == null) {
            return null;
        }

        if (N1 == root.getData() || N2 == root.getData()) {
            return root;
        }

        TreeNode<Integer> leftNode = lowestCommonAncestorOfTree_Helper(root.getLeft(), N1, N2);
        TreeNode<Integer> rightNode = lowestCommonAncestorOfTree_Helper(root.getRight(), N1, N2);

        if (leftNode != null && rightNode != null) {
            return root;
        }

        return leftNode == null ? rightNode : leftNode;

    }

    public void lowestCommonAncestorOfTree(TreeNode<Integer> root, int N1, int N2) {
        System.out.println("Lowest common ancestor of " + N1 + " " + N2 + ": " + lowestCommonAncestorOfTree_Helper(root, N1, N2));
    }

    class CheckTreeIsSumTree { /*Helper class for checkTreeIsSumTree_Helper method*/ int data = 0;
    }

    public boolean checkTreeIsSumTree_Helper(TreeNode<Integer> root, CheckTreeIsSumTree obj) {

        if (root == null) {
            obj.data = 0;
            return true;
        }

        CheckTreeIsSumTree leftSubTreeSum = new CheckTreeIsSumTree();
        CheckTreeIsSumTree rightSubTreeSum = new CheckTreeIsSumTree();

        boolean isLeftSubTreeSumTree = checkTreeIsSumTree_Helper(root.getLeft(), leftSubTreeSum);
        boolean isRightSubTreeSumTree = checkTreeIsSumTree_Helper(root.getRight(), rightSubTreeSum);

        //calculating data for the current root node itself
        obj.data = root.getData() + leftSubTreeSum.data + rightSubTreeSum.data;

        //current root node should not be be leaf
        if (!(root.getLeft() == null && root.getRight() == null)
                && //current root is not equal to the sum of left and rigth sub tree 
                (root.getData() != leftSubTreeSum.data + rightSubTreeSum.data)) {
            return false;
        }

        return isLeftSubTreeSumTree && isRightSubTreeSumTree;

    }

    public void checkTreeIsSumTree(TreeNode<Integer> root) {
        System.out.println("Check if a tree is sum tree: " + checkTreeIsSumTree_Helper(root, new CheckTreeIsSumTree()));
    }

    class TreeLongestPathNodeSum {

        /*Helper class for longestPathNodeSum method*/
        List<Integer> path = new ArrayList<>();
        int maxPathLength = path.size();
        int longestPathSum = 0;
        int maxSumOfAnyPath = 0;
    }

    private void longestPathNodeSum_Helper(TreeNode<Integer> root, TreeLongestPathNodeSum obj) {

        if (root == null) {
            return;
        }
        obj.path.add(root.getData());
        longestPathNodeSum_Helper(root.getLeft(), obj);
        longestPathNodeSum_Helper(root.getRight(), obj);

        int pathSum = 0;
        for (int nodes : obj.path) {
            pathSum += nodes;
        }
        if (obj.path.size() > obj.maxPathLength) {
            obj.longestPathSum = pathSum;
        }
        obj.maxSumOfAnyPath = Math.max(obj.maxSumOfAnyPath, pathSum);
        obj.maxPathLength = Math.max(obj.maxPathLength, obj.path.size());

        //remove the last added node
        obj.path.remove(obj.path.size() - 1);
    }

    public void longestPathNodeSum(TreeNode<Integer> root) {
        TreeLongestPathNodeSum obj = new TreeLongestPathNodeSum();
        longestPathNodeSum_Helper(root, obj);
        System.out.println("The sum of nodes of longest path of tree: " + obj.longestPathSum);
    }

    private void findPredecessorAndSuccessorInBST_Helper(TreeNode<Integer> root, int key, TreeNode<Integer>[] result) {

        if (root == null) {
            return;
        }

        if (root.getData() == key) {

            if (root.getLeft() != null) {
                //predecessor : rightmost node in the left subtree
                TreeNode<Integer> pred = root.getLeft();
                while (pred.getRight() != null) {
                    pred = pred.getRight();
                }

                result[0] = pred;
            }

            if (root.getRight() != null) {
                //successor : leftmost node in the right subtree
                TreeNode<Integer> succ = root.getRight();
                while (succ.getLeft() != null) {
                    succ = succ.getLeft();
                }

                result[1] = succ;
            }

            return;

        }

        //key is less than root data so move to whole left sub tree
        //ex: root = 2, key = 4
        //2 > 4 -> else
        if (root.getData() > key) {
            result[1] = root; //succ
            findPredecessorAndSuccessorInBST_Helper(root.getLeft(), key, result);
        } else {
            //else move to whole right sub tree
            //in else because 2 is not greater 4 
            //inorder of bst is sorted list [2, 4] and 2 will be pred of 4
            result[0] = root; //pred
            findPredecessorAndSuccessorInBST_Helper(root.getRight(), key, result);
        }

    }

    public void findPredecessorAndSuccessorInBST(TreeNode<Integer> root, int key) {

        //can use list also
        //[0] : predecessor, [1] : successor
        TreeNode<Integer>[] result = new TreeNode[2];
        findPredecessorAndSuccessorInBST_Helper(root, key, result);
        System.out.println("Predecessor and successor of BST: "
                + (result[0] != null ? result[0].getData() : "null") + " "
                + (result[1] != null ? result[1].getData() : "null"));

    }

    private int countNodesThatLieInGivenRange_Count = 0;

    private void countNodesThatLieInGivenRange_Helper(TreeNode<Integer> root, int low, int high) {

        if (root == null) {
            return;
        }

        if (root.getData() >= low && root.getData() <= high) {
            countNodesThatLieInGivenRange_Count++;
        }

        countNodesThatLieInGivenRange_Helper(root.getLeft(), low, high);
        countNodesThatLieInGivenRange_Helper(root.getRight(), low, high);

    }

    public void countNodesThatLieInGivenRange(TreeNode<Integer> root, int low, int high) {
        countNodesThatLieInGivenRange_Count = 0;
        countNodesThatLieInGivenRange_Helper(root, low, high);
        System.out.println("No. of nodes that lie in given range: " + countNodesThatLieInGivenRange_Count);
    }

    public void flattenBSTToLinkedList(TreeNode root) {

        if (root == null) {
            return;
        }

        /*Deque<TreeNode> dQueue = new ArrayDeque<>();
         dQueue.add(root);

         while (!dQueue.isEmpty()) {

         TreeNode curr = dQueue.removeFirst();

         if (curr.getRight() != null) {
         dQueue.addFirst(curr.getRight());
         }

         if (curr.getLeft() != null) {
         dQueue.addFirst(curr.getLeft());
         }

         if (!dQueue.isEmpty()) {
         curr.setRight(dQueue.peek());
         curr.setLeft(null);
         }

         }*/
        /*List<TreeNode> q = new ArrayList<>();
         q.add(root);
         while (!q.isEmpty()) {

         TreeNode curr = q.remove(0);

         if (curr.getRight() != null) {
         q.add(0, curr.getRight());
         }

         if (curr.getLeft() != null) {
         q.add(0, curr.getLeft());
         }

         if (!q.isEmpty()) {
         curr.setRight(q.get(0));
         curr.setLeft(null);
         }

         }*/
        //using LIFO stack
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {

            TreeNode curr = stack.pop();

            //we need left at peek of stack so pushing right first
            //and then left so that left can be at peek
            if (curr.getRight() != null) {
                stack.push(curr.getRight());
            }

            if (curr.getLeft() != null) {
                stack.push(curr.getLeft());
            }

            if (!stack.isEmpty()) {

                curr.setRight(stack.peek());
                curr.setLeft(null);
            }

        }

        //output:
        new BinaryTree(root).treeBFS();
        System.out.println();

    }

    private void diagonalTraversalOfTree_Helper(TreeNode<Integer> root, int level, Map<Integer, List<Integer>> result) {

        if (root == null) {
            return;
        }

        List<Integer> listAtLevel = result.get(level);

        if (listAtLevel == null) {
            listAtLevel = new ArrayList<>();
            listAtLevel.add(root.getData());
        } else {
            listAtLevel.add(root.getData());
        }

        result.put(level, listAtLevel);

        diagonalTraversalOfTree_Helper(root.getLeft(), level + 1, result);
        diagonalTraversalOfTree_Helper(root.getRight(), level, result);

    }

    public void diagonalTraversalOfTree(TreeNode<Integer> root) {

        Map<Integer, List<Integer>> result = new HashMap<>();
        diagonalTraversalOfTree_Helper(root, 0, result);
        System.out.println("Diagonal traversal of tree");
        for (Map.Entry<Integer, List<Integer>> e : result.entrySet()) {
            System.out.println(e.getValue());
        }
    }

    private int diameterOfTree_Helper(TreeNode<Integer> root, Height height) {

        if (root == null) {
            height.height = 0;
            return 0;
        }

        Height leftSubTreeHeight = new Height();
        Height rightSubTreeHeight = new Height();

        int leftTreeDiameter = diameterOfTree_Helper(root.getLeft(), leftSubTreeHeight);
        int rightTreeDiameter = diameterOfTree_Helper(root.getRight(), rightSubTreeHeight);

        //current node height
        height.height = Math.max(leftSubTreeHeight.height, rightSubTreeHeight.height) + 1;

        return Math.max(
                Math.max(leftTreeDiameter, rightTreeDiameter),
                leftSubTreeHeight.height + rightSubTreeHeight.height + 1
        );

    }

    public void diameterOfTree(TreeNode<Integer> root) {
        System.out.println("Diameter of tree: " + diameterOfTree_Helper(root, new Height()));
    }

    class CheckIfBinaryTreeIsMaxHeapClass {
        /*Helper class for checkIfBinaryTreeIsMaxHeap method*/

        int data;
    }

    private boolean checkIfBinaryTreeIsMaxHeap_Helper(TreeNode<Integer> root, CheckIfBinaryTreeIsMaxHeapClass obj) {

        if (root == null) {
            obj.data = Integer.MIN_VALUE;
            return true;
        }

        CheckIfBinaryTreeIsMaxHeapClass leftSubTree = new CheckIfBinaryTreeIsMaxHeapClass();
        CheckIfBinaryTreeIsMaxHeapClass rightSubTree = new CheckIfBinaryTreeIsMaxHeapClass();

        boolean isLeftMaxHeap = checkIfBinaryTreeIsMaxHeap_Helper(root.getLeft(), leftSubTree);
        boolean isRightMaxHeap = checkIfBinaryTreeIsMaxHeap_Helper(root.getRight(), rightSubTree);

        //calculating current node's object
        obj.data = root.getData();

        //if root's data is less than its immediate left OR right child return false
        if (root.getData() < leftSubTree.data || root.getData() < rightSubTree.data) {
            return false;
        }

        return isLeftMaxHeap && isRightMaxHeap;

    }

    public void checkIfBinaryTreeIsMaxHeap(TreeNode<Integer> root) {
        System.out.println("Given binary tree is max heap: "
                + checkIfBinaryTreeIsMaxHeap_Helper(root, new CheckIfBinaryTreeIsMaxHeapClass()));
    }

    public boolean checkIfAllLevelsOfTwoTreesAreAnagrams_1(TreeNode<Integer> root1, TreeNode<Integer> root2) {

        //this approach performs level order traversal first and then anagrams checking
        if (root1 == null && root2 == null) {
            return true;
        }

        if (root1 == null || root2 == null) {
            return false;
        }

        Map<Integer, List<Integer>> levelOrder1 = new TreeMap<>();
        levelOrderTraversal_Recursive_Helper(root1, 0, levelOrder1); //T: O(N)

        Map<Integer, List<Integer>> levelOrder2 = new TreeMap<>();
        levelOrderTraversal_Recursive_Helper(root2, 0, levelOrder2); //T: O(N)

        //if both tree are of different levels then two trees acn't be anagrams
        if (levelOrder1.size() != levelOrder2.size()) {
            return false;
        }

        //T: O(H) H = height of tree
        for (int level = 0; level < levelOrder1.size(); level++) {

            List<Integer> l1 = levelOrder1.get(level);
            List<Integer> l2 = levelOrder2.get(level);

            //sort: T: O(Logl1) + O(Logl2)
            Collections.sort(l1);
            Collections.sort(l2);

            //if levels of two trees after sorting are not equal then they are not anagram
            //ex l1.sort: [2,3], l2.sort: [3,4] then l1 != l2
            if (!l1.equals(l2)) {
                return false;
            }

        }

        return true;
    }

    public boolean checkIfAllLevelsOfTwoTreesAreAnagrams_2(TreeNode<Integer> root1, TreeNode<Integer> root2) {

        //this approach performs level order traversal and anagrams checking at the same time
        if (root1 == null && root2 == null) {
            return true;
        }

        if (root1 == null || root2 == null) {
            return false;
        }

        Queue<TreeNode<Integer>> q1 = new LinkedList<>();
        Queue<TreeNode<Integer>> q2 = new LinkedList<>();
        q1.add(root1);
        q2.add(root2);

        Queue<TreeNode<Integer>> intQ1 = new LinkedList<>();
        Queue<TreeNode<Integer>> intQ2 = new LinkedList<>();

        List<Integer> l1 = new ArrayList<>();
        List<Integer> l2 = new ArrayList<>();

        while (!q1.isEmpty() && !q2.isEmpty()) {

            TreeNode<Integer> curr1 = q1.poll();
            TreeNode<Integer> curr2 = q2.poll();

            l1.add(curr1.getData());
            l2.add(curr2.getData());

            if (curr1.getLeft() != null) {
                intQ1.add(curr1.getLeft());
            }

            if (curr1.getRight() != null) {
                intQ1.add(curr1.getRight());
            }

            if (curr2.getLeft() != null) {
                intQ2.add(curr2.getLeft());
            }

            if (curr2.getRight() != null) {
                intQ2.add(curr2.getRight());
            }

            if (q1.isEmpty() && q2.isEmpty()) {

                Collections.sort(l1);
                Collections.sort(l2);

                //if after sorting the nodes at a paticular level from both
                //the tree are not equal
                //ex l1.sort: [2,3], l2.sort: [3,4] then l1 != l2
                if (!l1.equals(l2)) {
                    return false;
                }

                l1.clear();
                l2.clear();

                //intQ holds the immediate child nodes of a parent node
                //if the no. of immediate child nodes are different then further 
                //checking for anagrams are not req.
                //ex T1: 1.left = 2, 1.right = 3
                //T2: 1.left = 2
                //at parent node = 1 intQ will hold immediate childs
                //intQ1 = [2,3], intQ2 = [2] here intQ1.size != intQ2.size
                if (intQ1.size() != intQ2.size()) {
                    return false;
                }

                q1.addAll(intQ1);
                q2.addAll(intQ2);

                intQ1.clear();
                intQ2.clear();

            }
        }

        //if none of the cond in while is false then all the levels in both tree are anagrams
        return true;
    }

    private boolean areTwoTreeIsoMorphic_Helper(TreeNode<Integer> root1, TreeNode<Integer> root2) {

        if (root1 == null && root2 == null) {
            return true;
        }

        if (root1 == null || root2 == null) {
            return false;
        }

        return root1.getData() == root2.getData()
                && ((areTwoTreeIsoMorphic_Helper(root1.getLeft(), root2.getRight()) && areTwoTreeIsoMorphic_Helper(root1.getRight(), root2.getLeft()))
                || (areTwoTreeIsoMorphic_Helper(root1.getLeft(), root2.getLeft()) && areTwoTreeIsoMorphic_Helper(root1.getRight(), root2.getRight())));

    }

    public boolean areTwoTreeIsoMorphic(TreeNode<Integer> root1, TreeNode<Integer> root2) {
        return areTwoTreeIsoMorphic_Helper(root1, root2);
    }

    private String findDuplicateSubtreeInAGivenTree_Inorder(TreeNode<Integer> root,
            Map<String, Integer> map, List<TreeNode<Integer>> subtrees) {

        if (root == null) {
            return "";
        }

        String str = "(";
        str += findDuplicateSubtreeInAGivenTree_Inorder(root.getLeft(), map, subtrees);
        str += String.valueOf(root.getData());
        str += findDuplicateSubtreeInAGivenTree_Inorder(root.getRight(), map, subtrees);
        str += ")";

//        System.out.println(str);
        if (map.containsKey(str) && map.get(str) == 1) {
            //System.out.println(root.getData()+ " "); //print the starting node of suplicate subtree
            subtrees.add(root);
        }

        map.put(str, map.getOrDefault(str, 0) + 1);

        return str;

    }

    public void findDuplicateSubtreeInAGivenTree(TreeNode<Integer> root) {
        Map<String, Integer> map = new HashMap<>();
        List<TreeNode<Integer>> subtrees = new ArrayList<>();
        findDuplicateSubtreeInAGivenTree_Inorder(root, map, subtrees);

        //output:
        //print level order of found subtrees
        for (TreeNode<Integer> tree : subtrees) {
            levelOrderTraversal_Recursive(tree);
        }
    }

    private void allNodesAtKDistanceFromRoot(TreeNode<Integer> root, int level,
            int K, List<Integer> result) {

        if (root == null) {
            return;
        }

        if (level == K) {
            result.add(root.getData());
        }

        allNodesAtKDistanceFromRoot(root.getLeft(), level + 1, K, result);
        allNodesAtKDistanceFromRoot(root.getRight(), level + 1, K, result);
    }

    private int printAllTheNodesAtKDistanceFromTargetNode_DFS(TreeNode<Integer> root, int target,
            int K, List<Integer> result) {

        if (root == null) {
            return -1;
        }

        if (root.getData() == target) {
            //search all the nodes at K dist below the target node
            allNodesAtKDistanceFromRoot(root, 0, K, result);
            return 1;
        }

        int left = printAllTheNodesAtKDistanceFromTargetNode_DFS(root.getLeft(), target, K, result);

        if (left != -1) {
            if (left == K) {
                result.add(root.getData());
            }
            allNodesAtKDistanceFromRoot(root.getRight(), left + 1, K, result);
            return left + 1;
        }

        int right = printAllTheNodesAtKDistanceFromTargetNode_DFS(root.getRight(), target, K, result);

        if (right != -1) {
            if (right == K) {
                result.add(root.getData());
            }
            allNodesAtKDistanceFromRoot(root.getLeft(), right + 1, K, result);
            return right + 1;
        }

        return -1;
    }

    public void printAllTheNodesAtKDistanceFromTargetNode(TreeNode<Integer> root, int target, int K) {

        List<Integer> result = new ArrayList<>();
        printAllTheNodesAtKDistanceFromTargetNode_DFS(root, target, K, result);
        //output:
        System.out.println("All nodes at K distance from target node: " + result);

    }

    private boolean deleteTreeNodesAndReturnForest_Helper(TreeNode<Integer> root,
            Set<Integer> deleteSet, List<TreeNode<Integer>> result) {

        if (root == null) {
            return false;
        }

        boolean deleteLeft = deleteTreeNodesAndReturnForest_Helper(root.getLeft(), deleteSet, result);
        boolean deleteRight = deleteTreeNodesAndReturnForest_Helper(root.getRight(), deleteSet, result);

        if (deleteLeft) {
            root.setLeft(null);
        }

        if (deleteRight) {
            root.setRight(null);
        }

        if (deleteSet.contains(root.getData())) {

            if (root.getLeft() != null) {
                result.add(root.getLeft());
            }

            if (root.getRight() != null) {
                result.add(root.getRight());
            }
            return true;
        }

        return deleteLeft && deleteRight && deleteSet.contains(root.getData());

    }

    private TreeNode<Integer> deleteTreeNodesAndReturnForest_Helper2(TreeNode<Integer> root,
            Set<Integer> deleteSet, List<TreeNode<Integer>> result) {

        //Easier explanation
        if (root == null) {
            return null;
        }

        root.setLeft(deleteTreeNodesAndReturnForest_Helper2(root.getLeft(), deleteSet, result));
        root.setRight(deleteTreeNodesAndReturnForest_Helper2(root.getRight(), deleteSet, result));

        if (deleteSet.contains(root.getData())) {

            if (root.getLeft() != null) {
                result.add(root.getLeft());
            }

            if (root.getRight() != null) {
                result.add(root.getRight());
            }

            return null;
        }

        return root;

    }

    public void deleteTreeNodesAndReturnForest(TreeNode<Integer> root, int[] toDelete) {

        List<TreeNode<Integer>> result = new ArrayList<>();

        if (root == null) {
            return;
        }

        Set<Integer> deleteSet = new HashSet<>();
        for (int x : toDelete) {
            deleteSet.add(x);
        }

//        boolean res = deleteTreeNodesAndReturnForest_Helper(root, deleteSet, result);
//
//        if (res == false || (res && !deleteSet.contains(root.getData()))) {
//            result.add(root);
//        }
        //Easier explanation
        deleteTreeNodesAndReturnForest_Helper2(root, deleteSet, result);
        if (!deleteSet.contains(root.getData())) {
            result.add(root);
        }

        //output:
        for (TreeNode<Integer> curr : result) {
            levelOrderTraversal_Iterative(curr);
            System.out.println();
        }

    }

    private TreeNode<Integer> constructBinaryTreeFromInorderPreorderArray_Helper(int preIndex, int inStart, int inEnd,
            int[] inorder, int[] preorder) {

        if (preIndex >= preorder.length || inStart > inEnd) {
            return null;
        }

        TreeNode<Integer> root = new TreeNode<>(preorder[preIndex]);

        int index = inStart;
        for (; index <= inEnd; index++) {
            if (preorder[preIndex] == inorder[index]) {
                break;
            }
        }

        root.setLeft(constructBinaryTreeFromInorderPreorderArray_Helper(preIndex + 1, inStart, index - 1, inorder, preorder));
        root.setRight(constructBinaryTreeFromInorderPreorderArray_Helper(preIndex + 1 + index - inStart, index + 1, inEnd, inorder, preorder));

        return root;
    }

    public void constructBinaryTreeFromInorderPreorderArray(int[] inorder, int[] preorder) {

        if (inorder.length != preorder.length) {
            return;
        }

        int n = inorder.length;
        TreeNode<Integer> root = constructBinaryTreeFromInorderPreorderArray_Helper(0, 0, n - 1, inorder, preorder);

        //output
        new BinaryTree<>(root).treeBFS();
        System.out.println();

    }

    private TreeNode<Integer> constructBinarySearchTreeFromPreorderArray_Helper(int[] preorder, int preStart, int preEnd) {

        if (preStart > preEnd) {
            return null;
        }

        TreeNode<Integer> root = new TreeNode<>(preorder[preStart]);

        int index = preStart + 1;
        for (; index <= preEnd; index++) {
            if (preorder[index] > preorder[preStart]) {
                break;
            }
        }

        root.setLeft(constructBinarySearchTreeFromPreorderArray_Helper(preorder, preStart + 1, index - 1));
        root.setRight(constructBinarySearchTreeFromPreorderArray_Helper(preorder, index, preEnd));

        return root;

    }

    public void constructBinarySearchTreeFromPreorderArray(int[] preorder) {

        int n = preorder.length;
        TreeNode<Integer> root = constructBinarySearchTreeFromPreorderArray_Helper(preorder, 0, n - 1);

        //output:
        new BinaryTree<Integer>(root).treeBFS();
        System.out.println();

    }

    private TreeNode<Integer> constructBinarySearchTreeFromPostorderArray_Helper(int[] postorder, int postStart, int postEnd) {

        if (postEnd > postStart) {
            return null;
        }

        TreeNode<Integer> root = new TreeNode<>(postorder[postStart]);

        int index = postStart - 1;
        for (; index >= postEnd; index--) {
            if (postorder[postStart] > postorder[index]) {
                break;
            }
        }

        root.setLeft(constructBinarySearchTreeFromPostorderArray_Helper(postorder, index, postEnd));
        root.setRight(constructBinarySearchTreeFromPostorderArray_Helper(postorder, postStart - 1, index + 1));

        return root;

    }

    public void constructBinarySearchTreeFromPostorderArray(int[] postorder) {

        int n = postorder.length;
        TreeNode<Integer> root = constructBinarySearchTreeFromPostorderArray_Helper(postorder, n - 1, 0);

        //output:
        new BinaryTree<Integer>(root).treeBFS();
        System.out.println();

    }

    private TreeNode<Integer> leavesOfTreeToDoublyLinkedListAndRemoveLeaves_DLL;
    private TreeNode<Integer> leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev;

    private boolean leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Helper(TreeNode<Integer> root) {

        if (root == null) {
            return false;
        }

        boolean leftLeaf = leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Helper(root.getLeft());
        boolean rightLeaf = leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Helper(root.getRight());

        //left most leaf node
        if (leavesOfTreeToDoublyLinkedListAndRemoveLeaves_DLL == null) {
            leavesOfTreeToDoublyLinkedListAndRemoveLeaves_DLL = root;
        }

        if (root.getLeft() == null && root.getRight() == null) {

            if (leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev != null) {
                leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev.setRight(root);
                root.setLeft(leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev);
            }

            leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev = root;

            return true;
        }

        if (leftLeaf) {
            root.setLeft(null);
        }

        if (rightLeaf) {
            root.setRight(null);
        }

        return false;
    }

    public void leavesOfTreeToDoublyLinkedListAndRemoveLeaves(TreeNode<Integer> root) {

        leavesOfTreeToDoublyLinkedListAndRemoveLeaves_DLL = null;
        leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev = null;

        leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Helper(root);

        //output;
        treeToDoublyLinkedList_Print(leavesOfTreeToDoublyLinkedListAndRemoveLeaves_DLL);

    }

    private void checkIfTwoNAryTreeAreMirror_PushStack(List<List<Integer>> tree1, Stack<Integer> stack, int root) {

        if (tree1.size() == 0) {
            return;
        }

        stack.push(root);
        List<Integer> childs = tree1.get(root);
        for (int childNodes : childs) {
            checkIfTwoNAryTreeAreMirror_PushStack(tree1, stack, childNodes);
        }

    }

    private void checkIfTwoNAryTreeAreMirror_PushQueue(List<List<Integer>> tree2, Queue<Integer> queue, int root) {

        if (tree2.size() == 0) {
            return;
        }

        List<Integer> childs = tree2.get(root);
        for (int childNodes : childs) {
            checkIfTwoNAryTreeAreMirror_PushQueue(tree2, queue, childNodes);
        }
        queue.add(root);

    }

    public boolean checkIfTwoNAryTreeAreMirror(List<List<Integer>> tree1, List<List<Integer>> tree2) {

        //Explanation: https://youtu.be/UGzXSDZv-SY
        Stack<Integer> stack = new Stack<>();
        checkIfTwoNAryTreeAreMirror_PushStack(tree1, stack, 0);

        Queue<Integer> queue = new LinkedList<>();
        checkIfTwoNAryTreeAreMirror_PushQueue(tree2, queue, 0);

//        System.out.println(stack+"---"+queue);
        if (stack.size() != queue.size()) {
            System.out.println("Not a mirror images");
            return false;
        }

        while (!stack.isEmpty() && !queue.isEmpty()) {
            if (stack.pop() != queue.poll()) {
                return false;
            }
        }
        return true;
    }

    public void deepestLeavesSumOfTree_Iterative(TreeNode<Integer> root) {

//        base edge case
//        if(root == null){
//            return 0;
//        }
        //do level order traversal
        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);

        int sum = 0;
        while (!q.isEmpty()) {

            sum = 0;

            //size of queue at a particular level is required beforehand
            //because we putting the child of curr node in the queue
            //we don't want those nodes to be consumed in the for() loop
            int size = q.size();
            for (int i = 0; i < size; i++) {

                //add up all the nodes at a particular level in the sum variable
                //and at the same time add the chiild nodes of curr node into queue
                TreeNode<Integer> curr = q.poll();

                if (curr != null) {
                    sum += curr.getData();
                    if (curr.getLeft() != null) {
                        q.add(curr.getLeft());
                    }

                    if (curr.getRight() != null) {
                        q.add(curr.getRight());
                    }
                }
            }

        }

        //output:
        System.out.println("Deepest leaves sum: " + sum);

    }

    private void smallestStringInTreeFromLeafToRoot_Helper(TreeNode<Integer> root, StringBuilder sb,
            PriorityQueue<String> strs) {

        if (root == null) {
            return;
        }

        sb.insert(0, (char) (root.getData() + 'a'));

        smallestStringInTreeFromLeafToRoot_Helper(root.getLeft(), sb, strs);
        smallestStringInTreeFromLeafToRoot_Helper(root.getRight(), sb, strs);

        if (root.getLeft() == null && root.getRight() == null) {
            strs.add(sb.toString());
        }

        sb.deleteCharAt(0);

    }

    public void smallestStringInTreeFromLeafToRoot(TreeNode<Integer> root) {

        //each nodes contains value 0 - 25 representing alphabets a - z
        //find the lexicographically smallest string (leaf to root) in the tree
        StringBuilder sb = new StringBuilder();
        PriorityQueue<String> strs = new PriorityQueue<>((a, b) -> a.compareTo(b));
        smallestStringInTreeFromLeafToRoot_Helper(root, sb, strs);

        //output
        System.out.println("Smallest string: " + strs.poll());

    }

    private void printSumWhereRootToLeafPathIsANumber_createNumberHelper(TreeNode<Integer> root, int num, List<Integer> nums) {

        if (root == null) {
            return;
        }

        printSumWhereRootToLeafPathIsANumber_createNumberHelper(root.getLeft(), num * 10 + root.getData(), nums);
        printSumWhereRootToLeafPathIsANumber_createNumberHelper(root.getRight(), num * 10 + root.getData(), nums);

        if (root.getLeft() == null && root.getRight() == null) {
            nums.add(num * 10 + root.getData());
        }

    }

    public void printSumWhereRootToLeafPathIsANumber(TreeNode<Integer> root) {

        //ex: 
        /* root:
         1
         2       3
         */
        //num1 = 12 (as 1->2 is a path)
        //num2 = 13 (as 1->3 is a path)
        //each path from root to leaf is a separate num
        //print sum of all such number
        List<Integer> nums = new ArrayList<>();
        printSumWhereRootToLeafPathIsANumber_createNumberHelper(root, 0, nums);

        int sum = 0;
        for (int num : nums) {
            sum += num;
        }

        //output:
        System.out.println("Sum of path of tree representing as a number: " + sum);

    }

    private void fixTwoSwappedNodesInBST_Helper_IsBST(TreeNode<Integer> root) {
        if (root == null) {
            return;
        }

        fixTwoSwappedNodesInBST_Helper_IsBST(root.getLeft());

        if (fixTwoSwappedNodesInBST_Prev != null && fixTwoSwappedNodesInBST_Prev.getData() > root.getData()) {

            if (fixTwoSwappedNodesInBST_First == null) {
                fixTwoSwappedNodesInBST_First = fixTwoSwappedNodesInBST_Prev;
                fixTwoSwappedNodesInBST_Middle = root;
            } else {
                fixTwoSwappedNodesInBST_Last = root;
            }
        }

        fixTwoSwappedNodesInBST_Prev = root;

        fixTwoSwappedNodesInBST_Helper_IsBST(root.getRight());
    }

    private void fixTwoSwappedNodesInBST_Helper(TreeNode<Integer> root) {

        fixTwoSwappedNodesInBST_First = null;
        fixTwoSwappedNodesInBST_Middle = null;
        fixTwoSwappedNodesInBST_Last = null;
        fixTwoSwappedNodesInBST_Prev = null;

        fixTwoSwappedNodesInBST_Helper_IsBST(root);

        if (fixTwoSwappedNodesInBST_First != null && fixTwoSwappedNodesInBST_Last != null) {
            int temp = fixTwoSwappedNodesInBST_First.getData();
            fixTwoSwappedNodesInBST_First.setData(fixTwoSwappedNodesInBST_Last.getData());
            fixTwoSwappedNodesInBST_Last.setData(temp);
        } else if (fixTwoSwappedNodesInBST_First != null && fixTwoSwappedNodesInBST_Middle != null) {
            int temp = fixTwoSwappedNodesInBST_First.getData();
            fixTwoSwappedNodesInBST_First.setData(fixTwoSwappedNodesInBST_Middle.getData());
            fixTwoSwappedNodesInBST_Middle.setData(temp);
        }

    }

    TreeNode<Integer> fixTwoSwappedNodesInBST_First;
    TreeNode<Integer> fixTwoSwappedNodesInBST_Middle;
    TreeNode<Integer> fixTwoSwappedNodesInBST_Last;
    TreeNode<Integer> fixTwoSwappedNodesInBST_Prev;

    public void fixTwoSwappedNodesInBST(TreeNode<Integer> root) {

        //https://www.geeksforgeeks.org/fix-two-swapped-nodes-of-bst/
        //actual
        new BinaryTree<Integer>(root).treeInorder();
        System.out.println();

        fixTwoSwappedNodesInBST_Helper(root);

        //output
        new BinaryTree<Integer>(root).treeInorder();
        System.out.println();

    }

    private TreeNode<Integer> mergeTwoBinaryTree_Heleper(TreeNode<Integer> root1, TreeNode<Integer> root2) {

        if (root1 == null) {
            return root2;
        }

        if (root2 == null) {
            return root1;
        }

        root1.setData(root1.getData() + root2.getData());

        root1.setLeft(mergeTwoBinaryTree_Heleper(root1.getLeft(), root2.getLeft()));
        root1.setRight(mergeTwoBinaryTree_Heleper(root1.getRight(), root2.getRight()));

        return root1;

    }

    public void mergeTwoBinaryTree(TreeNode<Integer> root1, TreeNode<Integer> root2) {
        //actual
        System.out.println("Actaul trees");
        new BinaryTree<Integer>(root1).treeBFS();
        System.out.println();
        new BinaryTree<Integer>(root2).treeBFS();
        System.out.println();

        mergeTwoBinaryTree_Heleper(root1, root2);

        //output
        System.out.println("Merged both trees in tree1");
        new BinaryTree<Integer>(root1).treeBFS();

    }

    private long numberOfWaysToCreateBSTAndBTWithGivenN_Factorial(long N) {
        long result = 1;
        for (long i = 1; i <= N; i++) {
            result *= i;
        }

        return result;
    }

    private long numberOfWaysToCreateBSTAndBTWithGivenN_BinomialCoeff(long N, long K) {
        long result = 1;

        if (K > N - K) {
            K = N - K;
        }

        for (long i = 0; i < K; i++) {
            result *= (N - i);
            result /= (i + 1);
        }

        return result;
    }

    private long numberOfWaysToCreateBSTAndBTWithGivenN_CatalanNumberOfGivenNthNumber(long N) {

        //Catalan number series:
        //https://www.youtube.com/watch?v=CMaZ69P1bAc
        //https://www.geeksforgeeks.org/program-nth-catalan-number/
        //https://www.geeksforgeeks.org/total-number-of-possible-binary-search-trees-with-n-keys/#
        long cat = numberOfWaysToCreateBSTAndBTWithGivenN_BinomialCoeff(2 * N, N);
        return cat / (N + 1);
    }

    public void numberOfWaysToCreateBSTAndBTWithGivenN(long N) {

        //problem: https://leetcode.com/problems/unique-binary-search-trees
        //Explanation :
        //https://www.geeksforgeeks.org/total-number-of-possible-binary-search-trees-with-n-keys/#
        //ways to create BST
        //find the catalan number of given Nth number 
        System.out.println("Number of ways to create a binary search tree with given N nodes: "
                + numberOfWaysToCreateBSTAndBTWithGivenN_CatalanNumberOfGivenNthNumber(N));

        //ways to create BT
        //find the catalan number of given Nth number * factorial(N)
        System.out.println("Number of ways to create a binary tree with given N nodes: "
                + (numberOfWaysToCreateBSTAndBTWithGivenN_CatalanNumberOfGivenNthNumber(N)
                * numberOfWaysToCreateBSTAndBTWithGivenN_Factorial(N)));

    }

    public boolean checkIfBinaryTreeIsCompleteOrNot(TreeNode<Integer> root) {

        //explanation: https://youtu.be/j16cwbLEf9w
        //complete binary tree:
        /*
         In a complete binary tree, every level, except possibly the last, 
         is completely filled, and all nodes in the last level are as far left 
         as possible. It can have between 1 and 2^h nodes inclusive at the 
         last level h.
         */
        //all the nodes at last level should be left-most alinged
        //if any null is present before the very last node at the level
        //that means tree is not complete binary tree
        boolean isNullBeforeLastNode = false;

        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            TreeNode<Integer> curr = q.poll();

            if (curr == null) {
                isNullBeforeLastNode = true;
            } else {
                if (isNullBeforeLastNode) {
                    return false;
                }
                //put all the left and right child nodes as it is
                //without checking for null
                q.add(curr.getLeft());
                q.add(curr.getRight());
            }
        }
        return true;
    }

    private void maximumWidthOfBinaryTree_Helper(TreeNode<Integer> root, int level,
            int position, Map<Integer, Integer> map) {
        if (root == null) {
            return;
        }

        map.putIfAbsent(level, position);
        maximumWidthOfBinaryTree_MaxWidth = Math.max(maximumWidthOfBinaryTree_MaxWidth,
                position - map.get(level) + 1);

        maximumWidthOfBinaryTree_Helper(root.getLeft(), level + 1, 2 * position, map);
        maximumWidthOfBinaryTree_Helper(root.getRight(), level + 1, 2 * position + 1, map);

    }

    int maximumWidthOfBinaryTree_MaxWidth;

    public void maximumWidthOfBinaryTree(TreeNode<Integer> root) {

        //explanation: https://youtu.be/sm4UdCO2868
        maximumWidthOfBinaryTree_MaxWidth = 0;
        Map<Integer, Integer> map = new HashMap<>();
        maximumWidthOfBinaryTree_Helper(root, 0, 0, map);

        //output
        System.out.println("Max Width of binary tree: " + maximumWidthOfBinaryTree_MaxWidth);
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

    public void nextGreaterElementInRightInArray(int[] arr) {

        Stack<Integer> st = new Stack<>();
        int[] result = new int[arr.length];
        int index = arr.length - 1;
        for (int i = arr.length - 1; i >= 0; i--) {

            while (!st.isEmpty() && st.peek() < arr[i]) {
                st.pop();
            }

            if (st.isEmpty()) {
                result[index--] = -1;
            } else {
                result[index--] = st.peek();
            }
            st.push(arr[i]);
        }

        //output
        for (int x : result) {
            System.out.print(x + " ");
        }
        System.out.println();

    }

    public void largestAreaInHistogram(int[] hist) {

        // Create an empty stack. The stack holds indexes of hist[] array 
        // The bars stored in stack are always in increasing order of their 
        // heights. 
        Stack<Integer> st = new Stack<>();
        int n = hist.length;
        int maxArea = 0; // Initialize max area 
        int top;  // To store top of stack 
        int areaWithTop; // To store area with top bar as the smallest bar 

        // Run through all bars of given histogram 
        int i = 0;
        while (i < n) {
            // If this bar is higher than the bar on top stack, push it to stack 
            if (st.isEmpty() || hist[st.peek()] <= hist[i]) {
                st.push(i++);

                // If this bar is lower than top of stack, then calculate area of rectangle  
                // with stack top as the smallest (or minimum height) bar. 'i' is  
                // 'right index' for the top and element before top in stack is 'left index' 
            } else {

                top = st.pop();  // store the top index 
                // Calculate the area with hist[tp] stack as smallest bar 
                areaWithTop = hist[top] * (st.isEmpty() ? i : i - st.peek() - 1);
                // update max area, if needed 
                maxArea = Math.max(maxArea, areaWithTop);
            }
        }

        // Now pop the remaining bars from stack and calculate area with every 
        // popped bar as the smallest bar 
        while (!st.isEmpty()) {
            top = st.pop();
            areaWithTop = hist[top] * (st.isEmpty() ? i : i - st.peek() - 1);
            maxArea = Math.max(maxArea, areaWithTop);
        }

        //output:
        System.out.println("Max area of histogram: " + maxArea);

    }

    public void postfixExpressionEvaluation_SingleDigit(String expr) {

        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < expr.length(); i++) {

            char ch = expr.charAt(i);
            if (Character.isDigit(ch)) {
                stack.push(ch - '0');
            } else {
                int num1 = stack.pop();
                int num2 = stack.pop();

                switch (ch) {
                    case '+':
                        stack.push(num2 + num1);
                        break;
                    case '-':
                        stack.push(num2 - num1);
                        break;
                    case '*':
                        stack.push(num2 * num1);
                        break;
                    case '/':
                        stack.push(num2 / num1);
                        break;
                }

            }

        }

        //output:
        System.out.println("Evaluation single digit expression: " + stack.pop());

    }

    public void postfixExpressionEvaluation_MultipleDigit(String expr) {

        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < expr.length(); i++) {

            char ch = expr.charAt(i);

            //space is needed in expr to distinguish b/w 2 different multiple digit
            if (ch == ' ') {
                continue;
            }

            //if we found atleat one digit
            //try to iterate i until we found a char ch which is not a digit
            if (Character.isDigit(ch)) {
                int createNum = 0;
                while (Character.isDigit(expr.charAt(i))) {
                    createNum = createNum * 10 + (expr.charAt(i) - '0');
                    i++; //this to further iterate i and find digit char 
                }
//                i--; //just to balance to one iter back
                stack.push(createNum);
            } else {
                int num1 = stack.pop();
                int num2 = stack.pop();

                switch (ch) {
                    case '+':
                        stack.push(num2 + num1);
                        break;
                    case '-':
                        stack.push(num2 - num1);
                        break;
                    case '*':
                        stack.push(num2 * num1);
                        break;
                    case '/':
                        stack.push(num2 / num1);
                        break;
                }

            }

        }

        //output:
        System.out.println("Evaluation multiple digit expression: " + stack.pop());
    }

    public void removeKDigitsToCreateSmallestNumber(String num, int K) {

        //explanation: https://youtu.be/vbM41Zql228
        if (K == num.length()) {
            System.out.println("Number formed: 0");
            return;
        }

        int index = 0;
        StringBuilder sb = new StringBuilder();
        Stack<Character> stack = new Stack<>();

        while (index < num.length()) {
            //Greedily take the smaller digit than stack's peek digit
            //ex: "1432219"
            //stack [1]
            //stack [1, 4] as not 1 > 4
            //stack [1, 3] as 4 > 3 
            //becasue if number has to be formed in that case 14 > 13 smaller num req 13 and so on.
            while (K != 0 && !stack.isEmpty() && stack.peek() > num.charAt(index)) {
                stack.pop();
                K--;
            }

            stack.push(num.charAt(index));
            index++;
        }

        //case when the all the digits in the stack are same
        //ex: 1111
        while (K != 0) {
            stack.pop();
            K--;
        }

        //form the number
        //ex: "1432219"
        //stack would be having [1, 2, 1, 9] <- peek
        //pop element and add then at 0th index so sb = "1219"
        while (!stack.isEmpty()) {
            sb.insert(0, stack.pop());
        }

        //case when there are leading zeros in num 
        //ex: 000234 => 234
        while (sb.length() > 1 && sb.charAt(0) == '0') {
            sb.deleteCharAt(0);
        }

        //output
        System.out.println("Number formed: " + sb.toString());
    }

    public void findTheMostCompetetiveSubsequenceOfSizeKFromArray(int[] arr, int K) {

        //problem: https://leetcode.com/problems/find-the-most-competitive-subsequence/
        //explanation: https://leetcode.com/problems/find-the-most-competitive-subsequence/discuss/1113429/Java-Brute-Force-Stack
        int[] result = new int[K];
        Stack<Integer> stack = new Stack<>();

        for (int i = 0; i < arr.length; i++) {

            int curr = arr[i];

            if (stack.size() != K || stack.peek() > curr) {

                int leftNums = arr.length - i;
                while (K - stack.size() < leftNums && !stack.isEmpty() && stack.peek() > curr) {
                    stack.pop();
                }
                stack.push(curr);
            }
        }

        int index = result.length - 1;
        while (!stack.isEmpty()) {
            result[index--] = stack.pop();
        }

        //output
        for (int x : result) {
            System.out.print(x + " ");
        }

        System.out.println();
    }

    private void rotAllAdjacent(int[][] basket,
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

    public void rottenOranges_DFS(int[][] basket) {

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

    public int rottenOranges_HashBased(int[][] grid) {

        Set<String> fresh = new HashSet<>();
        Set<String> rotten = new HashSet<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {

                if (grid[i][j] == 1) {
                    fresh.add(i + "" + j);
                }

                if (grid[i][j] == 2) {
                    rotten.add(i + "" + j);
                }

            }
        }

        int minTime = 0;
        int[][] dir = {
            {-1, 0},
            {1, 0},
            {0, -1},
            {0, 1}
        };

        while (fresh.size() > 0) {

            Set<String> infected = new HashSet<>();
            for (String coor : rotten) {
                int x = coor.charAt(0) - '0';
                int y = coor.charAt(1) - '0';
                for (int[] r : dir) {
                    int x_ = x + r[0];
                    int y_ = y + r[1];
                    if (fresh.contains(x_ + "" + y_)) {
                        fresh.remove(x_ + "" + y_);
                        infected.add(x_ + "" + y_);
                    }
                }
            }

            if (infected.size() == 0) {
                return -1;
            }

            rotten = infected;
            minTime++;

        }

        return minTime;
    }

    public void validSudoku(String[][] grid) {

        //Explanantion: https://youtu.be/Pl7mMcBm2b8
        HashSet<String> vis = new HashSet<>();

        for (int x = 0; x < grid.length; x++) {
            for (int y = 0; y < grid[x].length; y++) {

                String curr = grid[x][y];
                if (!curr.equals(".")) {

                    if (!vis.add(curr + " at row: " + x)
                            || !vis.add(curr + " at col: " + y)
                            || !vis.add(curr + " in cell: " + (x / 3) + "-" + (y / 3))) {
                        System.out.println("Invalid sudoku grid");
                        return;
                    }
                }
            }
        }

        System.out.println("Valid sudoku grid");

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

    public void kLargestElementInArray(int[] arr, int K) {

        PriorityQueue<Integer> minHeap = new PriorityQueue<>((o1, o2) -> o1.compareTo(o2));
        for (int x : arr) {
            minHeap.add(x);
            if (minHeap.size() > K) {
                minHeap.poll();
            }
        }

        int[] result = new int[minHeap.size()];
        int index = minHeap.size() - 1;
        while (!minHeap.isEmpty()) {

            result[index--] = minHeap.poll();

        }

        //output
        for (int x : result) {
            System.out.print(x + " ");
        }

        System.out.println();
    }

    public void mergeKSortedArrays_1(int[][] arr) {

        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int[] row : arr) {
            for (int cell : row) {
                minHeap.add(cell);
            }
        }

        List<Integer> sortedList = new ArrayList<>();
        while (!minHeap.isEmpty()) {

            sortedList.add(minHeap.poll());

        }

        //output:
        System.out.println("K sorted array into a list: " + sortedList);

    }

    public void mergeKSortedArrays_2(int[][] arr) {

        class Input {

            final int row;
            int col;
            final int colLength;

            public Input(int row, int col, int colLength) {
                this.row = row;
                this.col = col;
                this.colLength = colLength;
            }
        }

        //OPTIMISED
        //........................T: O(N*K*LogK)
        //........................S: O(K)
        PriorityQueue<Input> minHeap = new PriorityQueue<>((a, b) -> arr[a.row][a.col] - arr[b.row][b.col]);
        //minHeap will hold start coordinate(row, col) for all the elements in each row not the total R*C elements directly
        for (int r = 0; r < arr.length; r++) {
            if (arr[r].length > 0) {
                minHeap.add(new Input(r, 0, arr[r].length));
            }
        }
        //after this loop minHeap will have K instance of Input() holding (row, col)  for each row
        //minHeap.size() == K

        List<Integer> sortedList = new ArrayList<>();
        while (!minHeap.isEmpty()) {

            //At any point of time we poll Input from minHeap
            Input in = minHeap.poll();
            //we put the element at that row, col
            sortedList.add(arr[in.row][in.col]);
            //if new col is less than its col length
            //we update col and put that updated input(in) in minHeap back
            if (in.col + 1 < in.colLength) {
                in.col++;
                minHeap.add(in);
            }
        }

        //output:
        System.out.println("K sorted array into a list: " + sortedList);

    }

    public void kThLargestSumFromContigousSubarray(int[] arr, int K) {

        //arr[]: [20, -5, -1]
        //contSumSubarry: [20, 15, 14, -5, -6, -1]
        //20, 20+(-5), 20+(-5)+(-1), -5, -5+(-1), -1 
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        List<Integer> contSumSubarry = new ArrayList<>();
        //................................T: O(N^2)
        for (int i = 0; i < arr.length; i++) {
            contSumSubarry.add(arr[i]);
            int contSum = arr[i];
            for (int j = i + 1; j < arr.length; j++) {
                contSum += arr[j];
                contSumSubarry.add(contSum);
            }
        }

        //...............................T: O(LogK)
        for (int sum : contSumSubarry) {
            minHeap.add(sum);
            if (minHeap.size() > K) {
                minHeap.poll();
            }
        }

        System.out.println("kth largest sum from contigous subarray: " + minHeap.peek());
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

    private int findFirstOccurenceKInSortedArray(int[] arr, int K, int low, int high, int N) {
        if (high >= low) {
            int mid = low + (high - low) / 2;
            if ((mid == 0 || K > arr[mid - 1]) && arr[mid] == K) {
                return mid;
            } else if (K > arr[mid]) {
                return findFirstOccurenceKInSortedArray(arr, K, (mid + 1), high, N);
            } else {
                return findFirstOccurenceKInSortedArray(arr, K, low, (mid - 1), N);
            }
        }
        return -1;
    }

    private int findLastOccurenceKInSortedArray(int[] arr, int K, int low, int high, int N) {
        if (high >= low) {
            int mid = low + (high - low) / 2;
            if ((mid == N - 1 || K < arr[mid + 1]) && arr[mid] == K) {
                return mid;
            } else if (K < arr[mid]) {
                return findLastOccurenceKInSortedArray(arr, K, low, (mid - 1), N);
            } else {
                return findLastOccurenceKInSortedArray(arr, K, (mid + 1), high, N);
            }
        }
        return -1;
    }

    public void findFirstAndLastOccurenceOfKInSortedArray(int[] arr, int K) {

        int N = arr.length;
        int first = findFirstOccurenceKInSortedArray(arr, K, 0, N - 1, N);
        int last = findLastOccurenceKInSortedArray(arr, K, 0, N - 1, N);

        System.out.println(K + " first and last occurence: " + first + " " + last);

    }

    public int searchInRotatedSortedArray(int[] arr, int K) {

        int f = 0;
        int l = arr.length - 1;
        int N = arr.length;
        int mid = -1;

        while (l >= f) {

            mid = f + (l - f) / 2;
            if (arr[mid] == K) {
                return mid;
            }

            if (arr[f] <= arr[mid]) {

                if (K >= arr[f] && K < arr[mid]) {
                    l = mid - 1;
                } else {
                    f = mid + 1;
                }

            } else {
                if (K > arr[mid] && K <= arr[l]) {
                    f = mid + 1;
                } else {
                    l = mid - 1;
                }
            }

        }

        return -1;

    }

    public int searchInRotatedSortedArrayWithDuplicateArrayElement(int[] arr, int K) {

        int f = 0;
        int l = arr.length - 1;
        int N = arr.length;
        int mid = -1;

        while (l >= f) {

            //shift front till arr[f] is same as arr[f+1]
            //when f and f+1 elements are diff loop will end
            while (f < l && arr[f] == arr[f + 1]) {
                f++;
            }
            //shift last till arr[l] is same as arr[l-1]
            //when l and l-1 elements are diff loop will end
            while (f < l && arr[l] == arr[l - 1]) {
                l--;
            }

            mid = f + (l - f) / 2;
            if (arr[mid] == K) {
                return mid;
            }

            if (arr[f] <= arr[mid]) {

                if (K >= arr[f] && K < arr[mid]) {
                    l = mid - 1;
                } else {
                    f = mid + 1;
                }

            } else {
                if (K > arr[mid] && K <= arr[l]) {
                    f = mid + 1;
                } else {
                    l = mid - 1;
                }
            }

        }

        return -1;

    }

    public void findRepeatingAndMissingInUnsortedArray_1(int[] arr) {

        //problem statement: https://www.geeksforgeeks.org/find-a-repeating-and-a-missing-number/
        //arr: will be of size N and elements in arr[] will be [1..N]
        //.......................T: O(N)
        //.......................S: O(N)
        System.out.println("Approach 1");
        int[] count = new int[arr.length + 1];
        //get the occurence of arr element in count[] where count[i] i: elements in arr
        for (int i = 0; i < arr.length; i++) {
            count[arr[i]]++;
        }

        for (int i = 1; i < count.length; i++) {
            //first ith index that has count[i] = 0 is the element in arr which is supposed to be missing
            //count[i] == 0 => i = element in arr is supposed to be missing
            if (count[i] == 0) {
                System.out.println("Missing: " + i);
                break;
            }
        }

        for (int i = 1; i < count.length; i++) {
            //first ith index which has count[i] > 1 (occuring more that 1)
            //is the element which is repeating
            //count[i] > 1 => i = element in arr which is repeating
            if (count[i] > 1) {
                System.out.println("Repeating: " + i);
                break;
            }
        }

    }

    public void findRepeatingAndMissingInUnsortedArray_2(int[] arr) {

        //problem statement: https://www.geeksforgeeks.org/find-a-repeating-and-a-missing-number/
        ////explanation: https://youtu.be/aMsSF1Il3IY
        //OPTIMISED
        //.......................T: O(N)
        //.......................S: O(1)
        System.out.println("Approach 2");
        System.out.println("Repeating element: ");
        for (int i = 0; i < arr.length; i++) {
            int absVal = Math.abs(arr[i]);
            if (arr[absVal - 1] > 0) {
                arr[absVal - 1] = -arr[absVal - 1];
            } else {
                System.out.println(absVal);
            }
        }

        System.out.println("Missing element: ");
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > 0) {
                System.out.println(i + 1);
            }
        }

    }

    public boolean checkIfPairPossibleInArrayHavingGivenDiff(int[] arr, int diff) {

        //..................T; O(N)
        //..................S: O(N)
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < arr.length; i++) {
            //arr[x] - arr[y] = diff
            //arr[x] = diff + arr[y]
            //if set.contains(arr[y]) then pair is possible
            if (set.contains(arr[i])) {
                return true;
            }

            //arr[x] = arr[y] +diff
            set.add(arr[i] + diff);

        }

        return false;

    }

    private double squareRootOfANumber_BinarySearch(double n, double f, double l) {

        if (l > f) {

            double mid = f + (l - f) / 2.0;
            double sqr = mid * mid;

            if (sqr == n || Math.abs(n - sqr) < 0.00001) {
                return mid;
            } else if (sqr < n) {
                return squareRootOfANumber_BinarySearch(n, mid, l);
            } else {
                return squareRootOfANumber_BinarySearch(n, f, mid);
            }

        }

        return 1.0;

    }

    public double squareRootOfANumber(double n) {

        if (n == 0.0 || n == 1.0) {
            return n;
        }

        double i = 1;
        while (true) {
            double sqr = i * i;
            if (sqr == n) {
                return i;
            } else if (sqr > n) {
                //at this point where sqr of i is > n then that means sqr root for n lies b/w 
                // i-1 and i
                //ex sqrt(3) == 1.73 (lie b/w 1 and 2)
                // i = 1 sqr = 1*1 = 1
                //i = 2 sqr = 2*2 = 4
                //4 > n i.e 4 > 3 that means sqrt(3) lie in b/w 1 and 2
                //so we will do binary search i-1, i (1, 2)
                double res = squareRootOfANumber_BinarySearch(n, i - 1, i);
                return res;
            }
            i++;
        }
    }

    public void kThElementInTwoSortedArrays_1(int[] a, int[] b, int K) {

        //HEAP SORTING AND MERGING
        //...........................T: O(N.LogN)
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int a_ : a) {
            minHeap.add(a_);
        }

        for (int b_ : b) {
            minHeap.add(b_);
        }

        //pop out K element from heap
        int kthElement = -1;
        while (K-- != 0 && !minHeap.isEmpty()) {
            kthElement = minHeap.poll();
        }

        //output:
        System.out.println("Kth element in two sorted arrays: " + kthElement);

    }

    public void kThElementInTwoSortedArrays_2(int[] a, int[] b, int K) {

        //Two arrays are already sorted
        //...........................T: O(K)
        int iK = 0;
        int m = a.length;
        int n = b.length;
        int i = 0;
        int j = 0;
        int kthElement = -1;

        while (i < m && i < n) {

            if (a[i] < b[j]) {
                iK++;
                if (iK == K) {
                    kthElement = a[i];
                    break;
                }
                i++;
            } else {
                iK++;
                if (iK == K) {
                    kthElement = b[j];
                    break;
                }
                j++;
            }

        }

        //if loop ends beacuse we run out of one array element
        while (i < m) {
            iK++;
            if (iK == K) {
                kthElement = a[i];
            }
            i++;
        }

        while (j < n) {
            iK++;
            if (iK == K) {
                kthElement = b[j];
            }
            j++;
        }

        //output:
        System.out.println("Kth element in two sorted arrays: " + kthElement);

    }

    public void findMinimumInRotatedSortedArray(int[] arr) {

        //explanation: https://youtu.be/IzHR_U8Ly6c
        int n = arr.length;
        int start = 0;
        int end = n - 1;
        int minElement = Integer.MIN_VALUE;
        while (end >= start) {
            int mid = start + (end - start) / 2;

            if (mid > 0 && arr[mid] < arr[mid - 1]) {
                minElement = arr[mid];
                break;
            } else if (arr[start] <= arr[mid] && arr[mid] > arr[end]) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }

        System.out.println("Minimum in rotated sorted array: " + (minElement == Integer.MIN_VALUE ? arr[start] : minElement));

    }

    public void findLocalMinima(int[] arr) {

        int start = 0;
        int end = arr.length - 1;
        int mid = 0;
        while (end >= start) {

            mid = start + (end - start) / 2;

            if ((mid == 0 || arr[mid - 1] > arr[mid])
                    && (mid == arr.length - 1 || arr[mid + 1] > arr[mid])) {
                break;
            } else if (mid > 0 && arr[mid - 1] < arr[mid]) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }

        //output
        System.out.println("Local minima at index: " + mid + " element: " + arr[mid]);

    }

    public void findLocalMaxima(int[] arr) {

        int start = 0;
        int end = arr.length - 1;
        int mid = 0;
        while (end >= start) {

            mid = start + (end - start) / 2;

            if ((mid == 0 || arr[mid - 1] < arr[mid])
                    && (mid == arr.length - 1 || arr[mid + 1] < arr[mid])) {
                break;
            } else if (mid > 0 && arr[mid - 1] > arr[mid]) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }

        //output
        System.out.println("Local maxima at index: " + mid + " element: " + arr[mid]);

    }

    private int countElementsFromSecondArrayLessOrEqualToElementInFirstArray_FindLastOccurenceOfX(int[] arr,
            int x, int start, int end) {

        //MODIFIED BINARY SERACH FOR THIS QUESTION
        //SIMILAR TO findLastOccurenecOfKInSortedArray()
        if (end >= start) {

            int mid = start + (end - start) / 2;

            if ((mid == arr.length - 1 || arr[mid] < arr[mid + 1]) && arr[mid] == x) {
                return mid;
            } else if (x < arr[mid]) {
                return countElementsFromSecondArrayLessOrEqualToElementInFirstArray_FindLastOccurenceOfX(
                        arr, x, start, mid - 1);
            } else {
                return countElementsFromSecondArrayLessOrEqualToElementInFirstArray_FindLastOccurenceOfX(
                        arr, x, mid + 1, end);
            }
        }
        return end;
    }

    public void countElementsFromSecondArrayLessOrEqualToElementInFirstArray(int[] first, int[] second) {

        /*
         Brute force : T: O(N^2)
         `1. Use 2 for loop: 
         -> i for first[] 
         ---> count = 0
         ---> j for second[]
         ------> if second[j] <= first[i]: count++
         ---> print: count
         */
        //............................T: O((M + N) * LogN) where M = first.length, N = second.length
        //............................S: O(1)
        //https://www.geeksforgeeks.org/element-1st-array-count-elements-less-equal-2nd-array/
        Arrays.sort(second);
        List<Integer> result = new ArrayList<>();
        for (int x : first) {
            int index = countElementsFromSecondArrayLessOrEqualToElementInFirstArray_FindLastOccurenceOfX(
                    second, x, 0, second.length - 1);
            result.add(index + 1);
        }

        //output
        System.out.println("Count: " + result);

    }

    private int[] KMP_PatternMatching_Algorithm_LPSArray(String pattern, int size) {

        int[] lps = new int[size];
        lps[0] = 0; //always 0th index is 0
        int i = 1;
        int j = 0;
        while (i < size) {

            if (pattern.charAt(i) == pattern.charAt(j)) {
                j++;
                lps[i] = j;
                i++;
            } else {
                // char doesn't match
                // This is tricky. Consider the example. 
                // AAACAAAA and i = 7. The idea is similar 
                // to search step. 
                if (j != 0) {
                    j = lps[j - 1];

                    // Also, note that we do not increment 
                    // i here 
                } else {
                    lps[i] = j;
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

    public int editDistance_Recursion(String s1, String s2, int m, int n) {

        //https://www.geeksforgeeks.org/edit-distance-dp-5/
        //if s1 is empty then whole s2 is to be inserted to coonert s1 to s2
        if (m == 0) {
            return n;
        }

        //if s2 is empty then whole s1 is to be deleted to convert s1 to s2
        if (n == 0) {
            return m;
        }

        //if last char of two strings matches then just move ahead one char in both
        if (s1.charAt(m - 1) == s2.charAt(n - 1)) {
            return editDistance_Recursion(s1, s2, m - 1, n - 1);
        }

        //if the char doesn't matches then take the min of below 3
        return Math.min(editDistance_Recursion(s1, s2, m, n - 1), //insert
                Math.min(editDistance_Recursion(s1, s2, m - 1, n), //delete
                        editDistance_Recursion(s1, s2, m - 1, n - 1))) // replace
                + 1;

    }

    public int editDistance_DP_Memoization(String s1, String s2) {

        //https://www.geeksforgeeks.org/edit-distance-dp-5/
        int m = s1.length();
        int n = s2.length();
        int[][] memo = new int[m + 1][n + 1];

        //base cond
        for (int x = 0; x < m + 1; x++) {
            for (int y = 0; y < n + 1; y++) {
                if (x == 0) {
                    memo[x][y] = y;
                } else if (y == 0) {
                    memo[x][y] = x;
                }
            }
        }

        for (int x = 1; x < m + 1; x++) {
            for (int y = 1; y < n + 1; y++) {
                if (s1.charAt(x - 1) == s2.charAt(y - 1)) {
                    memo[x][y] = memo[x - 1][y - 1];
                } else {
                    memo[x][y] = 1 + Math.min(memo[x][y - 1], Math.min(memo[x - 1][y], memo[x - 1][y - 1]));
                }

            }
        }

        return memo[m][n];

    }

    public int coinChange_Recursion(int[] coins, int N, int K) {

        if (K == 0) {
            return 1;
        }

        if (K < 0) {
            return 0;
        }

        if (N <= 0 && K >= 1) {
            return 0;
        }

        return (coinChange_Recursion(coins, N, K - coins[N - 1]) + coinChange_Recursion(coins, N - 1, K));

    }

    public void coinChange_DP_Memoization(int[] coins, int K) {

        int N = coins.length;

        int[][] memo = new int[N + 1][K + 1];

        //base
        for (int x = 0; x < N + 1; x++) {
            for (int y = 0; y < K + 1; y++) {
                if (x == 0) {
                    memo[x][y] = 0;
                }

                if (y == 0) {
                    memo[x][y] = 1;
                }
            }
        }

        for (int x = 1; x < N + 1; x++) {
            for (int y = 1; y < K + 1; y++) {

                if (coins[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = memo[x][y - coins[x - 1]] + memo[x - 1][y];
                }

            }
        }

        System.out.println("Possible ways to make coin change: " + memo[N][K]);

    }

    public int knapSack01_Recusrion(int W, int[] weight, int[] value, int N) {

        //if either the value[] is empty(N == 0) we will not be able to make any profit
        //or the knapSack bag don't have the capacity(W == 0) then in that case profit is 0
        if (N == 0 || W == 0) {
            return 0;
        }

        //if the weight of a product is more than the knapSack capacity(W) then
        //in that case we have to just ignore that and move to another product
        if (weight[N - 1] > W) {
            return knapSack01_Recusrion(W, weight, value, N - 1);
        }

        //we now have 2 descision to make, we have to take max of these 2 descision
        //1. we can pick up a product add its value[product] in our profit 
        //and adjust knapSack capacity(W - weight[product]) and move to another product(N-1)
        //2. we can simply ingore this product and just move to another product(N-1)
        return Math.max(
                value[N - 1] + knapSack01_Recusrion(W - weight[N - 1], weight, value, N - 1),
                knapSack01_Recusrion(W, weight, value, N - 1));

    }

    public void knapSack01_DP_Memoization(int W, int[] weight, int[] value, int N) {

        int[][] memo = new int[N + 1][W + 1];

        //base cond
        for (int x = 0; x < N + 1; x++) {
            for (int y = 0; y < W + 1; y++) {
                //No product x == N == 0
                //No knapSack capacity y == W == 0
                if (x == 0 || y == 0) {
                    memo[x][y] = 0;
                }
            }
        }

        for (int x = 1; x < N + 1; x++) {
            for (int y = 1; y < W + 1; y++) {
                if (weight[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = Math.max(
                            value[x - 1] + memo[x - 1][y - weight[x - 1]],
                            memo[x - 1][y]);
                }
            }
        }

        System.out.println("The maximum profit with given knap sack: " + memo[N][W]);

    }

    public boolean subsetSum_Recursion(int[] arr, int sum, int N) {

        //if arr is empty and sum to prove is also 0 then in that case sum = 0 possible
        //as empty arr denotes empty sub set {}, {} which default sums up as 0
        if (N == 0 && sum == 0) {
            return true;
        }

        //if arr is empty and sum to prove is a non - zero number( >= 1) then in that case this given sum can't have
        //any sub set from arr as it is already empty
        if (N == 0 && sum != 0) {
            return false;
        }

        //if arr is not empty but the element are greater than sum then that element can't be used as sub set
        //just move to next element
        if (arr[N - 1] > sum) {
            return subsetSum_Recursion(arr, sum, N - 1);
        }

        //we now have 2 descision to make, any of the 2 descision makes the subset then pick that(OR operator)
        //1. we pick an element from arr and assume that it makes the subset then that sum is also to be reduced
        //to sum - arr[N-1]
        //2. we just leave this element from the array and move to next element
        return subsetSum_Recursion(arr, sum - arr[N - 1], N - 1) || subsetSum_Recursion(arr, sum, N - 1);

    }

    public void subsetSum_DP_Memoization(int[] arr, int sum, int N) {

        boolean[][] memo = new boolean[N + 1][sum + 1];

        //base cond
        for (int x = 0; x < N + 1; x++) {
            for (int y = 0; y < sum + 1; y++) {
                //if array is empty then any given sum is not possible (except sum == 0) 
                if (x == 0) {
                    memo[x][y] = false;
                }

                //if the given sum is just 0 then it can be prove even if the arrays is empty or full
                if (y == 0) {
                    memo[x][y] = true;
                }
            }
        }

        for (int x = 1; x < N + 1; x++) {
            for (int y = 1; y < sum + 1; y++) {
                if (arr[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = memo[x - 1][y - arr[x - 1]] || memo[x - 1][y];
                }
            }
        }

        System.out.println("The sub set for the given sum is possible: " + memo[N][sum]);

    }

    public void equalsSumPartition_SubsetSum(int[] arr, int N) {

        int arrSum = 0;
        for (int ele : arr) {
            arrSum += ele;
        }

        if (arrSum % 2 == 1) {
            //if odd no equal partition is possble for the given sum
            //arr = {1,5,5,11} arrSum = 22 == even can be divided into 2 half as {11}, {1,5,5}
            //if arrSum = 23 == odd no equal partition possible
            System.out.println("The equal sum partition for the given sum is not possbile as sum of array is odd");
            return;
        }

        System.out.println("The equal sum partition for the given array is possbile: ");
        //if arrSum == even the if we can prove the sum = arrSum/2 is possible
        //then other half of the sub set is by default will be eqaul to arrSum/2
        //arrSum = 22 == sum = arrSum/2 = 11 prove {11} then other half will be {1,5,5}
        subsetSum_DP_Memoization(arr, arrSum / 2, N);

    }

    public int longestCommonSubsequence_Recursion(String s1, String s2, int m, int n) {

        if (m == 0 || n == 0) {
            return 0;
        }

        if (s1.charAt(m - 1) == s2.charAt(n - 1)) {
            return longestCommonSubsequence_Recursion(s1, s2, m - 1, n - 1) + 1;
        }

        return Math.max(longestCommonSubsequence_Recursion(s1, s2, m, n - 1),
                longestCommonSubsequence_Recursion(s1, s2, m - 1, n));

    }

    public void longestCommonSubsequence_DP_Memoization(String s1, String s2, int m, int n) {

        int[][] memo = new int[m + 1][n + 1];

        //base cond
        //if s1 is empty and s2 is non-empty String no subseq length is possible
        //if s2 is empty and s1 is non-empty Strng no subseq length is possible
        for (int[] r : memo) {
            Arrays.fill(r, 0);
        }

        for (int x = 1; x < m + 1; x++) {
            for (int y = 1; y < n + 1; y++) {
                if (s1.charAt(x - 1) == s2.charAt(y - 1)) {
                    memo[x][y] = memo[x - 1][y - 1] + 1;
                } else {
                    memo[x][y] = Math.max(memo[x][y - 1], memo[x - 1][y]);
                }
            }
        }

        System.out.println("The longest common subsequence length for the given two string is: " + memo[m][n]);

    }

    private int longestRepeatingSubsequence_Recursion_Helper(String a, String b, int m, int n) {

        if (m == 0 || n == 0) {
            return 0;
        } else if (a.charAt(m - 1) == b.charAt(n - 1) && m != n) {
            return longestRepeatingSubsequence_Recursion_Helper(a, b, m - 1, n - 1) + 1;
        }

        return Math.max(longestRepeatingSubsequence_Recursion_Helper(a, b, m, n - 1),
                longestRepeatingSubsequence_Recursion_Helper(a, b, m - 1, n));
    }

    public int longestRepeatingSubsequence_Recursion(String str, int N) {
        return longestRepeatingSubsequence_Recursion_Helper(str, str, N, N);
    }

    public void longestRepeatingSubsequence_DP_Memoization(String str) {

        int N = str.length();
        int[][] memo = new int[N + 1][N + 1];

        //base cond
        //if string length is 0 then no subseq is possible
        //here there is only one string so mem[x][y] where x == 0 OR y == 0 memo[x][y] = 0
        for (int x = 1; x < N + 1; x++) {
            for (int y = 1; y < N + 1; y++) {
                if (str.charAt(x - 1) == str.charAt(y - 1) && x != y) {
                    memo[x][y] = memo[x - 1][y - 1] + 1;
                } else {
                    memo[x][y] = Math.max(memo[x][y - 1], memo[x - 1][y]);
                }
            }
        }

        //output:
        System.out.println("Longest repeating subsequence: " + memo[N][N]);

    }

    public void longestCommonSubstring_DP_Memoization(String a, String b) {

        int m = a.length();
        int n = b.length();

        int[][] memo = new int[m + 1][n + 1];

        //base cond: if any of the string is empty then common subtring is not possible
        //x == 0 OR y == 0 : memo[0][0] = 0
        int maxLenSubstring = 0;
        for (int x = 1; x < m + 1; x++) {
            for (int y = 1; y < n + 1; y++) {
                if (a.charAt(x - 1) == b.charAt(y - 1)) {
                    memo[x][y] = memo[x - 1][y - 1] + 1;
                    maxLenSubstring = Math.max(maxLenSubstring, memo[x][y]);
                } else {
                    memo[x][y] = 0;
                }
            }
        }

        //output:
        System.out.println("Longest common substring: " + maxLenSubstring);

    }

    public int maximumLengthOfPairChain_DP_Approach(int[][] pairs) {

        //https://leetcode.com/problems/maximum-length-of-pair-chain/solution/
        //.......................T: O(N^2)
        //.......................S: O(N)
        Arrays.sort(pairs, (a, b) -> a[0] - b[0]); //T: O(N.LogN)
        int N = pairs.length;
        int[] dp = new int[N];
        Arrays.fill(dp, 1);

        //T: O(N^2)
        for (int i = 1; i < N; i++) {
            for (int j = 0; j < i; j++) {
                if (pairs[j][1] < pairs[i][0]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }

        int ans = 0;
        for (int x : dp) {
            ans = Math.max(ans, x);
        }

        //overall T: O(N^2) as, N^2 > N.LogN
        return ans;

    }

    public int maximumLengthOfPairChain_Greedy_Approach(int[][] pairs) {

        //OPTIMISED
        //https://leetcode.com/problems/maximum-length-of-pair-chain/solution/
        //........................T: O(N.LogN)
        //........................S: O(1)
        Arrays.sort(pairs, (a, b) -> a[1] - b[1]); //T: O(N.LogN)
        int curr = Integer.MIN_VALUE;
        int ans = 0;
        for (int[] pair : pairs) { //T: O(N)

            if (curr < pair[0]) {
                curr = pair[1];
                ans++;
            }

        }

        //overall T: O(N.LogN) as, N.LogN > N
        return ans;

    }

    public int findBinomialCoefficient_Recursion(int n, int r) {

        //https://www.geeksforgeeks.org/binomial-coefficient-dp-9/
        //this approach have overlapping subproblems
        //Binomial coefficient : nCr formula = n!/r!(n - r)!
        //if r = 0 OR r = n, ans: 1 as, 
        //r == 0: n!/0!.(n - 0)! => n!/n! => 1
        //r == n: n!/n!.(n - n)! => n!/n! => 1
        //0! = 1
        if (r > n) {
            return 0;
        }

        if (r == 0 || r == n) {
            return 1;
        }

        return findBinomialCoefficient_Recursion(n - 1, r - 1) + findBinomialCoefficient_Recursion(n - 1, r);

    }

    public void findBinomialCoefficient_DP_Memoization(int n, int r) {

        int[][] memo = new int[n + 1][r + 1];
        //base cond
        for (int x = 0; x < n + 1; x++) {
            for (int y = 0; y < r + 1; y++) {

                if (y > x) {
                    memo[x][y] = 0;
                } else if (y == 0 || y == x) {
                    memo[x][y] = 1;
                }

            }
        }

        for (int x = 1; x < n + 1; x++) {
            for (int y = 1; y < r + 1; y++) {
                memo[x][y] = memo[x - 1][y - 1] + memo[x - 1][y];
            }
        }

        //output:
        System.out.println("Binomial coefficient (nCr) DP way: " + memo[n][r]);

    }

    public int friendsPairingProblem_Recursion(int n) {

        //https://www.geeksforgeeks.org/friends-pairing-problem/
        //if no friend is there nothing is possible
        if (n == 0) {
            return 0;
        }

        //if 1 friend is avaialbe he can only remain single
        if (n == 1) {
            return 1;
        }

        //if 2 friends are available there can be two ways
        //friend can remain single: {2} Or can be be paired as {1,2}
        if (n == 2) {
            return 2;
        }

        //if above cond doesn't fulfil we have two choices
        //1. ether we can remain single fun(n-1)
        //2. Or we can keep ourself and check others for pair: (n-1)*fun(n-2)
        return friendsPairingProblem_Recursion(n - 1) + (n - 1) * friendsPairingProblem_Recursion(n - 2);

    }

    public void friendsPairingProblem_DP_Memoization(int n) {

        //https://www.geeksforgeeks.org/friends-pairing-problem/
        int[] memo = new int[n + 1];
        //base cond
        memo[0] = 0;
        memo[1] = 1;
        memo[2] = 2;

        for (int i = 3; i < n + 1; i++) {
            memo[i] = memo[i - 1] + (i - 1) * memo[i - 2];
        }

        //output
        System.out.println("No. ways freinds can be paired: " + memo[n]);

    }

    public int sticklerThief_Recursion(int[] houses, int n) {

        //if no houses is available
        if (n == 0) {
            return 0;
        }

        //if only one house is available
        if (n == 1) {
            return houses[n - 1];
        }

        //2 choices
        //1. we choose not to pick a house and we simply move to next house
        //2. we choose to pick that house then we have to add the amount in that house in our result and move to 
        //alternate house (which is not adjacent(n-2))
        //just choose the max of these choices
        return Math.max(sticklerThief_Recursion(houses, n - 1), houses[n - 1] + sticklerThief_Recursion(houses, n - 2));

    }

    public void sticklerThief_DP_Memoization(int[] houses) {

        int n = houses.length;
        int[] memo = new int[n + 1];

        //base cond
        memo[0] = 0; //if no house is available
        memo[1] = houses[0]; //if only one house is available

        for (int i = 2; i < memo.length; i++) {

            memo[i] = Math.max(memo[i - 1], houses[i - 1] + memo[i - 2]);

        }

        //output;
        System.out.println("The maximum amount stickler thief can pick from alternate houses: " + memo[n]);

    }

    public void longestIncreasingSubsequence_BruteForce(int[] arr, int n) {

        int maxLengthLongestIncSubseq = 0;
        for (int i = 0; i < n; i++) {
            int currSeq = arr[i];
            int currMaxLength = 1;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] > currSeq) {
                    currSeq = arr[j];
                    currMaxLength++;
                }
            }
            maxLengthLongestIncSubseq = Math.max(maxLengthLongestIncSubseq, currMaxLength);

        }

        //output:
        System.out.println("Longest inc subseq of the given array is: " + maxLengthLongestIncSubseq);

    }

    int longestIncreasingSubsequence_LongestSeqLength;

    private int longestIncreasingSubsequence_Recursion_Helper(int[] arr, int n) {

        //if there is one element in arr then the longest seq length is just one
        if (n == 1) {
            return 1;
        }

        int res = 0;
        int maxLengthHere = 1;

        for (int i = 1; i < n; i++) {

            res = longestIncreasingSubsequence_Recursion_Helper(arr, i);
            if (arr[i - 1] < arr[n - 1] && res + 1 > maxLengthHere) {
                maxLengthHere = res + 1;
            }
        }

        longestIncreasingSubsequence_LongestSeqLength = Math.max(longestIncreasingSubsequence_LongestSeqLength, maxLengthHere);

        return maxLengthHere;
    }

    public int longestIncreasingSubsequence_Recursion(int[] arr, int n) {
        longestIncreasingSubsequence_LongestSeqLength = 0;
        longestIncreasingSubsequence_Recursion_Helper(arr, n);
        return longestIncreasingSubsequence_LongestSeqLength;
    }

    public void longestIncreasingSubsequence_DP_Memoization(int[] arr, int n) {

        int[] memo = new int[n];
        //base cond
        Arrays.fill(memo, 1);

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (arr[i] > arr[j] && memo[i] < memo[j] + 1) {
                    memo[i] = memo[j] + 1;
                }
            }
        }

        int maxLengthLongestIncSubseq = 0;
        for (int i = 0; i < memo.length; i++) {
            maxLengthLongestIncSubseq = Math.max(maxLengthLongestIncSubseq, memo[i]);
        }

        //output:
        System.out.println("Longest inc subseq of the given array is: " + maxLengthLongestIncSubseq);
    }

    public void maximumLengthOfRepeatedSubarray_DP_Problem(int[] arr1, int[] arr2) {

        //Approach is similar to longest common substring
        int m = arr1.length;
        int n = arr2.length;
        int[][] memo = new int[m + 1][n + 1];

        //base cond:
        //arr1 is empty no repeated values can be checked arr2
        //similarly arr2 is empty no repeated values can be checked with arr1
        //x == 0, y == 0 will 0
        int maxLen = 0;
        for (int x = 1; x < memo.length; x++) {
            for (int y = 1; y < memo[x].length; y++) {
                if (arr1[x - 1] == arr2[y - 1]) {
                    memo[x][y] = memo[x - 1][y - 1] + 1;
                    maxLen = Math.max(maxLen, memo[x][y]);
                } else {
                    memo[x][y] = 0;
                }
            }
        }

        //output
        System.out.println("Maximum length of repeated subarray: " + maxLen);

    }

    public void nMeetingRooms_Greedy(int[] startTime, int[] finishTime) {

        int n = startTime.length;

        int[][] input = new int[n][3];
        //[n][0] = startTime[i]
        //[n][1] = finishTime[i]
        //[n][2] = i

        for (int i = 0; i < n; i++) {
            input[i][0] = startTime[i];
            input[i][1] = finishTime[i];
            input[i][2] = i;
        }

        //sort the input in incr order of finishTime i.e, input[i][1]
        Arrays.sort(input, (a, b) -> a[1] - b[1]);

        int meetingsCanBeConducted = 1; //at least one meeting can be held
        List<Integer> indexOfMeetingTimings = new ArrayList<>();
        indexOfMeetingTimings.add(input[0][2] + 1); // 1 based index
        int endTime = input[0][1];
        for (int i = 1; i < n; i++) {

            int ithStartTime = input[i][0];
            int ithEndTime = input[i][1];

            if (endTime < ithStartTime) {
                meetingsCanBeConducted++;
                endTime = ithEndTime;
                indexOfMeetingTimings.add(input[i][2] + 1);
            }

        }

        System.out.println("No. of meetings can be conducted: " + meetingsCanBeConducted);
        System.out.println("Index of meetings can be conducted: " + indexOfMeetingTimings);

    }

    public void pairsOfMoviesCanBeWatchedDuringFlightDurationK_Greedy(int[] movieLength, int K) {

        //prepare index wise data
        int[][] input = new int[movieLength.length][2];
        for (int i = 0; i < movieLength.length; i++) {

            input[i][0] = movieLength[i];
            input[i][1] = i;

        }

        //sort the input
        Arrays.sort(input, new Comparator<int[]>() {

            @Override
            public int compare(int[] a, int[] b) {
                //0th index holds movieLength and sortng asc on basis of that
                return a[0] - b[0];
            }
        });

        int n = movieLength.length;
        int l = 0;
        int h = n - 1;
        int i = 0;
        int j = 0;
        int sum = 0;
        while (l < h) {

            int firstMovie = input[l][0];
            int secondMovie = input[h][0];

            if (firstMovie + secondMovie <= K) {

                if (sum < firstMovie + secondMovie) {
                    sum = firstMovie + secondMovie;
                    i = input[l][1];
                    j = input[h][1];
                }
                l++;
            } else if (firstMovie + secondMovie > K) {
                h--;
            }
        }

        //output:
        System.out.println("Pair of movies can be watched during flight duration K: "
                + (i + "," + j) + "-"
                + (movieLength[i] + "," + movieLength[j]));

    }

    public int choclateDistribution_Greedy(int[] choclates, int students) {

        //.........................T: O(N.LogN)
        //https://www.geeksforgeeks.org/chocolate-distribution-problem/
        //if no choclates are there OR there are no student available for distribution
        if (choclates.length == 0 || students == 0) {
            return 0;
        }

        //if the no. of choclates is less than the no. of student
        if (choclates.length < students) {
            return -1;
        }

        //sort the no of choclates
        Arrays.sort(choclates);

        int minDiff = Integer.MAX_VALUE;
        for (int i = 0; i + students - 1 < choclates.length; i++) {
            int currDiff = choclates[i + students - 1] - choclates[i];
            minDiff = Math.min(minDiff, currDiff);
        }

        //output:
        return minDiff;

    }

    public void minimumPlatformNeeded_BruteForce(int[] arr, int[] dep) {

        //.......................T: O(N^2)
        int n = arr.length;

        int maxPlatform = 1;
        for (int i = 0; i < n; i++) {
            int currPlatform = 1;
            for (int j = i + 1; j < n; j++) {

                if ((arr[i] >= arr[j] && arr[i] <= dep[j])
                        || (arr[j] >= arr[i] && arr[j] <= dep[i])) {
                    currPlatform++;
                }

                maxPlatform = Math.max(maxPlatform, currPlatform);

            }
        }

        //output:
        System.out.println("max platfrm needed: " + maxPlatform);

    }

    public void minimumPlatformNeeded_Greedy(int[] arr, int[] dep) {

        //.......................T: O(N.LogN)
        int n = arr.length;

        //.................T: O(N.LogN)
        Arrays.sort(arr);
        Arrays.sort(dep);

        int i = 1;
        int j = 0;
        int maxPlatform = 1;
        int currPlatform = 1;

        while (i < n && j < n) {

            if (arr[i] <= dep[j]) {
                currPlatform++;
                i++;
            } else if (arr[i] > dep[j]) {
                currPlatform--;
                j++;
            }

            maxPlatform = Math.max(maxPlatform, currPlatform);

        }

        //output:
        System.out.println("max platfrm needed: " + maxPlatform);

    }

    public void fractionalKnapsack(int[] weight, int[] value, int W) {

        int n = weight.length;

        //prepare data\
        double[][] input = new double[n][3];
        //0: weight, 1: value, 2: costPerWeight
        for (int i = 0; i < n; i++) {

            input[i][0] = weight[i];
            input[i][1] = value[i];
            input[i][2] = (double) value[i] / (double) weight[i];

        }

        //sort the input on basis of costPerWeight desc
        Arrays.sort(input, (a, b) -> (int) (b[2] - a[2])); //2: costPerWeight

        double maxValue = 0;
        for (int i = 0; i < n; i++) {

            int currWeight = (int) input[i][0];
            int currValue = (int) input[i][1];

            if (W - currWeight >= 0) {
                W -= currWeight;
                maxValue += currValue;
            } else {

                //if we can't pick up an item as whole
                //we will calculate the fraction we can pick up
                //i.e W/currWeight
                double frac = (double) W / currWeight;
                //value per frac
                maxValue += (currValue * frac);
                //estimating balanced capacity
                W = (int) (W - (currWeight * frac));
            }
        }

        //output:
        System.out.println("Max value can be picked up by fractional knapsack: " + maxValue);

    }
    
    public void swapTwoDigitAtMostToFormAGreaterNumber_Greedy(int num){
        
        //https://leetcode.com/problems/maximum-swap/
        char[] digit = String.valueOf(num).toCharArray();
        int[] index = new int[10];
        int ans = -1;
        for(int i = 0; i < digit.length; i++){
            index[digit[i] - '0'] = i;
        }
        
        for(int  i = 0; i < digit.length; i++){
            for(int d = 9; d > digit[i] - '0'; d--){
                if(index[d] > i){
                    char temp = digit[i];
                    digit[i] = digit[index[d]];
                    digit[index[d]] = temp;
                    System.out.println("Greater number after 2 digit swaps atmost: "+Integer.parseInt(String.valueOf(digit)));
                    return;
                }
            }
        }

        //output
        System.out.println("Greater number after 2 digit swaps atmost: "+num);
        
    }

    public void graphBFSAdjList_Graph(int V, List<List<Integer>> adjList) {

        List<Integer> result = new ArrayList<>();
        if (adjList == null || adjList.size() == 0) {
            return;
        }

        //actual
        for (int i = 0; i < adjList.size(); i++) {
            System.out.print(i + ": ");
            for (int v : adjList.get(i)) {
                System.out.print(v + " ");
            }
            System.out.println();
        }

        int sourceVertex = 0; // source point
        Queue<Integer> queue = new LinkedList<>();
        queue.add(sourceVertex); //source point
        boolean[] visited = new boolean[V];

        while (!queue.isEmpty()) {

            int node = queue.poll();
            visited[node] = true;
            result.add(node);
            List<Integer> childrens = adjList.get(node);
            if (childrens != null && childrens.size() > 0) {
                for (int vertex : childrens) {
                    if (visited[vertex] != true) {
                        queue.add(vertex);
                    }
                }
            }

        }

        //output:
        System.out.println("BFS of graph: " + result);

    }

    public void graphDFSAdjList_Graph(int V, List<List<Integer>> adjList) {

        List<Integer> result = new ArrayList<>();
        if (adjList == null || adjList.size() == 0) {
            return;
        }

        //actual
        for (int i = 0; i < adjList.size(); i++) {
            System.out.print(i + ": ");
            for (int v : adjList.get(i)) {
                System.out.print(v + " ");
            }
            System.out.println();
        }

        int sourceVertex = 0; //source point
        Stack<Integer> stack = new Stack<>();
        stack.add(sourceVertex); //source point
        boolean[] visited = new boolean[V];

        while (!stack.isEmpty()) {

            int node = stack.pop();
            visited[node] = true;
            result.add(node);
            List<Integer> childrens = adjList.get(node);
            if (childrens != null && childrens.size() > 0) {
                for (int childVertex : childrens) {
                    if (visited[childVertex] != true) {
                        stack.push(childVertex);
                    }
                }
            }
        }

        //output:
        System.out.println("DFS of graph: " + result);

    }

    private void graphDFSAdjList_Recursive_Helper(List<List<Integer>> adjList, int vertex,
            boolean[] visited, List<Integer> result) {

        visited[vertex] = true;
        result.add(vertex);
        List<Integer> childrens = adjList.get(vertex);
        for (int childVertex : childrens) {
            if (visited[childVertex] != true) {
                graphDFSAdjList_Recursive_Helper(adjList, childVertex, visited, result);
            }
        }

    }

    public void graphDFSAdjList_Recursive_Graph(int V, List<List<Integer>> adjList) {
        List<Integer> result = new ArrayList<>();
        boolean[] visited = new boolean[V];
        int sourceVertex = 0; //source point
        graphDFSAdjList_Recursive_Helper(adjList, sourceVertex, visited, result);
        System.out.println("DFS using recursion: " + result);
    }

    private void findPathRatInMaze_Helper(int[][] m, int n, int x, int y,
            StringBuilder sb, ArrayList<String> output) {

        if (x < 0 || x >= n || y < 0 || y >= n || m[x][y] == 0) {
            return;
        }

        if (x == n - 1 && y == n - 1 && m[x][y] == 1) {
            output.add(sb.toString());
            return;
        }

        int original = m[x][y];
        m[x][y] = 0;

        //Down
        sb.append("D");
        findPathRatInMaze_Helper(m, n, x + 1, y, sb, output);
        sb.deleteCharAt(sb.length() - 1);

        //Right
        sb.append("R");
        findPathRatInMaze_Helper(m, n, x, y + 1, sb, output);
        sb.deleteCharAt(sb.length() - 1);

        //Left
        sb.append("L");
        findPathRatInMaze_Helper(m, n, x, y - 1, sb, output);
        sb.deleteCharAt(sb.length() - 1);

        //Up
        sb.append("U");
        findPathRatInMaze_Helper(m, n, x - 1, y, sb, output);
        sb.deleteCharAt(sb.length() - 1);

        m[x][y] = original;

    }

    public void findPathRatInMaze_Graph(int[][] m, int n) {
        ArrayList<String> output = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        findPathRatInMaze_Helper(m, n, 0, 0, sb, output);
        System.out.println("All possible paths: " + output);
    }

    private void numberOfIslands_Helper(int[][] grid, int x, int y,
            int[][] dir, boolean[][] visited) {

        if (x < 0 || x >= grid.length || y < 0 || y >= grid[x].length
                || grid[x][y] == 0 || visited[x][y] == true) {
            return;
        }

        visited[x][y] = true;
        for (int i = 0; i < dir.length; i++) {

            int x_ = x + dir[i][0];
            int y_ = y + dir[i][1];
            numberOfIslands_Helper(grid, x_, y_, dir, visited);

        }
    }

    public void numberOfIslands_Graph(int[][] grid) {
        int[][] dir = {
            {-1, -1},
            {-1, 0},
            {-1, 1},
            {0, -1},
            {0, 1},
            {1, -1},
            {1, 0},
            {1, 1}
        };
        boolean[][] visited = new boolean[grid.length][grid[0].length];
        int islandCount = 0;
        for (int x = 0; x < grid.length; x++) {
            for (int y = 0; y < grid[x].length; y++) {

                if (grid[x][y] == 1 && visited[x][y] != true) {
                    islandCount++;
                    numberOfIslands_Helper(grid, x, y, dir, visited);
                }

            }
        }

        System.out.println("Number of separated islands: " + islandCount);

    }

    private boolean detectCycleInUndirectedGraphDFS_Helper(List<List<Integer>> adjList, int vertex, int parent, boolean[] visited) {
        visited[vertex] = true;

        List<Integer> childrens = adjList.get(vertex);
        for (int childVertex : childrens) {
            if (visited[childVertex] != true) {
                if (detectCycleInUndirectedGraphDFS_Helper(adjList, childVertex, vertex, visited)) {
                    return true;
                }
            } else if (childVertex != parent) {
                return true;
            }
        }
        return false;
    }

    public boolean detectCycleInUndirectedGraphDFS_Graph(int V, List<List<Integer>> adjList) {

        boolean[] visited = new boolean[V];
        for (int u = 0; u < V; u++) {
            if (visited[u] != true) {
                if (detectCycleInUndirectedGraphDFS_Helper(adjList, u, -1, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    private void topologicalSort_Helper(List<List<Integer>> adjList, int vertex, boolean[] visited, Stack<Integer> resultStack) {

        visited[vertex] = true;
        List<Integer> childrens = adjList.get(vertex);
        for (int childVertex : childrens) {
            if (visited[childVertex] != true) {
                topologicalSort_Helper(adjList, childVertex, visited, resultStack);
            }
        }

        resultStack.push(vertex);

    }

    public void topologicalSort_Graph(int V, List<List<Integer>> adjList) {

        Stack<Integer> resultStack = new Stack<>();
        boolean[] visited = new boolean[V];
        for (int u = 0; u < V; u++) {
            if (visited[u] != true) {
                topologicalSort_Helper(adjList, u, visited, resultStack);
            }
        }

        System.out.println("Topological sort: ");
        while (!resultStack.isEmpty()) {
            System.out.print(resultStack.pop() + " ");
        }
        System.out.println();
    }

    public boolean detectCycleInDirectedGraphDFS_Helper(List<List<Integer>> adjList, int vertex,
            boolean[] visited, boolean[] recurStack) {

        if (recurStack[vertex]) {
            return true;
        }

        if (visited[vertex]) {
            return false;
        }

        recurStack[vertex] = true;
        visited[vertex] = true;

        List<Integer> childrens = adjList.get(vertex);
        if (childrens != null && childrens.size() > 0) {
            for (int childVertex : childrens) {
                if (detectCycleInDirectedGraphDFS_Helper(adjList, childVertex, visited, recurStack)) {
                    return true;
                }
            }
        }
        recurStack[vertex] = false;
        return false;

    }

    public boolean detectCycleInDirectedGraphDFS_Graph(int V, List<List<Integer>> adjList) {
        boolean[] visited = new boolean[V];
        boolean[] recurStack = new boolean[V];
        for (int u = 0; u < V; u++) {
            if (detectCycleInDirectedGraphDFS_Helper(adjList, u, visited, recurStack)) {
                return true;
            }
        }
        return false;
    }

    public void floodFill_Helper(int[][] image, int srcR, int srcC,
            int srcColor, int newColor, boolean[][] visited) {

        //bounds check
        if (srcR < 0 || srcR >= image.length || srcC < 0 || srcC >= image[srcR].length
                || image[srcR][srcC] != srcColor
                || visited[srcR][srcC] == true) {
            return;
        }

        //mark it as visited first
        visited[srcR][srcC] = true;

        //do dfs in 4 adjacent dir
        //UP
        floodFill_Helper(image, srcR - 1, srcC, srcColor, newColor, visited);

        //DOWN
        floodFill_Helper(image, srcR + 1, srcC, srcColor, newColor, visited);

        //LEFT
        floodFill_Helper(image, srcR, srcC - 1, srcColor, newColor, visited);

        //RIGHT
        floodFill_Helper(image, srcR, srcC + 1, srcColor, newColor, visited);

        //at this point we can say we can't go any deep in dfs
        //start the flood fill and recurse back
        if (image[srcR][srcC] == srcColor) {
            image[srcR][srcC] = newColor;
        }

        //we have reached the last srcR and srcC where we can't go any further
        //probably we have flood filled that coordinate also
        //we must reverse the state of the visited coordinate back
        visited[srcR][srcC] = false;

    }

    public void floodFill(int[][] image, int srcR, int srcC, int newColor) {

        //actual
        System.out.println();
        for (int[] r : image) {
            for (int c : r) {
                System.out.print(c + "\t");
            }
            System.out.println();
        }

        int srcColor = image[srcR][srcC];
        boolean[][] visited = new boolean[image.length][image[0].length];
        floodFill_Helper(image, srcR, srcC, srcColor, newColor, visited);

        //output
        System.out.println("output: ");
        for (int[] r : image) {
            for (int c : r) {
                System.out.print(c + "\t");
            }
            System.out.println();
        }

    }

    public boolean checkIfGivenUndirectedGraphIsBinaryTree(int V, List<List<Integer>> adjList) {

        //two condition for a undirected graph to be tree
        //1. should not have a cycle
        //2. the graph should be connected
        boolean[] visited = new boolean[V];

        //check undriected cycle
        //if all nodes is reachable via vertex 0 without forming a cycle
        //if cycle is present then its not tree and return false
        //it will also mark visited = true all those nodes that are connected to each other
        if (detectCycleInUndirectedGraphDFS_Helper(adjList, 0, -1, visited)) {
            return false;
        }

        for (int u = 0; u < V; u++) {
            if (visited[u] != true) {
                return false;
            }
        }

        return true;
    }

    private void allPathFromSourceToTargetInDirectedAcyclicGraph_Helper(Map<Integer, List<Integer>> adj, int vertex,
            int target, boolean[] vis, List<Integer> curr, List<List<Integer>> result) {

        vis[vertex] = true;
        curr.add(vertex);

        if (vertex == target) {
            result.add(curr);
        }

        List<Integer> childrens = adj.getOrDefault(vertex, new ArrayList<>());
        for (int childVertex : childrens) {
            if (vis[childVertex] != true) {
                allPathFromSourceToTargetInDirectedAcyclicGraph_Helper(adj, childVertex, target, vis, new ArrayList<>(curr), result);
            }
        }
        vis[vertex] = false;
    }

    public void allPathFromSourceToTargetInDirectedAcyclicGraph(int[][] graph) {

        //https://leetcode.com/problems/all-paths-from-source-to-target/
        List<List<Integer>> result = new ArrayList<>();

        if (graph[0].length == 0 && graph[1].length == 0) {
            return;
        }

        //prepare data
        Map<Integer, List<Integer>> adj = new HashMap<>();
        for (int u = 0; u < graph.length; u++) {
            adj.putIfAbsent(u, new ArrayList<>());
            for (int nodes : graph[u]) {
                adj.get(u).add(nodes);
            }
        }

        int source = 0;
        int target = adj.size() - 1; //target = V - 1
        boolean[] vis = new boolean[graph.length];

        allPathFromSourceToTargetInDirectedAcyclicGraph_Helper(adj, source, target, vis, new ArrayList<>(), result);

        //output:
        System.out.println("All paths: " + result);

    }

    private boolean checkIfPathExistsFromSourceToDestination_DFS(int[][] grid, int x, int y) {

        if (x < 0 || x >= grid.length || y < 0 || y >= grid[x].length
                || grid[x][y] == 0) {
            return false;
        }

        if (grid[x][y] == 2) {
            return true;
        }

        int original = grid[x][y];
        grid[x][y] = 0;

        boolean anyPathPossible
                = checkIfPathExistsFromSourceToDestination_DFS(grid, x - 1, y)
                || checkIfPathExistsFromSourceToDestination_DFS(grid, x + 1, y)
                || checkIfPathExistsFromSourceToDestination_DFS(grid, x, y - 1)
                || checkIfPathExistsFromSourceToDestination_DFS(grid, x, y + 1);

        grid[x][y] = original;

        return anyPathPossible;

    }

    public boolean checkIfPathExistsFromSourceToDestination(int[][] grid) {

        int r = grid.length;
        int c = grid[0].length;

        for (int x = 0; x < r; x++) {
            for (int y = 0; y < c; y++) {
                if (grid[x][y] == 1 && checkIfPathExistsFromSourceToDestination_DFS(grid, x, y)) {
                    return true;
                }
            }
        }

        return false;
    }

    public void minimumCostToFillGivenBag_DP_Memoization(int[] cost, int W) {

        //0-1Knapsack problem
        //problem statement: https://practice.geeksforgeeks.org/problems/minimum-cost-to-fill-given-weight-in-a-bag1956/1
        //create normal data
        List<Integer> value = new ArrayList<>();
        List<Integer> weight = new ArrayList<>();

        int actualSize = 0;
        for (int i = 0; i < cost.length; i++) {
            if (cost[i] != -1) {
                value.add(cost[i]);
                weight.add(i + 1);
                actualSize++;
            }
        }

        int[][] memo = new int[actualSize + 1][W + 1];
        for (int x = 0; x < actualSize + 1; x++) {
            for (int y = 0; y < W + 1; y++) {
                if (x == 0) {
                    memo[x][y] = Integer.MAX_VALUE;
                }
                if (y == 0) {
                    memo[x][y] = 0;
                }
            }
        }

        for (int x = 1; x < actualSize + 1; x++) {
            for (int y = 1; y < W + 1; y++) {
                if (weight.get(x - 1) > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = Math.min(value.get(x - 1) + memo[x][y - weight.get(x - 1)],
                            memo[x - 1][y]);
                }
            }
        }

        //output
        System.out.println("Min cost: " + memo[actualSize][W]);

    }

    public void LRUCacheDesignImpl(List<String> operations, List<List<Integer>> inputs) {

        LRUCacheDesign lruObj = null;
        for (int i = 0; i < operations.size(); i++) {
            String operation = operations.get(i);
            switch (operation) {
                case "LRUCache":
                    int capacity = inputs.get(i).get(0);
                    lruObj = new LRUCacheDesign(capacity);
                    System.out.println("Object created:" + operation);
                    break;
                case "put":
                    int key = inputs.get(i).get(0);
                    int value = inputs.get(i).get(1);
                    lruObj.put(key, value);
                    System.out.println("Put: " + key + " " + value);
                    break;
                case "get":
                    key = inputs.get(i).get(0);
                    System.out.println("Get: " + lruObj.get(key));
                    break;
            }
        }

    }

    //MY AMAZON ONLINE ASSESSMENT
    public boolean robotRodeo(String command) {

        //initial state of robot
        int x = 0;
        int y = 0;
        int dir = 0; //0 = North, 1 = East, 2 = South, 3 = West (in clockwise direction of actual N-E-S-W dir)
        for (char move : command.toCharArray()) {

            if (move == 'R') {
                //for a given curr dir, R would be on +1 side in clockwise way
                //where %4 will bound our dir upto 4 actual dir(N-E-S-W dir)
                dir = (dir + 1) % 4;
            } else if (move == 'L') {
                //for a given curr dir, L would be on +3 side in clockwise way
                //where %4 will bound our dir upto 4 actual dir(N-E-S-W dir)
                dir = (dir + 3) % 4;
            } else {
                //now move = G
                if (dir == 0) {
                    //North and G, move robot to vertical up(or to north) from given (x,y)
                    y++;
                } else if (dir == 1) {
                    //East and G, move robot to horizontal right(or to east side) from given (x,y)
                    x++;
                } else if (dir == 2) {
                    //South and G, move robot to vertical down(or to south side) from given (x,y)
                    y--;
                } else { //dir == 3
                    //West and G, move robot to horizontal left(or to west side) from given (x,y)
                    x--;
                }

            }
        }

        //if the robot returned to same inital (x,y) = (0,0) after all moves in given command
        //that means there exists a cycle
        return dir != 0 || (x == 0 && y == 0);
    }

    public
            int swapsRequiredToSortArray(int[] arr) {

        class Pair {

            int element;
            int index;

            public Pair(int element, int index) {
                this.element = element;
                this.index = index;
            }
        }
        int N = arr.length;
        List<Pair> list = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            list.add(new Pair(arr[i], i));
        }

        Collections.sort(list, (a, b) -> a.element - b.element);

        int result = 0;

        for (int i = 0; i < N; i++) {

            if (list.get(i).index == i) {
                continue;
            } else {

                int index = list.get(i).index;
                Pair p = list.get(index);
                list.set(index, list.get(i));
                list.set(i, p);
            }

            if (i != list.get(i).index) {
                i--;
            }

            result++;

        }

        return result;
    }

    public void checkBinaryNumberStreamIsDivisibleByN(int[] binaryStream, int N) {

        //explanantion: https://www.geeksforgeeks.org/check-divisibility-binary-stream/
        //see method 2
        /*
        
         formula: 
         if bit is 1 new decimal = 2 * prevDecimal + 1
         if bit is 0 new decimal = 2 * prevDecimal
        
         Ex:
         binaryStream = [1,0,1,0,1]
         prevDecimal = 0
         binaryFormed = ""
         i = 0
         bit = 1 -> binaryFormed.append(bit) = "1" == actualDecimal(binaryFormed) = 1
         if bit == 1: prevDecimal = (2 * prevDecimal) + 1
         ---> prevDecimal = 2 * 0 + 1 = 1
        
         i = 1
         bit = 0 -> binaryFormed.append(bit) = "10" == actualDecimal(binaryFormed) = 2
         if bit == 0: prevDecimal = (2 * prevDecimal)
         ---> prevDecimal = 2 * 1 = 2
        
         i = 2
         bit = 1 -> binaryFormed.append(bit) = "101" == actualDecimal(binaryFormed) = 5
         if bit == 1: prevDecimal = (2 * prevDecimal) + 1
         ---> prevDecimal = 2 * 2 + 1= 5
        
         so on...
         */
        int remainder = 0;
        int decimal = 0;
        StringBuilder sb = new StringBuilder(); //just for output purpose, not necessary to use
        for (int bit : binaryStream) {

            if (bit == 0) {
                remainder = (2 * remainder) % N;
                decimal = 2 * decimal;
            } else if (bit == 1) {
                remainder = (2 * remainder + 1) % N;
                decimal = 2 * decimal + 1;
            }

            sb.append(bit);
            if (remainder == 0) {
                System.out.println("Binary formed: " + sb.toString() + " dec("+decimal+") is divisible by " + N);
            } else {
                System.out.println("Binary formed: " + sb.toString() + " dec("+decimal+") is not divisible by " + N);
            }
        }
    }

    private String convertNumberToWords_Helper(int n, String suff) {

        // Strings at index 0 is not used, it is to make array 
        // indexing simple 
        String one[] = {"", "one ", "two ", "three ", "four ",
            "five ", "six ", "seven ", "eight ",
            "nine ", "ten ", "eleven ", "twelve ",
            "thirteen ", "fourteen ", "fifteen ",
            "sixteen ", "seventeen ", "eighteen ",
            "nineteen "};

        // Strings at index 0 and 1 are not used, they is to 
        // make array indexing simple 
        String ten[] = {"", "", "twenty ", "thirty ", "forty ",
            "fifty ", "sixty ", "seventy ", "eighty ",
            "ninety "};

        String str = "";
        // if n is more than 19, divide it 
        if (n > 19) {
            str += ten[n / 10] + one[n % 10];
        } else {
            str += one[n];
        }

        // if n is non-zero 
        if (n != 0) {
            str += suff;
        }

        return str;
    }

    public void convertNumberToWords(long n) {

        //https://www.geeksforgeeks.org/program-to-convert-a-given-number-to-words-set-2/
        StringBuilder sb = new StringBuilder();

        // handles digits at ten millions and hundred 
        // millions places (if any) 
        sb.append(convertNumberToWords_Helper((int) (n / 1000_000_0) % 100, "crore "));

        // handles digits at hundred thousands and one 
        // millions places (if any) 
        sb.append(convertNumberToWords_Helper((int) ((n / 100_000) % 100), "lakh "));

        // handles digits at thousands and tens thousands 
        // places (if any) 
        sb.append(convertNumberToWords_Helper((int) ((n / 1000) % 100), "thousand "));

        // handles digit at hundreds places (if any) 
        sb.append(convertNumberToWords_Helper((int) ((n / 100) % 10), "hundred "));

        if (n > 100 && n % 100 > 0) {
            sb.append("and ");
        }

        // handles digits at ones and tens places (if any) 
        sb.append(convertNumberToWords_Helper((int) (n % 100), ""));

        //output
        System.out.println("In words: \n" + sb.toString());

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
//        System.out.println("Level order traversal of tree iterative & recursive");
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
//        obj.levelOrderTraversal_Recursive(root1);
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
//        System.out.println("Longest common subsequence");
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
//        obj.removeConsecutiveDuplicateInString("aaaaaaaaaaaa");
        //......................................................................
//        Row: 108
//        System.out.println("Majority Element");
//        obj.majorityElement_1(new int[] { 1, 3, 3, 1, 2 });
//        obj.majorityElement_1(new int[] { 1, 3, 3, 3, 2 });
//        obj.majorityElement_2(new int[] { 1, 3, 3, 1, 2 }); //MOORE'S VOTING ALGO
//        obj.majorityElement_2(new int[] { 1, 3, 3, 3, 2 }); //MOORE'S VOTING ALGO
        //......................................................................
//        Row: 195
//        System.out.println("Convert tree to its sum tree");
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
//        //https://leetcode.com/problems/path-sum-iii/
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
//        Row: 349, 269
//        System.out.println("Min cost to combine ropes of diff lengths into one big rope");
//        obj.minCostOfRope(new int[]{4, 3, 2, 6});
        //......................................................................
//        Row: 344, 89, 271
//        System.out.println("Reorganise string");
//        //https://leetcode.com/problems/reorganize-string/
//        System.out.println("Reorganise string output: "+obj.reorganizeString("aab"));
//        System.out.println("Reorganise string output: "+obj.reorganizeString("aaab"));
//        System.out.println("Reorganise string output: "+obj.reorganizeString("bbbbb"));
//        System.out.println("Reorganise string output: "+obj.reorganizeString("geeksforgeeks"));
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
//        obj.longestPrefixAlsoSuffixInString_KMPAlgo("aabcavefaabca");
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
//        System.out.println("Lowest common ancestor of two given node/ node values for binary tree and binary search tree both");
//        TreeNode<Integer> root1 = new TreeNode<>(5);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode<>(3));
//        root1.getLeft().setRight(new TreeNode<>(4));
//        obj.lowestCommonAncestorOfTree(root1, 3, 4);
//        root1 = new TreeNode<>(5);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode<>(3));
//        root1.getLeft().setRight(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(6));
//        obj.lowestCommonAncestorOfTree(root1, 3, 6);
//        //CASE OF BST
//        root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.lowestCommonAncestorOfTree(root1, 0, 5);
//        root1 = new TreeNode<>(5);
//        root1.setLeft(new TreeNode(4));
//        root1.getLeft().setLeft(new TreeNode<>(3));
//        root1.setRight(new TreeNode(6));
//        root1.getRight().setRight(new TreeNode(7));
//        root1.getRight().getRight().setRight(new TreeNode(8));
//        obj.lowestCommonAncestorOfTree(root1, 7, 8);
        //......................................................................
//        Row: 69, 416
//        System.out.println("Edit distance recursion/ DP memoization");
//        String s1 = "sunday";
//        String s2 = "saturday";
//        System.out.println("Edit distance recursion: "+obj.editDistance_Recursion(s1, s2, s1.length(), s2.length()));
//        System.out.println("Edit distance dp memoization: "+obj.editDistance_DP_Memoization(s1, s2));
        //......................................................................
//        Row: 84
//        System.out.println("Second most occuring word in list");
//        obj.secondMostOccuringWordInStringList(new String[]{"aaa", "bbb", "ccc", "bbb", "aaa", "aaa"});
        //......................................................................
//        Row: 101
//        System.out.println("Find first and last occurence of K in sorted array");
//        obj.findFirstAndLastOccurenceOfKInSortedArray(new int[]{1, 3, 5, 5, 5, 5, 67, 123, 125}, 5);
//        obj.findFirstAndLastOccurenceOfKInSortedArray(new int[]{1, 3, 5, 5, 5, 5, 67, 123, 125}, 9);
        //......................................................................
//        Row: 141, 143
//        System.out.println("Detect and print starting node of a loop cycle in linked list 2 approaches");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(node.getNext().getNext()); //Node 5 connects to Node 3
//        System.out.println("Is there a loop in linked list: "+obj.detectLoopCycleInLinkedList_HashBased(node));
//        System.out.println("Is there a loop in linked list: "+obj.detectLoopCycleInLinkedList_Iterative(node)); //T: O(N), S: O(1) //OPTIMISED
//        node = new Node<>(3);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(0));
//        node.getNext().getNext().setNext(new Node<>(-4));
//        node.getNext().getNext().getNext().setNext(node.getNext());
//        System.out.println("Is there a loop in linked list: "+obj.detectLoopCycleInLinkedList_HashBased(node));
//        System.out.println("Is there a loop in linked list: "+obj.detectLoopCycleInLinkedList_Iterative(node)); //T: O(N), S: O(1) //OPTIMISED
        //......................................................................
//        Row: 142
//        System.out.println("Detect and remove loop cycle in linked list 2 approaches");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(node.getNext().getNext()); //Node 5 connects to Node 3
//        obj.detectAndRemoveLoopCycleInLinkedList_HashBased(node);
//        node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(node.getNext().getNext()); //Node 5 connects to Node 3
//        obj.detectAndRemoveLoopCycleInLinkedList_Iterative(node); //OPTIMISED
        //......................................................................
//        Row: 145
//        System.out.println("Remove duplicates element in unsorted linked list 2 different outputs");
//        Node<Integer> node = new Node<>(3);
//        node.setNext(new Node<>(4));
//        node.getNext().setNext(new Node<>(5));
//        node.getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.removeDuplicatesFromUnSortedLinkedListOnlyConsecutive(node);
//        node = new Node<>(3);
//        node.setNext(new Node<>(4));
//        node.getNext().setNext(new Node<>(5));
//        node.getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.removeDuplicatesFromUnSortedLinkedListAllExtraOccuernce(node);
        //......................................................................
//        Row: 153
//        System.out.println("Find the middle element of the linked list");
//        Node<Integer> node = new Node<>(3);
//        node.setNext(new Node<>(5));
//        node.getNext().setNext(new Node<>(2));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(7));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        System.out.println("Middle element: "+obj.findMiddleNodeOfLinkedList(node).getData());
//        node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        System.out.println("Middle element: "+obj.findMiddleNodeOfLinkedList(node).getData());
        //......................................................................
//        Row: 151
//        System.out.println("Sort linked list using merge sort");
//        Node<Integer> node = new Node<>(3);
//        node.setNext(new Node<>(5));
//        node.getNext().setNext(new Node<>(2));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(7));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        new LinkedListUtil<Integer>(obj.mergeSortDivideAndMerge(node)).print();
//        node = new Node<>(3);
//        node.setNext(new Node<>(3));
//        node.getNext().setNext(new Node<>(7));
//        node.getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        new LinkedListUtil<Integer>(obj.mergeSortDivideAndMerge(node)).print();
        //......................................................................
//        Row: 198
//        System.out.println("Check if a tree is sum tree");
//        TreeNode<Integer> root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(20));
//        root.getLeft().setLeft(new TreeNode<>(10));
//        root.getLeft().setRight(new TreeNode<>(10));
//        root.setRight(new TreeNode<>(30)); //NOT A SUM TREE
//        obj.checkTreeIsSumTree(root);
//        root = new TreeNode<>(3);
//        root.setLeft(new TreeNode<>(2));
//        root.setRight(new TreeNode<>(1)); //SUM TREE
//        obj.checkTreeIsSumTree(root);
        //......................................................................
//        Row: 410
//        System.out.println("Coin change DP problem");
//        int[] coins = {1, 2, 3};
//        int N = coins.length;
//        int K = 4;
//        System.out.println("Possible ways to make change using recursion: "+obj.coinChange_Recursion(coins, N, K));
//        obj.coinChange_DP_Memoization(coins, K);
        //......................................................................
//        Row: 411
//        System.out.println("0-1 knap sack DP problem");
//        int[] weight = {4,5,1};
//        int[] value = {1,2,3};
//        int N = value.length;
//        int W = 4;
//        System.out.println("The maximum profit can be made with given knap sack using recursion: "+obj.knapSack01_Recusrion(W, weight, value, N));
//        obj.knapSack01_DP_Memoization(W, weight, value, N);
//        weight = new int[]{4,5,6};
//        value = new int[]{1,2,3};
//        N = value.length;
//        W = 3;
//        System.out.println("The maximum profit can be made with given knap sack using recursion: "+obj.knapSack01_Recusrion(W, weight, value, N));
//        obj.knapSack01_DP_Memoization(W, weight, value, N);
        //......................................................................
//        Row: 417, 282
//        System.out.println("Subset sum DP problem");
//        int[] arr = new int[]{1, 5, 5, 11};
//        int N = arr.length;
//        int sum = 11;
//        System.out.println("The sub set for the given sum is possible: "+obj.subsetSum_Recursion(arr, sum, N));
//        obj.subsetSum_DP_Memoization(arr, sum, N);
//        System.out.println("Equal sum partition for the given array is possible or not");
//        obj.equalsSumPartition_SubsetSum(arr, N);
//        //arr to be different
//        arr = new int[]{1,5,5,12};
//        obj.equalsSumPartition_SubsetSum(arr, N);
        //......................................................................
//        Row: 423
//        System.out.println("Longest common sub sequence of 2 strings DP problem");
//        String s1 = "ABCDGH";
//        String s2 = "AEDFHR";
//        System.out.println("The longest common sub sequence length for the given 2 strings: "+obj.longestCommonSubsequence_Recursion(s1, s2, s1.length(), s2.length()));
//        obj.longestCommonSubsequence_DP_Memoization(s1, s2, s1.length(), s2.length());
//        s1 = "ABCDGH";
//        s2 = "";
//        System.out.println("The longest common sub sequence length for the given 2 strings: "+obj.longestCommonSubsequence_Recursion(s1, s2, s1.length(), s2.length()));
//        obj.longestCommonSubsequence_DP_Memoization(s1, s2, s1.length(), s2.length());
        //......................................................................
//        Row: 97
//        System.out.println("Check two strings are isomorphic or not");
//        //https://www.geeksforgeeks.org/check-if-two-given-strings-are-isomorphic-to-each-other/
//        String s1 = "aab";
//        String s2 = "xxy";
//        System.out.println("Is isomorphic strings 1: "+obj.checkIsomorphicStrings_1(s1, s2));
//        System.out.println("Is isomorphic strings 2: "+obj.checkIsomorphicStrings_2(s1, s2));
//        s1 = "aab";
//        s2 = "xyz";
//        System.out.println("Is isomorphic strings 1: "+obj.checkIsomorphicStrings_1(s1, s2));
//        System.out.println("Is isomorphic strings 2: "+obj.checkIsomorphicStrings_2(s1, s2));
//        s1 = "13";
//        s2 = "42";
//        System.out.println("Is isomorphic strings 1: "+obj.checkIsomorphicStrings_1(s1, s2));
//        System.out.println("Is isomorphic strings 2: "+obj.checkIsomorphicStrings_2(s1, s2));
        //......................................................................
//        Row: 96
//        System.out.println("Transform one string to another with min gievn no of operations");
//        //https://www.geeksforgeeks.org/transform-one-string-to-another-using-minimum-number-of-given-operation/
//        System.out.println("Transform operations required: " + obj.transformOneStringToAnotherWithMinOprn("EACBD", "EABCD"));
//        System.out.println("Transform operations required: " + obj.transformOneStringToAnotherWithMinOprn("EACCD", "EABCD"));
        //......................................................................
//        Row: 154
//        System.out.println("Check if a linked list is circular linked list");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(node); //CIRCULAR 6 -> 1
//        System.out.println("Check if given linked list is circular linked list: " + obj.checkIfLinkedListIsCircularLinkedList(node));
//        node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(6)); //NOT CIRCULAR 6 -> NULL
//        System.out.println("Check if given linked list is circular linked list: " + obj.checkIfLinkedListIsCircularLinkedList(node));
        //......................................................................
//        Row: 152
//        System.out.println("Quick sort in linked list");
//        //https://www.geeksforgeeks.org/quick-sort/
//        Node<Integer> node = new Node<>(10);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(5));
//        node.getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().setNext(new Node<>(3));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.quickSortInLinkedList(node);
        //......................................................................
//        Row: 304
//        System.out.println("Find next greater element");
//        obj.nextGreaterElementInRightInArray(new int[]{1,3,2,4});
//        obj.nextGreaterElementInRightInArray(new int[]{1,2,3,4,5}); //STACK WILL HOLD N ELEMENT S: O(N)
//        obj.nextGreaterElementInRightInArray(new int[]{5,4,3,2,1}); //STACK WILL NOT HOLD N ELEMENT S: O(1)
        //......................................................................
//        Row: 339
//        System.out.println("First K largest element in array");
//        obj.kLargestElementInArray(new int[]{12, 5, 787, 1, 23}, 2);
//        obj.kLargestElementInArray(new int[]{1, 23, 12, 9, 30, 2, 50}, 3);
        //......................................................................
//        Row: 20, 70
//        System.out.println("Next permutation");
//        //https://leetcode.com/problems/next-permutation/solution/
//        obj.nextPermutation(new int[]{1,2,3});
//        obj.nextPermutation(new int[]{4,3,2,1});
//        obj.nextPermutation(new int[]{1,3,1,4,7,6,2});
//        obj.nextPermutation(new int[]{2,7,4,3,2});
//        obj.nextPermutation(new int[]{1, 2, 3, 6, 5, 4});
        //......................................................................
//        Row: 27
//        System.out.println("Factorial of large number");
//        //https://www.geeksforgeeks.org/factorial-large-number/
//        obj.factorialLargeNumber(1);
//        obj.factorialLargeNumber(5);
//        obj.factorialLargeNumber(10);
//        obj.factorialLargeNumber(897);
        //......................................................................
//        Row: 103
//        System.out.println("Search in rotated sorted array");
//        System.out.println("The target is found at location: "+ obj.searchInRotatedSortedArray(new int[]{4,5,6,7,0,1,2}, 0));
//        System.out.println("The target is found at location: "+ obj.searchInRotatedSortedArray(new int[]{4,5,6,7,0,1,2}, 4));
//        System.out.println("The target is found at location: "+ obj.searchInRotatedSortedArray(new int[]{4,5,6,7,0,1,2}, 3));
//        //https://leetcode.com/problems/search-in-rotated-sorted-array-ii/
//        System.out.println("The target is found at location in arr with duplicate elements: "
//                + obj.searchInRotatedSortedArrayWithDuplicateArrayElement(new int[]{1,0,1,1,1}, 0));
//        System.out.println("The target is found at location in arr with duplicate elements: "
//                + obj.searchInRotatedSortedArrayWithDuplicateArrayElement(new int[]{2,5,6,0,0,1,2}, 0));
//        System.out.println("The target is found at location in arr with duplicate elements: "
//                + obj.searchInRotatedSortedArrayWithDuplicateArrayElement(new int[]{2,5,6,0,0,1,2}, 3));
        //......................................................................
//        Row: 146
//        System.out.println("Move last node of linked list to front");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.moveLastNodeToFrontOfLinkedList(node);
        //......................................................................
//        Row: 147
//        System.out.println("Add 1 to linked list");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        obj.addOneToLinkedList(node);
//        node = new Node<>(9); //COND WHEN METHOD WILL CREATE NEWHEAD TO STORE EXTRA CARRY IN THE SUM RECURSION
//        node.setNext(new Node<>(9));
//        node.getNext().setNext(new Node<>(9));
//        node.getNext().getNext().setNext(new Node<>(9));
//        obj.addOneToLinkedList(node);
        //......................................................................
//        Row: 167
//        System.out.println("Sort linked list of 0s, 1s, 2s using 2 approaches");
//        //https://www.geeksforgeeks.org/sort-a-linked-list-of-0s-1s-or-2s/
//        Node<Integer> node = new Node<>(0);
//        node.setNext(new Node<>(1));
//        node.getNext().setNext(new Node<>(0));
//        node.getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        new LinkedListUtil<>(obj.mergeSortDivideAndMerge(node)).print(); //SIMPLE MERGE SORT APPROACH T: O(N.LogN)
//        node = new Node<>(0);
//        node.setNext(new Node<>(1));
//        node.getNext().setNext(new Node<>(0));
//        node.getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        obj.sortLinkedListOf012_2(node); //SIMPLE MANIPULATION OF NODE T: O(N)
        //......................................................................
//        Row: 202
//        System.out.println("Sum of node on the longest path of tree from root to leaf");
//        TreeNode<Integer> root = new TreeNode<>(4);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(7));
//        root.getLeft().setRight(new TreeNode<>(1));
//        root.getLeft().getRight().setLeft(new TreeNode<>(6)); //LONGEST PATH
//        root.setRight(new TreeNode<>(5));
//        root.getRight().setLeft(new TreeNode<>(2));
//        root.getRight().setRight(new TreeNode<>(3));
//        obj.longestPathNodeSum(root);
        //......................................................................
//        Row: 34
//        System.out.println("Rain water trapping 2 approaches");
//        obj.rainWaterTrappingUsingStack(new int[]{3,0,0,2,0,4});
//        obj.rainWaterTrappingUsingTwoPointers(new int[]{3,0,0,2,0,4});
//        obj.rainWaterTrappingUsingStack(new int[]{6,9,9});
//        obj.rainWaterTrappingUsingTwoPointers(new int[]{6,9,9});
//        obj.rainWaterTrappingUsingStack(new int[]{7,4,0,9});
//        obj.rainWaterTrappingUsingTwoPointers(new int[]{7,4,0,9});
//        obj.rainWaterTrappingUsingTwoPointers(new int[]{3,4,1});
        //......................................................................
//        Row: 28
//        System.out.println("Find maximum product subarray");
//        obj.findMaximumProductSubarray(new int[]{2,3,-2,4});
//        obj.findMaximumProductSubarray(new int[]{-2,0,-1});
        //......................................................................
//        Row: 217
//        System.out.println("Find predecessor and successor of given node in BST");
//        //https://www.geeksforgeeks.org/inorder-predecessor-successor-given-key-bst/
//        //predecessors and successor can be found when we do the inorder traversal of tree
//        //inorder traversal of BST is sorted list of node data
//        //for below BST inorder list [0,2,3,4,5,6,7,8,9]
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.findPredecessorAndSuccessorInBST(root1, 6);
//        obj.findPredecessorAndSuccessorInBST(root1, 2);
//        obj.findPredecessorAndSuccessorInBST(root1, 5);
//        obj.findPredecessorAndSuccessorInBST(root1, 10); //ONLY PREDECESSOR IS POSSIBLE
//        obj.findPredecessorAndSuccessorInBST(root1, -1); //ONLY SUCCESSOR IS POSSIBLE
        //......................................................................
//        Row: 424, 64
//        System.out.println("Longest Repeating Subsequence DP problem");
//        System.out.println("Longest repeating subsequence: "+obj.longestRepeatingSubsequence_Recursion("axxxy", 5)); //xx, xx
//        obj.longestRepeatingSubsequence_DP_Memoization("axxxy"); //xx, xx
        //......................................................................
//        Row: 441
//        System.out.println("Longest common substring DP problem");
//        obj.longestCommonSubstring_DP_Memoization("ABCDGH", "ACDGHR");
        //......................................................................
//        Row: 441
//        System.out.println("Maximum length of pair chain 2 approaches");
//        //https://leetcode.com/problems/maximum-length-of-pair-chain/
//        System.out.println("maximum length of pair chain DP approach: "+
//                obj.maximumLengthOfPairChain_DP_Approach(new int[][]{
//                    {1,2},
//                    {3,4},
//                    {2,3}
//                }));
//        System.out.println("maximum length of pair chain Greedy approach: "+
//                obj.maximumLengthOfPairChain_Greedy_Approach(new int[][]{
//                    {1,2},
//                    {3,4},
//                    {2,3}
//                }));
        //......................................................................
//        Row: 412
//        System.out.println("Binomial coefficient DP problem");
//        //https://www.geeksforgeeks.org/binomial-coefficient-dp-9/
//        System.out.println("Binomial coefficient recursive way: "+obj.findBinomialCoefficient_Recursion(5, 2));
//        obj.findBinomialCoefficient_DP_Memoization(5, 2);
//        System.out.println("Binomial coefficient recursive way: "+obj.findBinomialCoefficient_Recursion(5, 6));
//        obj.findBinomialCoefficient_DP_Memoization(5, 6);
//        System.out.println("Binomial coefficient recursive way: "+obj.findBinomialCoefficient_Recursion(5, 5));
//        obj.findBinomialCoefficient_DP_Memoization(5, 5);
//        System.out.println("Binomial coefficient recursive way: "+obj.findBinomialCoefficient_Recursion(5, 0));
//        obj.findBinomialCoefficient_DP_Memoization(5, 0);
        //......................................................................
//        Row: 229
//        System.out.println("Count BST nodes that lie in the given range");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.countNodesThatLieInGivenRange(root1, 1, 4);
//        obj.countNodesThatLieInGivenRange(root1, 6, 9);
        //......................................................................
//        Row: 235
//        System.out.println("Flatten BST to linked list (skewed tree)");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.flattenBSTToLinkedList(root1);
        //......................................................................
//        Row: 158
//        System.out.println("Reverse a doubly linked list");
//        Node<Integer> node = new Node<>(3);
//        Node<Integer> next = new Node<>(4);
//        node.setNext(next);
//        next.setPrevious(node);
//        Node<Integer> nextToNext = new Node<>(5);
//        next.setNext(nextToNext);
//        nextToNext.setPrevious(next);
//        obj.reverseDoublyLinkedList(node);
        //......................................................................
//        Row: 18, 13
//        System.out.println("Kaden's algorithm approaches");
//        //https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/
//        int a[] = {-2, -3, 4, -1, -2, 1, 5, -3};
//        obj.kadaneAlgorithm(a);
//        obj.kadaneAlgorithm_PointingIndexes(a);
        //......................................................................
//        Row: 10
//        System.out.println("Move all negative elements to one side of array");
//        obj.moveNegativeElementsToOneSideOfArray(new int[]{-12, 11, -13, -5, 6, -7, 5, -3, -6});
//        obj.moveNegativeElementsToOneSideOfArray(new int[]{ -1, 2, -3, 4, 5, 6, -7, 8, 9});
//        obj.moveNegativeElementsToOneSideOfArray(new int[]{ -1, -2, -3, -1, -10, -7});
//        obj.moveNegativeElementsToOneSideOfArray(new int[]{ 1, 2, 3, 1, 10, 7});
//        obj.moveNegativeElementsToOneSideOfArray(new int[]{ 1, -2, -3, -1, -10, -7});
        //......................................................................
//        Row: 11
//        System.out.println("Find union and intersection of two arrays");
//        obj.findUnionAndIntersectionOfTwoArrays(new int[]{1,2,3,4,5}, new int[]{1,2,3});
//        obj.findUnionAndIntersectionOfTwoArrays(new int[]{4,9,5}, new int[]{9,4,9,8,4});
        //......................................................................
//        Row: 12
//        System.out.println("Cyclically rotate element in array by 1");
//        obj.rotateArrayByK(new int[]{1, 2, 3, 4, 5}, 1);
//        obj.rotateArrayByK(new int[]{1, 2, 3, 4, 5}, 4);
        //......................................................................
//        Row: 14
//        System.out.println("Minimize the difference between the heights");
//        //https://www.geeksforgeeks.org/minimize-the-maximum-difference-between-the-heights/
//        obj.minimizeDifferenceBetweenHeights(new int[]{1, 5, 8, 10}, 2);
//        obj.minimizeDifferenceBetweenHeights(new int[]{4, 6}, 10);
        //......................................................................
//        Row: 191
//        System.out.println("Diagonal traversal of tree");
//        //https://www.geeksforgeeks.org/diagonal-traversal-of-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.diagonalTraversalOfTree(root1);
        //......................................................................
//        Row: 180
//        System.out.println("Diameter of tree DP on tree problem");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.diameterOfTree(root1);
        //......................................................................
//        Row: 238, 251
//        System.out.println("N meeting in a room/ Activity selection");
//        int[] startTime = {1, 3, 0, 5, 8, 5};
//        int[] finishTime = {2, 4, 6, 7, 9, 9};
//        obj.nMeetingRooms_Greedy(startTime, finishTime);
        //......................................................................
//        Row: 357, 358
//        System.out.println("BFS/DFS directed graph");
//        List<List<Integer>> adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList());
//        adjList.add(2, Arrays.asList(4));
//        adjList.add(3, Arrays.asList());
//        adjList.add(4, Arrays.asList());
//        obj.graphBFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Recursive_Graph(adjList.size(), adjList);
//        adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList(5));
//        adjList.add(2, Arrays.asList(4));
//        adjList.add(3, Arrays.asList());
//        adjList.add(4, Arrays.asList(3));
//        adjList.add(5, Arrays.asList());
//        obj.graphBFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Recursive_Graph(adjList.size(), adjList);
//        adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1));
//        adjList.add(1, Arrays.asList(2));
//        adjList.add(2, Arrays.asList(3));
//        adjList.add(3, Arrays.asList(4));
//        adjList.add(4, Arrays.asList(5));
//        adjList.add(5, Arrays.asList());
//        obj.graphBFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Recursive_Graph(adjList.size(), adjList);
        //......................................................................
//        Row: 361, 275
//        System.out.println("Search in maze");
//        int[][] maze = new int[][]{
//            {1, 0, 0, 0},
//            {1, 1, 0, 1},
//            {1, 1, 0, 0},
//            {0, 1, 1, 1}
//        };
//        obj.findPathRatInMaze_Graph(maze, maze.length);
//        maze = new int[][]{
//            {1, 0, 0, 0},
//            {1, 1, 0, 1},
//            {1, 1, 0, 0},
//            {0, 1, 1, 0}
//        };
//        obj.findPathRatInMaze_Graph(maze, maze.length);
        //......................................................................
//        Row: 371
//        System.out.println("No. of Island");
//        int[][] grid = {{0, 1, 1, 1, 0, 0, 0}, {0, 0, 1, 1, 0, 1, 0}};
//        obj.numberOfIslands_Graph(grid);
        //......................................................................
//        Row: 348
//        System.out.println("Check if binary tree is heap (max heap)");
//        TreeNode<Integer> root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(7));
//        root.getLeft().setLeft(new TreeNode<>(6));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(8));
//        root.getRight().setLeft(new TreeNode<>(4));
//        obj.checkIfBinaryTreeIsMaxHeap(root);
//        root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(7));
//        root.getLeft().setLeft(new TreeNode<>(6));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(8));
//        root.getRight().setLeft(new TreeNode<>(9)); 
//        obj.checkIfBinaryTreeIsMaxHeap(root);
//        root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(7));
//        root.getLeft().setLeft(new TreeNode<>(6));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(8));
//        root.getRight().setLeft(new TreeNode<>(4)); 
//        root.getRight().setRight(new TreeNode<>(3)); 
//        obj.checkIfBinaryTreeIsMaxHeap(root);
        //......................................................................
//        Row: 91
//        System.out.println("Arrange all anagrams together");
//        obj.arrangeAllWordsAsTheirAnagrams(Arrays.asList("act", "god", "cat", "dog", "tac"));
        //......................................................................
//        Row: 90
//        System.out.println("Minimum character to be added at front of string to make it pallindrome");
//        obj.characterAddedAtFrontToMakeStringPallindrome_1("ABC"); // 2 char = B,C (ex CBABC)
//        obj.characterAddedAtFrontToMakeStringPallindrome_1("ABA"); // 0 char already pallindrome
//        obj.characterAddedAtFrontToMakeStringPallindrome_2("ABC"); //KMP based
//        obj.characterAddedAtFrontToMakeStringPallindrome_2("ABA"); //KMP based
        //......................................................................
//        Row: 360
//        System.out.println("Detect cycle in undirected graph DFS");
//        //https://www.geeksforgeeks.org/detect-cycle-undirected-graph/
//        List<List<Integer>> adjList = new ArrayList<>(); //CYCLE //0 <--> 1 <--> 2 <--> 0
//        adjList.add(0, Arrays.asList(1, 2)); 
//        adjList.add(1, Arrays.asList(0, 2));
//        adjList.add(2, Arrays.asList(0, 1));
//        System.out.println("Is there a cycle in undirected graph: " + obj.detectCycleInUndirectedGraphDFS_Graph(adjList.size(), adjList));
//        adjList = new ArrayList<>(); //NO CYCLE 0 <--> 1 <--> 2
//        adjList.add(0, Arrays.asList(1));
//        adjList.add(1, Arrays.asList(0, 2));
//        adjList.add(2, Arrays.asList(1));
//        System.out.println("Is there a cycle in undirected graph: " + obj.detectCycleInUndirectedGraphDFS_Graph(adjList.size(), adjList));
//        adjList = new ArrayList<>(); //CYCLE //FAIL CASE // 0 <--> 1 //CASE OF DIRECTED GRAPH
//        adjList.add(0, Arrays.asList(1));
//        adjList.add(1, Arrays.asList(0));
//        System.out.println("Is there a cycle in undirected graph: " + obj.detectCycleInUndirectedGraphDFS_Graph(adjList.size(), adjList));
        //......................................................................
//        Row: 368
//        System.out.println("Topological sort graph");    
//        List<List<Integer>> adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList());
//        adjList.add(1, Arrays.asList());
//        adjList.add(2, Arrays.asList(3));
//        adjList.add(3, Arrays.asList(1));
//        adjList.add(4, Arrays.asList(0,1));
//        adjList.add(5, Arrays.asList(0,2));
//        obj.topologicalSort_Graph(adjList.size(), adjList);
        //......................................................................
//        Row: 439
//        System.out.println("Minimum cost to fill the given bag");
//        //https://www.geeksforgeeks.org/minimum-cost-to-fill-given-weight-in-a-bag/
//        obj.minimumCostToFillGivenBag_DP_Memoization(new int[]{20, 10, 4, 50, 100}, 5);
//        obj.minimumCostToFillGivenBag_DP_Memoization(new int[]{-1, -1, 4, 3, -1}, 5);
        //......................................................................
//        Row: 22
//        System.out.println("Best time to buy and sell stock");
//        obj.bestProfitToBuySellStock(new int[]{7,1,5,3,6,4});
//        obj.bestProfitToBuySellStock(new int[]{7,6,4,3,1});
        //......................................................................
//        Row: 23
//        System.out.println("Find all pairs in array whose sum is given to K");
//        //https://www.geeksforgeeks.org/count-pairs-with-given-sum/
//        obj.countAllPairsInArrayThatSumIsK(new int[]{1, 5, 7, 1}, 6);
//        obj.countAllPairsInArrayThatSumIsK(new int[]{1, 1, 1, 1}, 2);
        //......................................................................
//        Row: 60
//        System.out.println("Check if one string is rotation of other string");
//        //https://www.geeksforgeeks.org/a-program-to-check-if-strings-are-rotations-of-each-other/
//        System.out.println("Check if one string is rotation: "+obj.checkIfOneStringRotationOfOtherString("AACD", "ACDA"));
        //......................................................................
//        Row: 312
//        System.out.println("Largest area of histogram");
//        //https://www.geeksforgeeks.org/largest-rectangle-under-histogram/
//        obj.largestAreaInHistogram(new int[]{6, 2, 5, 4, 5, 1, 6});
//        obj.largestAreaInHistogram(new int[]{6,9,8});
        //......................................................................
//        Row: 418
//        System.out.println("Friends pairing DP problem");
//        //https://www.geeksforgeeks.org/friends-pairing-problem/
//        System.out.println("No. of ways friends can be paired recursion: "+obj.friendsPairingProblem_Recursion(4));
//        obj.friendsPairingProblem_DP_Memoization(4);
        //......................................................................
//        Row: 341
//        System.out.println("Merge k sorted arrays (heap)");
//        int[][] arr = new int[][]{
//            {1,2,3},
//            {4,5,6},
//            {7,8,9}
//        };
//        obj.mergeKSortedArrays_1(arr);
//        obj.mergeKSortedArrays_2(arr);
        //......................................................................
//        Row: 343
//        System.out.println("Kth largest sum from contigous subarray");
//        //https://www.geeksforgeeks.org/k-th-largest-sum-contiguous-subarray/
//        obj.kThLargestSumFromContigousSubarray(new int[]{10, -10, 20, -40}, 6);
//        obj.kThLargestSumFromContigousSubarray(new int[]{20, -5, -1}, 3);
        //......................................................................
//        Row: 329
//        System.out.println("Check if all levels in two trees are anagrams of each other");
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(4));
//        root1.getRight().setRight(new TreeNode<>(5));
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.setRight(new TreeNode<>(2));
//        root2.getRight().setLeft(new TreeNode<>(5));
//        root2.getRight().setRight(new TreeNode<>(4));
//        System.out.println("Check if all levels of two trees are anagrams 1: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_1(root1, root2));
//        System.out.println("Check if all levels of two trees are anagrams 2: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_2(root1, root2));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(6));
//        root1.getLeft().setRight(new TreeNode<>(7));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(4));
//        root1.getRight().setRight(new TreeNode<>(5));
//        root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.setRight(new TreeNode<>(2));
//        root2.getRight().setLeft(new TreeNode<>(5));
//        root2.getRight().setRight(new TreeNode<>(4));
//        System.out.println("Check if all levels of two trees are anagrams 1: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_1(root1, root2));
//        System.out.println("Check if all levels of two trees are anagrams 2: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_2(root1, root2));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(2));
//        root1.getRight().setLeft(new TreeNode<>(4));
//        root1.getRight().setRight(new TreeNode<>(5));
//        root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.setRight(new TreeNode<>(2));
//        root2.getRight().setLeft(new TreeNode<>(5));
//        root2.getRight().setRight(new TreeNode<>(4));
//        System.out.println("Check if all levels of two trees are anagrams 1: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_1(root1, root2));
//        System.out.println("Check if all levels of two trees are anagrams 2: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_2(root1, root2));
        //......................................................................
//        Row: 78
//        System.out.println("Count the presence of given string in char array");
//        char[][] charArr = new char[][]{
//            {'D','D','D','G','D','D'},
//            {'B','B','D','E','B','S'},
//            {'B','S','K','E','B','K'},
//            {'D','D','D','D','D','E'},
//            {'D','D','D','D','D','E'},
//            {'D','D','D','D','D','G'}
//           };
//        String str= "GEEKS";
//        obj.countOccurenceOfGivenStringInCharArray(charArr, str);
//        charArr = new char[][]{
//            {'B','B','M','B','B','B'},
//            {'C','B','A','B','B','B'},
//            {'I','B','G','B','B','B'},
//            {'G','B','I','B','B','B'},
//            {'A','B','C','B','B','B'},
//            {'M','C','I','G','A','M'}
//           };
//        str= "MAGIC";
//        obj.countOccurenceOfGivenStringInCharArray(charArr, str);
        //......................................................................
//        Row: 149
//        System.out.println("Intersection of two sorted linked list");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        Node<Integer> node2 = new Node<>(2);
//        node2.setNext(new Node<>(4));
//        node2.getNext().setNext(new Node<>(4));
//        node2.getNext().getNext().setNext(new Node<>(6));
//        obj.intersectionOfTwoSortedLinkedList(node2, node2);
        //......................................................................
//        Row: 26
//        System.out.println("Check if any sub array with sum 0 is present or not");
//        System.out.println("Is there with subarray sum 0 "+obj.checkIfSubarrayWithSum0(new int[]{4, 2, -3, 1, 6}));
//        System.out.println("Is there with subarray sum 0 "+obj.checkIfSubarrayWithSum0(new int[]{4, 2, 0, -1}));
        //......................................................................
//        Row: 31
//        System.out.println("Maximum profit by buying seling stocks atmost twice");
//        obj.bestProfitToBuySellStockAtMostTwice(new int[]{ 2, 30, 15, 10, 8, 25, 80 });
//        obj.bestProfitToBuySellStockAtMostTwice(new int[]{ 2, 30, 80, 10, 8, 25, 60 });
        //......................................................................
//        Row: 107, 16
//        System.out.println("Find repeating and missing in unsorted array");
//        //https://www.geeksforgeeks.org/find-a-repeating-and-a-missing-number/
//        obj.findRepeatingAndMissingInUnsortedArray_1(new int[]{7, 3, 4, 5, 5, 6, 2 });
//        obj.findRepeatingAndMissingInUnsortedArray_1(new int[]{3,1,3});
//        obj.findRepeatingAndMissingInUnsortedArray_2(new int[]{7, 3, 4, 5, 5, 6, 2 });
//        obj.findRepeatingAndMissingInUnsortedArray_2(new int[]{3,1,3});
        //......................................................................
//        Row: 110
//        System.out.println("Check if any pair possible in an array having given difference");
//        System.out.println("Check if any pair is possible in the array having given diff: "+
//                obj.checkIfPairPossibleInArrayHavingGivenDiff(new int[]{5, 20, 3, 2, 5, 80}, 78));
//        System.out.println("Check if any pair is possible in the array having given diff: "+
//                obj.checkIfPairPossibleInArrayHavingGivenDiff(new int[]{90, 70, 20, 80, 50}, 45));
        //......................................................................
//        Row: 150
//        System.out.println("Intersection point in two given linked list (by ref linkage) 2 approach");
//        Node<Integer> common = new Node<>(15);
//        common.setNext(new Node<>(30));
//        Node<Integer> node1 = new Node<>(3);
//        node1.setNext(new Node<>(9));
//        node1.getNext().setNext(new Node<>(6));
//        node1.getNext().getNext().setNext(common);
//        Node<Integer> node2 = new Node<>(10);
//        node2.setNext(common);
//        obj.intersectionPointOfTwoLinkedListByRef(node1, node2);
//        obj.intersectionPointOfTwoLinkedListByRef_HashBased(node1, node2);
//        common = new Node<>(4);
//        common.setNext(new Node<>(5));
//        common.getNext().setNext(new Node<>(6));
//        node1 = new Node<>(1);
//        node1.setNext(new Node<>(2));
//        node1.getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().setNext(common);
//        node2 = new Node<>(10);
//        node2.setNext(new Node<>(20));
//        node2.getNext().setNext(common);
//        obj.intersectionPointOfTwoLinkedListByRef(node1, node2);
//        obj.intersectionPointOfTwoLinkedListByRef_HashBased(node1, node2);
        //......................................................................
//        Row: 359
//        System.out.println("Detect cycle in directed graph using DFS");
//        //https://www.geeksforgeeks.org/detect-cycle-in-a-graph/
//        List<List<Integer>> adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1)); //CYCLE 0 --> 1 --> 2 --> 0
//        adjList.add(1, Arrays.asList(2));
//        adjList.add(2, Arrays.asList(0));
//        System.out.println("Is there a cycle in directed graph: " + obj.detectCycleInDirectedGraphDFS_Graph(adjList.size(), adjList));
//        adjList = new ArrayList<>(); //NO CYCLE // 0 --> 1 --> 2
//        adjList.add(0, Arrays.asList(1));
//        adjList.add(1, Arrays.asList(2));
//        adjList.add(2, Arrays.asList());
//        System.out.println("Is there a cycle in directed graph: " + obj.detectCycleInDirectedGraphDFS_Graph(adjList.size(), adjList));
//        adjList = new ArrayList<>(); //CYCLE // 0 --> 1 --> 0
//        adjList.add(0, Arrays.asList(1));
//        adjList.add(1, Arrays.asList(0));
//        System.out.println("Is there a cycle in directed graph: " + obj.detectCycleInDirectedGraphDFS_Graph(adjList.size(), adjList));
        //......................................................................
//        Row: 363
//        System.out.println("Flood fill");
//        int[][] image = new int[][]{
//            {1,1,1},
//            {1,1,0},
//            {1,0,1}
//        };
//        obj.floodFill(image, 1, 1, 2);
//        image = new int[][]{
//            {0,0,0},
//            {0,0,0}
//        };
//        obj.floodFill(image, 0, 0, 2);
//        image = new int[][]{
//            {0,0,0,0,0},
//            {0,1,1,1,0},
//            {0,1,1,1,0},
//            {0,1,1,1,0},
//            {0,0,0,0,0},
//        };
//        obj.floodFill(image, 2, 2, 3);
        //......................................................................
//        Row: 49
//        System.out.println("Maximum size of rectangle in binary matrix");
//        int[][] mat = new int[][]{
//            {0, 1, 1, 0},
//            {1, 1, 1, 1},
//            {1, 1, 1, 1},
//            {1, 1, 0, 0},};
//        obj.maxAreaOfRectangleInBinaryMatrix(mat);
//        mat = new int[][]{
//            {0, 0, 0, 0},
//            {0, 1, 1, 0},
//            {0, 1, 1, 0},
//            {0, 0, 0, 0},};
//        obj.maxAreaOfRectangleInBinaryMatrix(mat);
        //......................................................................
//        Row: 19
//        System.out.println("Merge intervals");
//        int[][] intervals = new int[][]{
//            {1, 3}, {2, 6}, {8, 10}, {15, 18}
//        };
//        obj.mergeIntervals_1(intervals);
//        obj.mergeIntervals_2(intervals);
//        intervals = new int[][]{
//            {1, 4}, {4, 5}
//        };
//        obj.mergeIntervals_1(intervals);
//        obj.mergeIntervals_2(intervals);
        //......................................................................
//        Row: 47
//        System.out.println("Row with maximum 1s in the matrix");
//        //https://www.geeksforgeeks.org/find-the-row-with-maximum-number-1s/
//        int[][] mat = new int[][]{
//            {0, 1, 1, 1},
//            {0, 0, 1, 1},
//            {1, 1, 1, 1},
//            {0, 0, 0, 0}
//        };
//        obj.maximumOnesInRowOfABinarySortedMatrix_1(mat);
//        obj.maximumOnesInRowOfABinarySortedMatrix_2(mat); //OPTIMISED
//        mat = new int[][]{
//            {0, 0, 0, 0}
//        };
//        obj.maximumOnesInRowOfABinarySortedMatrix_1(mat);
//        obj.maximumOnesInRowOfABinarySortedMatrix_2(mat); //OPTIMISED
        //......................................................................
//        Row: 45
//        System.out.println("Find a value in row wise sorted matrix");
//        int[][] mat = new int[][]{
//            {1, 3, 5, 7}, 
//            {10, 11, 16, 20}, 
//            {23, 30, 34, 60}
//        };
//        obj.findAValueInRowWiseSortedMatrix(mat, 13);
//        mat = new int[][]{
//            {1, 3, 5, 7}, 
//            {10, 11, 16, 20}, 
//            {23, 30, 34, 60}
//        };
//        obj.findAValueInRowWiseSortedMatrix(mat, 11);
        //......................................................................
//        Row: 65
//        System.out.println("Print all subsequences of the given string");
//        //https://www.geeksforgeeks.org/print-subsequences-string/
//        obj.printAllSubSequencesOfAString("abc");
//        obj.printAllSubSequencesOfAString("aaaa");
        //......................................................................
//        Row: 44
//        System.out.println("Spiral matrix traversal");
//        int[][] mat = new int[][]{
//            {1, 2, 3, 4},
//            {5, 6, 7, 8},
//            {9, 10, 11, 12},
//            {13, 14, 15, 16}
//        };
//        obj.spiralMatrixTraversal(mat);
//        mat = new int[][]{
//            {1, 2, 3, 4},
//           {5, 6, 7, 8},
//           {9, 10, 11, 12}
//        };
//        obj.spiralMatrixTraversal(mat);
        //......................................................................
//        Row: 156
//        System.out.println("Check singly linked list is pallindrome or not");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(1));
//        System.out.println("Is linked list pallindrome: "+obj.checkIfLinkedListPallindrome_1(node));
//        System.out.println("Is linked list pallindrome OPTIMISED: "+obj.checkIfLinkedListPallindrome_2(node));
//        node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        System.out.println("Is linked list pallindrome: "+obj.checkIfLinkedListPallindrome_1(node));
//        System.out.println("Is linked list pallindrome OPTIMISED: "+obj.checkIfLinkedListPallindrome_2(node));
        //......................................................................
//        Row: 307
//        System.out.println("Postfix expression evaluation");
//        obj.postfixExpressionEvaluation_SingleDigit("23+");
//        obj.postfixExpressionEvaluation_SingleDigit("231*+9-");
//        obj.postfixExpressionEvaluation_MultipleDigit("10 20 +");
//        obj.postfixExpressionEvaluation_MultipleDigit("100 200 * 10 /");
//        obj.postfixExpressionEvaluation_MultipleDigit("100 200 + 10 / 1000 +");
        //......................................................................
//        Row: 71, 301
//        System.out.println("Balanced parenthesis evaluation");
//        System.out.println(obj.balancedParenthesisEvaluation("()"));
//        System.out.println(obj.balancedParenthesisEvaluation("({[]})"));
//        System.out.println(obj.balancedParenthesisEvaluation(")}]"));
//        System.out.println(obj.balancedParenthesisEvaluation("({)}"));
        //......................................................................
//        Row: 104
//        System.out.println("Square root of a number");
//        System.out.println("Square root of a number: "+obj.squareRootOfANumber(4));
//        System.out.println("Square root of a number: "+obj.squareRootOfANumber(1));
//        System.out.println("Square root of a number: "+obj.squareRootOfANumber(3));
//        System.out.println("Square root of a number: "+obj.squareRootOfANumber(1.5));
        //......................................................................
//        Row: 211
//        System.out.println("Tree isomorphic");
//        //https://www.geeksforgeeks.org/tree-isomorphism-problem/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(3));
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.getLeft().setLeft(new TreeNode<>(4));
//        root2.setRight(new TreeNode<>(2));
//        System.out.println("Are two tres isomorphic: "+obj.areTwoTreeIsoMorphic(root1, root2));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(3));
//        root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.setRight(new TreeNode<>(2));
//        root2.getRight().setRight(new TreeNode<>(4));
//        System.out.println("Are two tres isomorphic: "+obj.areTwoTreeIsoMorphic(root1, root2));
        //......................................................................
//        Row: 210
//        System.out.println("Duplicate subtrees in a tree");
//        //https://www.geeksforgeeks.org/find-duplicate-subtrees/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(2));
//        root1.getRight().getLeft().setLeft(new TreeNode<>(4));
//        root1.getRight().setRight(new TreeNode<>(4));
//        obj.findDuplicateSubtreeInAGivenTree(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Print all the nodes that are at K distance from the target node");
//        TreeNode<Integer> root1 = new TreeNode<>(3);
//        root1.setLeft(new TreeNode<>(5));
//        root1.getLeft().setLeft(new TreeNode<>(6));
//        root1.getLeft().setRight(new TreeNode<>(2));
//        root1.getLeft().getRight().setLeft(new TreeNode<>(7));
//        root1.getLeft().getRight().setRight(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(1));
//        root1.getRight().setLeft(new TreeNode<>(0));
//        root1.getRight().setRight(new TreeNode<>(8));
//        obj.printAllTheNodesAtKDistanceFromTargetNode(root1, 5, 2);
//        obj.printAllTheNodesAtKDistanceFromTargetNode(root1, 3, 3);
//        obj.printAllTheNodesAtKDistanceFromTargetNode(root1, 6, 3);
        //......................................................................
//        Row: 39
//        System.out.println("Minimum no of operations required to make an array pallindrome");
//        obj.minOperationsToMakeArrayPallindrome(new int[]{10, 15, 10});
//        obj.minOperationsToMakeArrayPallindrome(new int[]{1, 4, 5, 9, 1});
//        obj.minOperationsToMakeArrayPallindrome(new int[]{1, 2, 3, 4});
        //......................................................................
//        Row: 112
//        System.out.println("maximum sum such that no 2 elements are adjacent / Sticler thief DP problem");
//        int[] houses = new int[]{5,5,10,100,10,5};
//        System.out.println("The maximum amount stickler thief can pick from alternate houses: "+obj.sticklerThief_Recursion(houses, houses.length));
//        obj.sticklerThief_DP_Memoization(houses);
//        houses = new int[]{1,2,3};
//        System.out.println("The maximum amount stickler thief can pick from alternate houses: "+obj.sticklerThief_Recursion(houses, houses.length));
//        obj.sticklerThief_DP_Memoization(houses);
//        houses = new int[]{5};
//        System.out.println("The maximum amount stickler thief can pick from alternate houses: "+obj.sticklerThief_Recursion(houses, houses.length));
//        obj.sticklerThief_DP_Memoization(houses);
        //......................................................................
//        Row: 203
//        System.out.println("Check if given undirected graph is a binary tree or not");
//        //https://www.geeksforgeeks.org/check-given-graph-tree/#:~:text=Since%20the%20graph%20is%20undirected,graph%20is%20connected%2C%20otherwise%20not.
//        List<List<Integer>> adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList(0));
//        adjList.add(2, Arrays.asList(0));
//        adjList.add(3, Arrays.asList(0, 4));
//        adjList.add(4, Arrays.asList(3));
//        System.out.println("Is graph is binary tree: " + obj.checkIfGivenUndirectedGraphIsBinaryTree(adjList.size(), adjList));
//        adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList(0, 2)); // CYCLE 0 <--> 1 <--> 2
//        adjList.add(2, Arrays.asList(0, 1));
//        adjList.add(3, Arrays.asList(0, 4));
//        adjList.add(4, Arrays.asList(3));
//        System.out.println("Is graph is binary tree: " + obj.checkIfGivenUndirectedGraphIsBinaryTree(adjList.size(), adjList));
//        adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList(0));
//        adjList.add(2, Arrays.asList(0));
//        adjList.add(3, Arrays.asList(0));
//        adjList.add(4, Arrays.asList()); // vertex 4 is not connected
//        System.out.println("Is graph is binary tree: " + obj.checkIfGivenUndirectedGraphIsBinaryTree(adjList.size(), adjList));
        //......................................................................
//        Row: 174
//        System.out.println("First non repeating character from the stream of character");
//        obj.firstNonRepeatingCharacterFromStream("geeksforgeeksandgeeksquizfor");
//        obj.firstNonRepeatingCharacterFromStream("aaaaa");
//        obj.firstNonRepeatingCharacterFromStream("abcd");
//        obj.firstNonRepeatingCharacterFromStream("aabbccdd");
        //......................................................................
//        Row: 425
//        System.out.println("Longest increasing subsequence");
//        //https://www.geeksforgeeks.org/longest-increasing-subsequence-dp-3/
//        int[] arr = new int[]{10, 22, 9, 33, 21, 50, 41, 60}; //LONGEST INC SEQ 10, 22, 33, 50, 60: length = 5
//        obj.longestIncreasingSubsequence_BruteForce(arr, arr.length);
//        System.out.println("Longest increasing subsequnec in the givve array: " + obj.longestIncreasingSubsequence_Recursion(arr, arr.length));
//        obj.longestIncreasingSubsequence_DP_Memoization(arr, arr.length);
//        arr = new int[]{3, 10, 2, 1, 20};
//        obj.longestIncreasingSubsequence_BruteForce(arr, arr.length);
//        System.out.println("Longest increasing subsequnec in the givve array: " + obj.longestIncreasingSubsequence_Recursion(arr, arr.length));
//        obj.longestIncreasingSubsequence_DP_Memoization(arr, arr.length);
//        arr = new int[]{3, 2};
//        obj.longestIncreasingSubsequence_BruteForce(arr, arr.length);
//        System.out.println("Longest increasing subsequnec in the givve array: " + obj.longestIncreasingSubsequence_Recursion(arr, arr.length));
//        obj.longestIncreasingSubsequence_DP_Memoization(arr, arr.length);
//        arr = new int[]{0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}; //LONGEST INC SEQ 0, 2, 6, 9, 13, 15: length 6
//        obj.longestIncreasingSubsequence_BruteForce(arr, arr.length); //FAIL CASE
//        System.out.println("Longest increasing subsequnec in the givve array: " + obj.longestIncreasingSubsequence_Recursion(arr, arr.length));
//        obj.longestIncreasingSubsequence_DP_Memoization(arr, arr.length);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Pair of movie watching during a flight");
//        //problem: https://www.geeksforgeeks.org/amazon-interview-experience-sde-2-10/
//        //solution: https://leetcode.com/discuss/interview-question/313719/Amazon-Online-Assessment-2019-Tho-sum-closest/291502
//        obj.pairsOfMoviesCanBeWatchedDuringFlightDurationK_Greedy(new int[]{90, 85, 75, 60, 120, 150, 125}, 250);
//        obj.pairsOfMoviesCanBeWatchedDuringFlightDurationK_Greedy(new int[]{27, 1,10, 39, 12, 52, 32, 67, 76}, 77);
        //......................................................................
//        Row: 261
//        System.out.println("Choclate distributions");
//        //https://www.geeksforgeeks.org/chocolate-distribution-problem/
//        System.out.println("Min diff in distribution of choclates among the students: "
//                +obj.choclateDistribution_Greedy(new int[]{12, 4, 7, 9, 2, 23,
//                    25, 41, 30, 40, 28,
//                    42, 30, 44, 48, 43,
//                   50}, 7));
//        System.out.println("Min diff in distribution of choclates among the students: "
//                +obj.choclateDistribution_Greedy(new int[]{7, 3, 2, 4, 9, 12, 56}, 3));
        //......................................................................
//        Row: 123
//        System.out.println("Kth element in 2 sorted array 2 approaches");
//        //https://www.geeksforgeeks.org/k-th-element-two-sorted-arrays/
//        obj.kThElementInTwoSortedArrays_1(new int[]{2, 3, 6, 7, 9 }, new int[]{1, 4, 8, 10}, 5);
//        obj.kThElementInTwoSortedArrays_2(new int[]{2, 3, 6, 7, 9 }, new int[]{1, 4, 8, 10}, 5); //OPTIMISED
        //......................................................................
//        Row: 72
//        System.out.println("Word break");
//        Set<String> set = new HashSet<>();
//        set.addAll(Arrays.asList("mobile","samsung","sam","sung","man","mango","icecream","and",  
//                            "go","i","like","ice","cream"));
//        System.out.println("Word break possible recursive: "+obj.wordBreak_Recursive("ilikesamsung", set)); 
//        System.out.println("Word break possible recursive: "+obj.wordBreak_Recursive("ilike", set));
//        System.out.println("Word break possible recursive: "+obj.wordBreak_Recursive("ilikedhokhla", set));
//        System.out.println("Word break possible recursive: "+obj.wordBreak_Recursive("andicecreamhill", set));
//        System.out.println("Word break possible dp: "+obj.wordBreak_DP_Problem("ilikesamsung", set)); 
//        System.out.println("Word break possible dp: "+obj.wordBreak_DP_Problem("ilike", set));
//        System.out.println("Word break possible dp: "+obj.wordBreak_DP_Problem("ilikedhokhla", set));
//        System.out.println("Word break possible dp: "+obj.wordBreak_DP_Problem("andicecreamhill", set));
        //......................................................................
//        Row: 245
//        System.out.println("Minimum platform needed");
//        //https://www.geeksforgeeks.org/minimum-number-platforms-required-railwaybus-station/
//        obj.minimumPlatformNeeded_BruteForce(new int[]{900, 940, 950, 1100, 1500, 1800}, 
//                new int[]{910, 1200, 1120, 1130, 1900, 2000});
//        obj.minimumPlatformNeeded_Greedy(new int[]{900, 940, 950, 1100, 1500, 1800}, 
//                new int[]{910, 1200, 1120, 1130, 1900, 2000});
        //......................................................................
//        Row: 242
//        System.out.println("Fractional knapsack");
//        //https://www.geeksforgeeks.org/fractional-knapsack-problem/
//        obj.fractionalKnapsack(new int[]{10,20,30}, new int[]{60, 100, 120}, 50);
//        obj.fractionalKnapsack(new int[]{10,20}, new int[]{60, 100}, 50);
//        obj.fractionalKnapsack(new int[]{60,70}, new int[]{60, 100}, 50);
        //......................................................................
//        Row: 326
//        System.out.println("Rotten oranges");
//        //Hash BASED: this approach rot all those oranges that are adjacent to a rotten orange in 1 unit of time
//        System.out.println("Rottening all the fresh oranges are possile in time: "
//                + obj.rottenOranges_HashBased(new int[][]{
//                    {2, 1, 1}, {1, 1, 0}, {0, 1, 1}
//                }));
//        System.out.println("Rottening all the fresh oranges are possile in time: "
//                + obj.rottenOranges_HashBased(new int[][]{
//                    {2,1,1},{0,1,1},{1,0,1}
//                }));
//        System.out.println("Rottening all the fresh oranges are possile in time: "
//                + obj.rottenOranges_HashBased(new int[][]{
//                    {0,2}
//                }));
//        //DFS BASED: this approach rot all the oranges that are connected to a rotten orange
//        //this follows flood fill way
//        obj.rottenOranges_DFS(new int[][]{
//                    {2, 1, 1}, {1, 1, 0}, {0, 1, 1}
//                });
//        obj.rottenOranges_DFS(new int[][]{
//                    {2,1,1},{0,1,1},{1,0,1}
//                });
//        obj.rottenOranges_DFS(new int[][]{
//                    {0,2}
//                });
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT : MY AMAZON ONLINE ASSESSMENT
//        System.out.println("Robot rodeo/ Robot bounded in the circle");
//        //https://leetcode.com/problems/robot-bounded-in-circle/
//        System.out.println("Is robot moving in circle: " + obj.robotRodeo("GGLLGG"));
//        System.out.println("Is robot moving in circle: " + obj.robotRodeo("GG"));
//        System.out.println("Is robot moving in circle: " + obj.robotRodeo("GL"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT : MY AMAZON ONLINE ASSESSMENT
//        System.out.println("Swaps required to sort the array");
//        //https://www.geeksforgeeks.org/minimum-number-of-swaps-required-to-sort-an-array-set-2/
//        System.out.println("Min swaps: "+obj.swapsRequiredToSortArray(new int[]{1, 5, 4, 3, 2 }));
//        System.out.println("Min swaps: "+obj.swapsRequiredToSortArray(new int[]{7,1,2 }));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("All path from source to target in directed acyclic graph");
//        obj.allPathFromSourceToTargetInDirectedAcyclicGraph(new int[][]{
//            {1, 2}, {3}, {3}, {}
//        });
//        obj.allPathFromSourceToTargetInDirectedAcyclicGraph(new int[][]{
//            {4, 3, 1}, {3, 2, 4}, {3}, {4}, {}
//        });
//        obj.allPathFromSourceToTargetInDirectedAcyclicGraph(new int[][]{
//            {2}, {}, {1}
//        });
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest substring without repeating characters");
//        //https://leetcode.com/problems/longest-substring-without-repeating-characters/
//        obj.longestSubstringWithoutRepeatingChar("abcabcbb");
//        obj.longestSubstringWithoutRepeatingChar("bbbbb");
//        obj.longestSubstringWithoutRepeatingChar("pwwkew");
//        obj.longestSubstringWithoutRepeatingChar("");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum window substring containing all the character of string t in string s");
//        //https://leetcode.com/problems/minimum-window-substring/
//        obj.minimumWindowSubstring("", "");
//        obj.minimumWindowSubstring("ADOBECODEBANC", "ABC");
//        obj.minimumWindowSubstring("a", "a");
//        obj.minimumWindowSubstring("aa", "aa");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Delete tree nodes and return forest");
//        //https://leetcode.com/problems/delete-nodes-and-return-forest/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(3));
//        root1.setRight(new TreeNode<>(2));
//        root1.getRight().setLeft(new TreeNode<>(5));
//        root1.getRight().getLeft().setLeft(new TreeNode<>(6));
//        root1.getRight().getLeft().getLeft().setLeft(new TreeNode<>(7));
//        root1.getRight().getLeft().setRight(new TreeNode<>(8));
//        root1.getRight().setRight(new TreeNode<>(4));
//        obj.deleteTreeNodesAndReturnForest(root1, new int[]{8, 1, 6});
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(3));
//        root1.setRight(new TreeNode<>(2));
//        obj.deleteTreeNodesAndReturnForest(root1, new int[]{3, 2});
        //......................................................................
//        Row: 321
//        System.out.println("LRU cache design");
//        List<String> operations = Arrays.asList("LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get");
//        List<List<Integer>> inputs = Arrays.asList(
//                Arrays.asList(2),
//                Arrays.asList(1, 1),
//                Arrays.asList(2, 2),
//                Arrays.asList(1),
//                Arrays.asList(3, 3),
//                Arrays.asList(2),
//                Arrays.asList(4, 4),
//                Arrays.asList(1),
//                Arrays.asList(3),
//                Arrays.asList(4)
//        );
//        
//        obj.LRUCacheDesignImpl(operations, inputs);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Product of array excluding element itself");
//        obj.productOfArrayExcludingElementItself_BruteForce(new int[]{1, 2, 3, 4});
//        obj.productOfArrayExcludingElementItself_Optimised1(new int[]{1, 2, 3, 4});
//        obj.productOfArrayExcludingElementItself_Optimised2(new int[]{1, 2, 3, 4}); //TIME & SPACE OPTIMISED
        //......................................................................
//        Row: 196
//        System.out.println("Construct binary tree from inorder and preorder");
//        //https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
//        obj.constructBinaryTreeFromInorderPreorderArray(new int[]{9, 3, 15, 20, 7}, new int[]{3, 9, 20, 15, 7});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count all the occurences of anagrams of given pattern in the text SLIDING WINDOW");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-324-sde2/
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-324-sde2/
//        //https://leetcode.com/problems/permutation-in-string/
//        obj.countAllOccurencesOfPatternInGivenString("forxxorfxdofr", "for");
//        obj.countAllOccurencesOfPatternInGivenString("aabaabaa", "aaba");
//        obj.countAllOccurencesOfPatternInGivenString("aabaabaa", "xyz");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Construct BST from given preorder/postorder array");
//        //https://practice.geeksforgeeks.org/problems/preorder-to-postorder4423/1/
//        //PREORDER ARRAY
//        obj.constructBinarySearchTreeFromPreorderArray(new int[]{40,30,35,80,100});
//        obj.constructBinarySearchTreeFromPreorderArray(new int[]{40,30,32,35,80,90,100,120});
//        //POSTORDER ARRAY
//        obj.constructBinarySearchTreeFromPostorderArray(new int[]{35, 30, 100, 80, 40});
//        obj.constructBinarySearchTreeFromPostorderArray(new int[]{35, 32, 30, 120, 100, 90, 80, 40});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Convert all the leaves of tree to DLL and remove leaves from tree");
//        //https://practice.geeksforgeeks.org/problems/leaves-to-dll/1/
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(3));
//        obj.leavesOfTreeToDoublyLinkedListAndRemoveLeaves(root);
//        root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(3));
//        root.getRight().setLeft(new TreeNode<>(6));
//        root.getRight().setRight(new TreeNode<>(7));
//        obj.leavesOfTreeToDoublyLinkedListAndRemoveLeaves(root);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximum of all subarray in size of K");
//        //https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-size-k/
//        //https://practice.geeksforgeeks.org/problems/maximum-of-all-subarrays-of-size-k3101/1/
//        obj.maximumOfAllSubArrayOfSizeK(new int[]{1, 2, 3, 1, 4, 5, 2, 3, 6}, 3);
//        obj.maximumOfAllSubArrayOfSizeK(new int[]{8, 5, 10, 7, 9, 4, 15, 12, 90, 13}, 4);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check two N-ary trees are mirror image of each other");
//        //https://youtu.be/UGzXSDZv-SY
//        List<List<Integer>> tree1 = Arrays.asList(
//                Arrays.asList(1, 2, 3, 4), //0
//                Arrays.asList(5, 6), //1
//                Arrays.asList(7), //2
//                Arrays.asList(), //3
//                Arrays.asList(), //4
//                Arrays.asList(), //5
//                Arrays.asList(), //6
//                Arrays.asList() //7
//        );
//        List<List<Integer>> tree2 = Arrays.asList(
//                Arrays.asList(4, 3, 2, 1), //0
//                Arrays.asList(6, 5), //1
//                Arrays.asList(7), //2
//                Arrays.asList(), //3
//                Arrays.asList(), //4
//                Arrays.asList(), //5
//                Arrays.asList(), //6
//                Arrays.asList() //7
//        );
//        System.out.println("Check if two N-ary trees are mirror of each other: "
//                + obj.checkIfTwoNAryTreeAreMirror(tree1, tree2));
//        tree1 = Arrays.asList(
//                Arrays.asList(1, 2, 3, 4), //0
//                Arrays.asList(5, 6), //1
//                Arrays.asList(7), //2
//                Arrays.asList(), //3
//                Arrays.asList(), //4
//                Arrays.asList(), //5
//                Arrays.asList(), //6
//                Arrays.asList() //7
//        );
//        //BOTH TREE ARE SAME NOW THIS SHOULD GIVE FALSE
//        tree2 = Arrays.asList(
//                Arrays.asList(1, 2, 3, 4), //0
//                Arrays.asList(5, 6), //1
//                Arrays.asList(7), //2
//                Arrays.asList(), //3
//                Arrays.asList(), //4
//                Arrays.asList(), //5
//                Arrays.asList(), //6
//                Arrays.asList() //7
//        );
//        System.out.println("Check if two N-ary trees are mirror of each other: "
//                + obj.checkIfTwoNAryTreeAreMirror(tree1, tree2));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find minimum in rotated sorted array");
//        //https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
//        obj.findMinimumInRotatedSortedArray(new int[]{3,4,5,1,2});
//        obj.findMinimumInRotatedSortedArray(new int[]{0,1,2,3,4,5});
//        obj.findMinimumInRotatedSortedArray(new int[]{4,6,8,10,1,3,});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Deepest leaves sum in the tree");
//        //https://leetcode.com/problems/deepest-leaves-sum/
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.getLeft().getLeft().setLeft(new TreeNode<>(7));
//        root.setRight(new TreeNode<>(3));
//        root.getRight().setRight(new TreeNode<>(6));
//        root.getRight().getRight().setRight(new TreeNode<>(8));
//        obj.deepestLeavesSumOfTree_Iterative(root);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Average waiting time");
//        //https://leetcode.com/problems/average-waiting-time/
//        obj.averageWaitingTime(new int[][]{{1,2},{2,5},{4,3}});
//        obj.averageWaitingTime(new int[][]{{5,2},{5,4},{10,3},{20,1}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Partition labels");
//        //https://leetcode.com/problems/partition-labels/
//        obj.partitionLabels("ababcbacadefegdehijhklij");
//        obj.partitionLabels("aabbaaccddcc");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest repeating character replacement (SLIDING WINDOW)");
//        //https://leetcode.com/problems/longest-repeating-character-replacement/
//        obj.longestRepeatingCharacterByKReplacement("ABAB", 2);
//        obj.longestRepeatingCharacterByKReplacement("AABABBA", 1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("3Sum");
//        //https://leetcode.com/problems/3sum/
//        obj.threeSum(new int[]{-1,0,1,2,-1,-4});
//        obj.threeSum(new int[]{0}); //no trplet sum is posibble
//        obj.threeSum(new int[]{});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum time differenec");
//        //https://leetcode.com/problems/minimum-time-difference/
//        System.out.println("Min tme difference: "+obj.minimumTimeDifference(Arrays.asList("23:59","00:00")));
//        System.out.println("Min tme difference: "+obj.minimumTimeDifference(Arrays.asList("00:00","23:59","00:00")));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Contigous array (subarray with equal no of 0 and 1)");
//        //https://leetcode.com/problems/contiguous-array/
//        obj.contigousArrayWithEqualZeroAndOne(new int[]{0,1});
//        obj.contigousArrayWithEqualZeroAndOne(new int[]{0,1,0,1,0});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Generate balanced parenthesis");
//        //https://www.geeksforgeeks.org/print-all-combinations-of-balanced-parentheses/
//        //https://leetcode.com/problems/generate-parentheses/
//        obj.generateBalancedParenthesis(1);
//        obj.generateBalancedParenthesis(2);
//        obj.generateBalancedParenthesis(3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find whether path exists");
//        //https://practice.geeksforgeeks.org/problems/find-whether-path-exist5238/1
//        System.out.println("Path exists: " + obj.checkIfPathExistsFromSourceToDestination(new int[][]{
//            {1, 3},
//            {3, 2}
//        }));
//        System.out.println("Path exists: " + obj.checkIfPathExistsFromSourceToDestination(new int[][]{
//            {3, 0, 3, 0, 0}, {3, 0, 0, 0, 3}, {3, 3, 3, 3, 3}, {0, 2, 3, 0, 0}, {3, 0, 0, 1, 3}
//        }));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Reverse Linked list in K groups alternatively");
//        //https://www.interviewbit.com/problems/reverse-alternate-k-nodes/
//        Node<Integer> head = new Node<>(3);
//        head.setNext(new Node<>(4));
//        head.getNext().setNext(new Node<>(7));
//        head.getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().setNext(new Node<>(6));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        head.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(15));
//        head.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(61));
//        head.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(16));
//        new LinkedListUtil<Integer>(obj.reverseLinkedListInKGroupsAlternatively(head, 3)).print();
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Smallest String in a tree starting from leaf to root");
//        //https://leetcode.com/problems/smallest-string-starting-from-leaf/
//        TreeNode<Integer> root = new TreeNode<>(0);
//        root.setLeft(new TreeNode<>(1));
//        root.getLeft().setLeft(new TreeNode<>(3));
//        root.getLeft().setRight(new TreeNode<>(4));
//        root.setRight(new TreeNode<>(2));
//        root.getRight().setLeft(new TreeNode<>(3));
//        root.getRight().setRight(new TreeNode<>(4));
//        obj.smallestStringInTreeFromLeafToRoot(root);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Print the sum where each root-to-leaf path of tree represent a number");
//        //https://leetcode.com/problems/sum-root-to-leaf-numbers/
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.setRight(new TreeNode<>(3));
//        obj.printSumWhereRootToLeafPathIsANumber(root);
//        root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.getLeft().getLeft().setLeft(new TreeNode<>(7));
//        root.setRight(new TreeNode<>(3));
//        root.getRight().setRight(new TreeNode<>(6));
//        root.getRight().getRight().setRight(new TreeNode<>(8));
//        obj.printSumWhereRootToLeafPathIsANumber(root);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Max sum path in two sorted arrays");
//        //https://www.geeksforgeeks.org/maximum-sum-path-across-two-arrays/
//        obj.maxSumPathInTwoSortedArrays(new int[]{2, 3, 7, 10, 12}, new int[]{1, 5, 7, 8});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Asteroid collision");
//        //https://leetcode.com/problems/asteroid-collision/
//        obj.asteroidCollision(new int[] {5,10,-5});
//        obj.asteroidCollision(new int[] {8, -8});
//        obj.asteroidCollision(new int[] {10,2,-5});
//        obj.asteroidCollision(new int[] {3,2,1,-10});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest pallindromic substring");
//        //https://leetcode.com/problems/longest-palindromic-substring/
//        obj.longestPallindromicSubstring("racecar");
//        obj.longestPallindromicSubstring("babccaa");
//        obj.longestPallindromicSubstring("aabbaa");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Jump game");
//        //https://leetcode.com/problems/jump-game/
//        obj.jumpGame(new int[]{2, 3, 1, 1, 4});
//        obj.jumpGame(new int[]{3, 2, 1, 0, 4});
//        obj.jumpGame(new int[]{1});
//        obj.jumpGame(new int[]{0});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Next greater element 2 (consider array to cyclic)");
//        //https://leetcode.com/problems/next-greater-element-ii/
//        obj.nextGreaterElement2_CyclicArray(new int[]{1, 2, 1});
//        obj.nextGreaterElement2_CyclicArray(new int[]{5, 4, 3, 2, 1});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Valid sudoku");
//        //https://leetcode.com/problems/valid-sudoku/
//        obj.validSudoku(new String[][]{
//            {"5", "3", ".", ".", "7", ".", ".", ".", "."}, 
//            {"6", ".", ".", "1", "9", "5", ".", ".", "."}, 
//            {".", "9", "8", ".", ".", ".", ".", "6", "."}, 
//            {"8", ".", ".", ".", "6", ".", ".", ".", "3"}, 
//            {"4", ".", ".", "8", ".", "3", ".", ".", "1"}, 
//            {"7", ".", ".", ".", "2", ".", ".", ".", "6"}, 
//            {".", "6", ".", ".", ".", ".", "2", "8", "."}, 
//            {".", ".", ".", "4", "1", "9", ".", ".", "5"}, 
//            {".", ".", ".", ".", "8", ".", ".", "7", "9"}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Diagonal matrix traversal");
//        //https://www.geeksforgeeks.org/print-matrix-diagonal-pattern/
//        obj.diagonalMatrixTraversal(new int[][]{
//            {1, 2, 3},
//            {4, 5, 6},
//            {7, 8, 9}});
//        obj.diagonalMatrixTraversal(new int[][]{
//            {1, 2, 3},
//            {4, 5, 6}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Score of parenthesis");
//        //https://leetcode.com/problems/score-of-parentheses
//        obj.scoreOfParenthesis("()");
//        obj.scoreOfParenthesis("()()");
//        obj.scoreOfParenthesis("(())");
//        obj.scoreOfParenthesis("(()())");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("minimun remove of character to make valid parenthesis");
//        //https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/
//        obj.minimumCharRemovalToMakeValidParenthesis("lee(t(c)o)de)");
//        obj.minimumCharRemovalToMakeValidParenthesis("a)b(c)d");
//        obj.minimumCharRemovalToMakeValidParenthesis(")))(((");
//        obj.minimumCharRemovalToMakeValidParenthesis("()()");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Repeated substring pattern");
//        //https://leetcode.com/problems/repeated-substring-pattern/
//        System.out.println("Repeated substring pattern possible: "+obj.repeatedSubstringPattern("abab"));
//        System.out.println("Repeated substring pattern possible: "+obj.repeatedSubstringPattern("aba"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find median in data stream");
//        //https://leetcode.com/problems/find-median-from-data-stream/
//        obj.findMedianInDataStream(new int[] {5, 15, 1, 3, 2, 8, 7, 9, 10, 6, 11, 4});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Pairs of Songs With Total Durations Divisible by 60");
//        //https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/
//        obj.numPairsDivisibleBy60(new int[]{30,20,150,100,40});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count inversion in the arrays (using modified merge sort)");
//        obj.countInversion(new int[]{1, 20, 6, 4, 5});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum window length where subarray is greater OR equal to target sum");
//        //https://leetcode.com/problems/minimum-size-subarray-sum/
//        obj.minimumWindowSubarrayForTargetSumK(new int[] {2,3,1,2,4,3}, 7);
//        obj.minimumWindowSubarrayForTargetSumK(new int[] {1,4,4}, 4);
//        obj.minimumWindowSubarrayForTargetSumK(new int[] {1,1,1,1,1,1,1,1}, 11);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximum length of repeated subarray (DP PROBLEM)");
//        //https://leetcode.com/problems/maximum-length-of-repeated-subarray/
//        obj.maximumLengthOfRepeatedSubarray_DP_Problem(new int[]{1,2,3,2,1}, new int[]{3,2,1,4,7});
//        obj.maximumLengthOfRepeatedSubarray_DP_Problem(new int[]{0,1,1,1,1,1}, new int[]{0,1,0,1,0,1});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Fix two swapped nodes of the BST");
//        //https://www.geeksforgeeks.org/fix-two-swapped-nodes-of-bst/
//        TreeNode<Integer> root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(5));
//        root.getLeft().setLeft(new TreeNode<>(2));
//        root.getLeft().setRight(new TreeNode<>(20)); //SWAPPED
//        root.setRight(new TreeNode<>(8)); //SWAPPED
//        obj.fixTwoSwappedNodesInBST(root);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Merge two binary trees");
//        //https://leetcode.com/problems/merge-two-binary-trees/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(3));
//        root1.getLeft().setLeft(new TreeNode<>(5));
//        root1.setRight(new TreeNode<>(2));
//        TreeNode<Integer> root2 = new TreeNode<>(2);
//        root2.setLeft(new TreeNode<>(1));
//        root2.getLeft().setRight(new TreeNode<>(4));
//        root2.setRight(new TreeNode<>(3));
//        root2.getRight().setRight(new TreeNode<>(7));
//        obj.mergeTwoBinaryTree(root1, root2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Flip M 0s in binary array and find maximum length of consecutive 1s");
//        obj.flipMZerosFindMaxLengthOfConsecutiveOnes(new int[]{1, 0, 1}, 1);
//        obj.flipMZerosFindMaxLengthOfConsecutiveOnes(new int[]{1, 0, 0, 1, 1, 0, 1, 0, 1, 1}, 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Local minima and local maxima (BINARY SEARCH)");
//        //https://www.geeksforgeeks.org/find-local-minima-array/
//        obj.findLocalMinima(new int[]{1, 2, 3});
//        obj.findLocalMinima(new int[]{23, 8, 15, 2, 3});
//        obj.findLocalMinima(new int[]{9, 6, 3, 14, 5, 7, 4});
//        obj.findLocalMinima(new int[]{3, 2, 1});
//        //MAXIMA
//        obj.findLocalMaxima(new int[]{3, 2, 1});
//        obj.findLocalMaxima(new int[]{1, 2, 3});
//        obj.findLocalMaxima(new int[]{9, 6, 3, 14, 5, 7, 4});
//        obj.findLocalMaxima(new int[]{23, 8, 15, 2, 3});
//        obj.findLocalMaxima(new int[]{23, 8});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Reorder linked list");
//        //https://leetcode.com/problems/reorder-list/
//        Node<Integer> head = new Node<Integer>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.reorderLinkedList(head);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Remove K digits and create the smallest num");
//        //https://leetcode.com/problems/remove-k-digits/
//        obj.removeKDigitsToCreateSmallestNumber("1432219", 3);
//        obj.removeKDigitsToCreateSmallestNumber("10", 2);
//        obj.removeKDigitsToCreateSmallestNumber("10200", 1);
//        obj.removeKDigitsToCreateSmallestNumber("4321", 2);
//        obj.removeKDigitsToCreateSmallestNumber("1234", 2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find the most competetive subsequence of length K in the given array");
//        //https://leetcode.com/problems/find-the-most-competitive-subsequence/
//        obj.findTheMostCompetetiveSubsequenceOfSizeKFromArray(new int[]{3,5,2,6}, 2);
//        obj.findTheMostCompetetiveSubsequenceOfSizeKFromArray(new int[]{2,4,3,3,5,4,9,6}, 4);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Odd even linked list");
//        //https://leetcode.com/problems/odd-even-linked-list/
//        Node<Integer> head = new Node<Integer>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.rearrangeLinkedListAsOddIndexFirstAndEvenIndexAtEnd(head);
//        head = new Node<Integer>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.rearrangeLinkedListAsOddIndexFirstAndEvenIndexAtEnd(head);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Number of ways to create BST and BT with given N (req info: Catalan number series)");
//        //https://leetcode.com/problems/unique-binary-search-trees/
//        //https://www.geeksforgeeks.org/total-number-of-possible-binary-search-trees-with-n-keys/#
//        obj.numberOfWaysToCreateBSTAndBTWithGivenN(3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check if binary tree is complete binary tree or not");
//        //https://leetcode.com/problems/check-completeness-of-a-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().setRight(new TreeNode<>(5));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(6)); //COMPLETE BINARY TREE
//        System.out.println("Is tree complete binary tree: "+obj.checkIfBinaryTreeIsCompleteOrNot(root1));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().setLeft(new TreeNode<>(5));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setRight(new TreeNode<>(7)); //NOT-COMPLETE BINARY TREE
//        System.out.println("Is tree complete binary tree: "+obj.checkIfBinaryTreeIsCompleteOrNot(root1));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("maximum width of binary tree");
//        //https://leetcode.com/problems/maximum-width-of-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(3));
//        root1.getLeft().setLeft(new TreeNode<>(5));
//        root1.getLeft().setRight(new TreeNode<>(3));
//        root1.setRight(new TreeNode<>(2));
//        root1.getRight().setRight(new TreeNode<>(9));
//        obj.maximumWidthOfBinaryTree(root1);
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(3));
//        root1.getLeft().setLeft(new TreeNode<>(5));
//        root1.getLeft().getLeft().setLeft(new TreeNode<>(6));
//        root1.setRight(new TreeNode<>(2));
//        root1.getRight().setRight(new TreeNode<>(9));
//        root1.getRight().getRight().setRight(new TreeNode<>(7));
//        obj.maximumWidthOfBinaryTree(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Sum of elements in matrix except given row and col");
//        //https://www.geeksforgeeks.org/find-sum-of-all-elements-in-a-matrix-except-the-elements-in-given-row-andor-column-2/
//        int[][] rowAndCol = new int[][]{
//            {0, 0},
//            {1, 1},
//            {0, 1}
//        };
//        int[][] matrix = new int[][]{
//            {1, 1, 2},
//            {3, 4, 6},
//            {5, 3, 2}
//        };
//        obj.sumOfElementsInMatrixExceptGivenRowAndCol(matrix, rowAndCol);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count number elements from second array less than or equal to element in first array");
//        //https://www.geeksforgeeks.org/element-1st-array-count-elements-less-equal-2nd-array/
//        obj.countElementsFromSecondArrayLessOrEqualToElementInFirstArray(
//                new int[]{1, 2, 3, 4, 7, 9},
//                new int[]{0, 1, 2, 1, 1, 4});
//        obj.countElementsFromSecondArrayLessOrEqualToElementInFirstArray(
//                new int[]{5, 10, 2, 6, 1, 8, 6, 12},
//                new int[]{6, 5, 11, 4, 2, 3, 7});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check if input binary number stream is divisible by N");
//        //https://www.geeksforgeeks.org/check-divisibility-binary-stream/
//        obj.checkBinaryNumberStreamIsDivisibleByN(new int[]{1,0,1,0,1}, 3);
//        obj.checkBinaryNumberStreamIsDivisibleByN(new int[]{1,0,1}, 5);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Convert number to words");
//        //https://www.geeksforgeeks.org/program-to-convert-a-given-number-to-words-set-2/
//        obj.convertNumberToWords(438237764);
//        obj.convertNumberToWords(0);
//        obj.convertNumberToWords(9);
//        obj.convertNumberToWords(1000);
//        obj.convertNumberToWords(222);
//        obj.convertNumberToWords(1234567891);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Maximum swaps to form greater number by swaping 2 digit atmost ");
        //https://leetcode.com/problems/maximum-swap/
        obj.swapTwoDigitAtMostToFormAGreaterNumber_Greedy(2736);
        obj.swapTwoDigitAtMostToFormAGreaterNumber_Greedy(9973);
        obj.swapTwoDigitAtMostToFormAGreaterNumber_Greedy(1002);
    }

}
