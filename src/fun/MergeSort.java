/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

/**
 *
 * @author RAVI
 */
public class MergeSort {
    
    private static void merging(int[] a, int f, int l, int mid) {
        
        int s1=mid - f + 1;
        int s2 = l - mid;
        
        int[] L = new int[s1];
        int[] R = new int[s2];
        
        for(int i=0; i<s1; i++){
            L[i] = a[f + i];
        }
        
        for(int i=0; i<s2; i++){
            R[i] = a[mid + 1 + i];
        }
        
        
        // Initial indexes of first and second subarrays 
        int i = 0, j = 0; 
  
        // Initial index of merged subarry array 
        int k = f; 
        while (i < s1 && j < s2) { 
            if (L[i] <= R[j]) { 
                a[k] = L[i]; 
                i++; 
            } 
            else { 
                a[k] = R[j]; 
                j++; 
            } 
            k++; 
        } 
  
        /* Copy remaining elements of L[] if any */
        while (i < s1) { 
            a[k] = L[i]; 
            i++; 
            k++; 
        } 
  
        /* Copy remaining elements of R[] if any */
        while (j < s2) { 
            a[k] = R[j]; 
            j++; 
            k++; 
        } 
        
        
    }
    
    public static void mergeSort(int[] a, int f, int l){
        
        if(f<l){
            
            int mid = (f+l)/2;
            
            mergeSort(a, f, mid);
            mergeSort(a, mid+1, l);
            
            merging(a, f, l, mid);
            
        }
        
    }
    
    
    public static void main(String[] args) {
        
        int[] a = {9, 4, 3, 6, 7, 1, 2, 11, 5};
        
        mergeSort(a, 0, a.length-1);
        
        for(int x: a){
            System.out.print(x+" ");
        }
        
    }

    
    
}
